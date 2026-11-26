import math

import torch
from torch import Tensor, IntTensor, BoolTensor
from torch.nn.attention.flex_attention import BlockMask, _mask_mod_signature


def exist(item):
    return item is not None


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.autocast(device_type="cuda", enabled=False)
def get_freqs(dim, max_period=10000.):
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=dim, dtype=torch.float32) / dim
    )
    return freqs


def fractal_flatten(x, rope, cu_seqlens, shape, fractal=False):
    _, height, width = shape
    if fractal:
        pixel_size = 8
        x = local_patching(x, shape, (1, pixel_size, pixel_size), dim=0)
        rope = local_patching(rope, shape, (1, pixel_size, pixel_size), dim=0)
        x = x.flatten(0, 1)
        rope = rope.flatten(0, 1)
    else:
        x = x.flatten(0, 2)
        rope = rope.flatten(0, 2)
    cu_seqlens = cu_seqlens * (height * width)
    return x, rope, cu_seqlens


def fractal_unflatten(x, cu_seqlens, shape, fractal=False):
    length, height, width = shape
    if fractal:
        pixel_size = 8
        x = x.reshape(-1, pixel_size ** 2, *x.shape[1:])
        x = local_merge(x, shape, (1, pixel_size, pixel_size), dim=0)
    else:
        x = x.reshape(*shape, *x.shape[1:])
    cu_seqlens = cu_seqlens // (height * width)
    return x, cu_seqlens


def mean_var_len(x, cu_seqlens, dim=0):
    return torch.cat([
        local_x.mean(dim=dim) for local_x in torch.split(x, torch.diff(cu_seqlens).tolist(), dim=0)
    ], dim=0)


def local_patching(x, shape, group_size, dim=0):
    duration, height, width = shape
    g1, g2, g3 = group_size
    x = x.reshape(*x.shape[:dim], duration//g1, g1, height//g2, g2, width//g3, g3, *x.shape[dim+3:])
    x = x.permute(
        *range(len(x.shape[:dim])),
        dim, dim+2, dim+4, dim+1, dim+3, dim+5,
        *range(dim+6, len(x.shape))
    )
    x = x.flatten(dim, dim+2).flatten(dim+1, dim+3)
    return x


def local_merge(x, shape, group_size, dim=0):
    duration, height, width = shape
    g1, g2, g3 = group_size
    x = x.reshape(*x.shape[:dim], duration//g1, height//g2, width//g3, g1, g2, g3, *x.shape[dim+2:])
    x = x.permute(
        *range(len(x.shape[:dim])),
        dim, dim+3, dim+1, dim+4, dim+2, dim+5,
        *range(dim+6, len(x.shape))
    )
    x = x.flatten(dim, dim+1).flatten(dim+1, dim+2).flatten(dim+2, dim+3)
    return x


def global_patching(x, shape, group_size, dim=0):
    latent_group_size = [axis // axis_group_size for axis, axis_group_size in zip(shape, group_size)]
    x = local_patching(x, shape, latent_group_size, dim)
    x = x.transpose(dim, dim+1)
    return x


def global_merge(x, shape, group_size, dim=0):
    latent_group_size = [axis // axis_group_size for axis, axis_group_size in zip(shape, group_size)]
    x = x.transpose(dim, dim+1)
    x = local_merge(x, shape, latent_group_size, dim)
    return x


@torch.compile()
@torch.no_grad()
def fast_sta_nabla(T: int, H: int, W: int, wT: int = 3, wH: int = 3, wW: int = 3,
             device="cuda") -> Tensor:
    l = torch.Tensor([T, H, W]).max()
    r = torch.arange(0, l, 1, dtype=torch.int16, device=device)
    mat = (r.unsqueeze(1) - r.unsqueeze(0)).abs()
    sta_t, sta_h, sta_w = mat[:T, :T].flatten(), mat[:H, :H].flatten(), mat[:W, :W].flatten()
    sta_t = sta_t <= wT // 2
    sta_h = sta_h <= wH // 2
    sta_w = sta_w <= wW // 2
    sta_hw = (sta_h.unsqueeze(1) * sta_w.unsqueeze(0)).reshape(H, H, W, W).transpose(1, 2).flatten()
    sta = (sta_t.unsqueeze(1) * sta_hw.unsqueeze(0)).reshape(T, T, H*W, H*W).transpose(1, 2)
    return sta.reshape(T*H*W, T*H*W)


@torch.compile(dynamic=True)
@torch.no_grad()
def nablaT_v2_doc(q: Tensor, k: Tensor, seq: Tensor, T: int, H: int, W: int, wT: int = 3,
                  wH: int = 3, wW: int = 3, thr: float = 0.9, add_sta: bool = True,
                  method='topcdf', device="cuda") -> BlockMask:
    assert method == 'topcdf' or method == 'topk', "nabla method should be topcdf or topk"
    # Map estimation
    B, h, S, D = q.shape
    qa = q.reshape(B, h, S // 64, 64, D).mean(-2)
    ka = k.reshape(B, h, S // 64, 64, D).mean(-2).transpose(-2, -1)
    map = qa @ ka

    d = torch.diff(seq)
    doc = torch.eye(d.numel(), dtype=torch.bool, device=device).\
        repeat_interleave(d*H*W, dim=0).repeat_interleave(d*H*W, dim=1)
    map += doc.log()
    map = torch.softmax(map / math.sqrt(D), dim=-1)
    if method == 'topcdf':
        # Map binarization
        vals, inds = map.sort(-1)
        cvals = vals.cumsum_(-1)
        mask = (cvals >= 1 - thr).int()
        mask = mask.gather(-1, inds.argsort(-1))
    else:
        map = map.reshape(B*h*S//64, S//64)
        dl = d.tolist()
        start_row = 0
        mask = torch.zeros_like(map)
        for di in dl:
            d_full = di*W*H*h*B
            end_row = start_row + d_full
            k = max(1, int(thr * di*W*H))
            group = map[start_row : end_row,:]
            _, topk_indices = torch.topk(group, k, dim=-1)
            row_indices = torch.arange(start_row, end_row).view(-1, 1)
            mask[row_indices, topk_indices] = 1
            start_row = end_row
        mask = mask.reshape(B, h, S//64, S//64)
    if add_sta:
        sta = fast_sta_nabla(T, H, W, wT, wH, wW, device=device).unsqueeze_(0).unsqueeze_(0)
        mask = torch.logical_or(mask, sta)
    mask = torch.logical_and(mask, doc)

    # BlockMask creation
    kv_nb = mask.sum(-1).to(torch.int32)
    kv_inds = mask.argsort(dim=-1, descending=True).to(torch.int32)
    return BlockMask.from_kv_blocks(
        torch.zeros_like(kv_nb),
        kv_inds,
        kv_nb,
        kv_inds,
        BLOCK_SIZE=64,
        mask_mod=None
    )


@torch.compile(dynamic=True)
@torch.no_grad()
def nablaT_v2_doc_mfcausal(q: Tensor, k: Tensor, seq: Tensor, T: int, H: int, W: int, wT: int = 3,
                           wH: int = 3, wW: int = 3, thr: float = 0.9, add_sta: bool = True,
                           mf: int = 2, device="cuda") -> BlockMask:
    # Map estimation
    B, h, S, D = q.shape
    qa = q.reshape(B, h, S // 64, 64, D).mean(-2)
    ka = k.reshape(B, h, S // 64, 64, D).mean(-2).transpose(-2, -1)
    map = qa @ ka

    d = torch.diff(seq)
    doc1 = torch.eye(d.numel(), dtype=torch.bool, device=device).\
        repeat_interleave(d, dim=0).repeat_interleave(d, dim=1).tril()
    cl = [[c.sum().item() for c in torch.ones((dd,)).split(mf)] for dd in d]
    cl = torch.Tensor([x for xs in cl for x in xs]).int().cuda()
    doc2 = torch.eye(cl.numel(), dtype=torch.bool, device=device).\
        repeat_interleave(cl, dim=0).repeat_interleave(cl, dim=1)
    doc = torch.logical_or(doc1, doc2).\
        repeat_interleave(H*W, dim=0).repeat_interleave(H*W, dim=1)
    map += doc.log()
    map = torch.softmax(map / math.sqrt(D), dim=-1)

    # Map binarization
    vals, inds = map.sort(-1)
    cvals = vals.cumsum_(-1)
    mask = (cvals >= 1 - thr).int()
    mask = mask.gather(-1, inds.argsort(-1))

    if add_sta:
        sta = fast_sta_nabla(T, H, W, wT, wH, wW, device=device).unsqueeze_(0).unsqueeze_(0)
        mask = torch.logical_or(mask, sta)
    mask = torch.logical_and(mask, doc)

    # BlockMask creation
    kv_nb = mask.sum(-1).to(torch.int32)
    kv_inds = mask.argsort(dim=-1, descending=True).to(torch.int32)
    return BlockMask.from_kv_blocks(
        torch.zeros_like(kv_nb),
        kv_inds,
        kv_nb,
        kv_inds,
        BLOCK_SIZE=64,
        mask_mod=None
    )
