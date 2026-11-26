import math

import torch
from torch import nn
from flash_attn import flash_attn_varlen_func


@torch.autocast(device_type="cuda", enabled=False)
def get_freqs(dim, max_period=10000.):
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=dim, dtype=torch.float32) / dim
    )
    return freqs


@torch.autocast(device_type="cuda", enabled=False)
def apply_scale_shift_norm(norm, x, scale, shift, idx):
    return norm(x) * (scale.index_select(0, idx) + 1.0) + shift.index_select(0, idx)


@torch.autocast(device_type="cuda", enabled=False)
def apply_gate_sum(x, out, gate, idx):
    return x + gate.index_select(0, idx) * out


@torch.autocast(device_type="cuda", enabled=False)
def apply_rotary(x, rope):
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2).to(torch.float32)
    x_out = rope[..., 0] * x_[..., 0] + rope[..., 1] * x_[..., 1]
    return x_out.reshape(*x.shape)


class TimeEmbeddings(nn.Module):

    def __init__(self, model_dim, time_dim, max_period=10000.):
        super().__init__()
        assert model_dim % 2 == 0
        self.model_dim = model_dim
        self.max_period = max_period
        self.register_buffer('freqs', get_freqs(model_dim // 2, max_period), persistent=False)

        self.in_layer = nn.Linear(model_dim, time_dim, bias=True)
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, time_dim, bias=True)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, time_arg):
        args = torch.outer(time_arg, self.freqs.to(device=time_arg.device))
        time_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        time_embed = self.out_layer(self.activation(self.in_layer(time_embed.to(self.in_layer.weight.dtype))))
        time_embed_idx = torch.arange(time_embed.shape[0], device=time_embed.device, dtype=torch.int32)
        return time_embed, time_embed_idx


class TextEmbeddings(nn.Module):

    def __init__(self, text_dim, model_dim):
        super().__init__()
        self.in_layer = nn.Linear(text_dim, model_dim, bias=True)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, text_embed):
        text_embed = self.in_layer(text_embed)
        return self.norm(text_embed).type_as(text_embed)


class VisualEmbeddings(nn.Module):

    def __init__(self, visual_dim, model_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = nn.Linear(math.prod(patch_size) * visual_dim, model_dim)

    def forward(self, x, visual_cu_seqlens):
        if self.patch_size[0] > 1:
            idxs = torch.ones(x.shape[0], dtype=torch.int32, device=visual_cu_seqlens.device)
            idxs[visual_cu_seqlens[:-1]] += self.patch_size[0] - 1
            x = torch.repeat_interleave(x, idxs, dim=0)
            visual_cu_seqlens = visual_cu_seqlens + torch.arange(visual_cu_seqlens.shape[0], device=visual_cu_seqlens.device, dtype=torch.int32)
        if len(x.shape) == 4:
            duration, height, width, dim = x.shape

            x = x.view(
                duration // self.patch_size[0], self.patch_size[0],
                height // self.patch_size[1], self.patch_size[1],
                width // self.patch_size[2], self.patch_size[2], dim
            ).permute(0, 2, 4, 1, 3, 5, 6).flatten(3, 6)
        elif len(x.shape) == 2:
            duration, dim = x.shape
        visual_cu_seqlens = visual_cu_seqlens // self.patch_size[0]
        return self.in_layer(x), visual_cu_seqlens


class RoPE1D(nn.Module):

    def __init__(self, dim, max_pos=2048, max_period=10000., freqs_scaling=1.):
        super().__init__()
        self.max_period = max_period
        self.dim = dim
        self.max_pos = max_pos
        self.freqs_scaling = freqs_scaling
        freq = get_freqs(dim // 2, max_period) * freqs_scaling
        pos = torch.arange(max_pos, dtype=freq.dtype)
        self.register_buffer(f'args', torch.outer(pos, freq), persistent=False)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, pos):
        args = self.args[pos]
        rope = torch.stack([torch.cos(args), -torch.sin(args), torch.sin(args), torch.cos(args)], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class RoPE3D(nn.Module):

    def __init__(self, axes_dims, max_pos=(128, 128, 128), max_period=10000.):
        super().__init__()
        self.axes_dims = axes_dims
        self.max_pos = max_pos
        self.max_period = max_period

        for i, (axes_dim, ax_max_pos) in enumerate(zip(axes_dims, max_pos)):
            freq = get_freqs(axes_dim // 2, max_period)
            pos = torch.arange(ax_max_pos, dtype=freq.dtype)
            self.register_buffer(f'args_{i}', torch.outer(pos, freq), persistent=False)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, shape, pos, scale_factor=(1., 1., 1.)):
        duration, height, width = shape
        args_t = self.args_0[pos[0]] / scale_factor[0]
        args_h = self.args_1[pos[1]] / scale_factor[1]
        args_w = self.args_2[pos[2]] / scale_factor[2]

        args = torch.cat([
            args_t.view(duration, 1, 1, -1).repeat(1, height, width, 1),
            args_h.view(1, height, 1, -1).repeat(duration, 1, width, 1),
            args_w.view(1, 1, width, -1).repeat(duration, height, 1, 1)
        ], dim=-1)
        rope = torch.stack([torch.cos(args), -torch.sin(args), torch.sin(args), torch.cos(args)], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class Modulation(nn.Module):

    def __init__(self, time_dim, model_dim, num_params):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, num_params * model_dim)
        self.out_layer.weight.data.zero_()
        self.out_layer.bias.data.zero_()

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, x):
        return self.out_layer(self.activation(x))


class MultiheadSelfAttention(nn.Module):

    def __init__(self, num_channels, head_dim):
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)


    def get_qkv(self, x):
        query = self.to_query(x)
        key = self.to_key(x)
        value = self.to_value(x)

        shape = query.shape[:-1] # for TP compatibility
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*shape, self.num_heads, -1)
        value = value.reshape(*shape, self.num_heads, -1)

        return query, key, value

    def norm_qk(self, q, k):
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        return q, k

    def scaled_dot_product_attention(self, query, key, value, cu_seqlens):

        max_seqlen = torch.diff(cu_seqlens).max()
        out = flash_attn_varlen_func(query, key, value, cu_seqlens, cu_seqlens,
                                     int(max_seqlen), int(max_seqlen))
        return out.flatten(-2)

    def forward(self, x, rope, cu_seqlens):
        query, key, value = self.get_qkv(x)
        query, key = self.norm_qk(query, key)
        query = apply_rotary(query, rope).type_as(query)
        key = apply_rotary(key, rope).type_as(key)

        out = self.scaled_dot_product_attention(query, key, value, cu_seqlens)

        out = self.out_layer(out)
        return out


class MultiheadCrossAttention(nn.Module):


    def __init__(self, q_dim, head_dim, kv_dim=None):
        super().__init__()
        if kv_dim is None:
            kv_dim = q_dim
        assert q_dim % head_dim == 0
        self.num_heads = q_dim // head_dim

        self.to_query = nn.Linear(q_dim, q_dim)
        self.to_key   = nn.Linear(kv_dim, q_dim)
        self.to_value = nn.Linear(kv_dim, q_dim)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)
        self.out_layer = nn.Linear(q_dim, q_dim)

    def get_qkv(self, x, cond):
        query = self.to_query(x)
        key = self.to_key(cond)
        value = self.to_value(cond)
        shape, cond_shape = query.shape[:-1], key.shape[:-1] # change order for TP compatibility
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*cond_shape, self.num_heads, -1)
        value = value.reshape(*cond_shape, self.num_heads, -1)

        return query, key, value

    def norm_qk(self, q, k):
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        return q, k

    def scaled_dot_product_attention(self, query, key, value, cu_seqlens, cond_cu_seqlens):
        max_seqlen = torch.diff(cu_seqlens).max()
        cond_max_seqlen = torch.diff(cond_cu_seqlens).max()
        out = flash_attn_varlen_func(query, key, value, cu_seqlens, cond_cu_seqlens,
                                     int(max_seqlen), int(cond_max_seqlen))
        return out.flatten(-2)

    def forward(self, x, cond, cu_seqlens, cond_cu_seqlens, rope_q=None, rope_kv=None, routing=None):
        query, key, value = self.get_qkv(x, cond)
        query, key = self.norm_qk(query, key)

        if rope_q is not None:
            query = apply_rotary(query, rope_q).type_as(query)
        if rope_kv is not None:
            key = apply_rotary(key, rope_kv).type_as(key)

        if routing is None:
            out = self.scaled_dot_product_attention(query, key, value, cu_seqlens, cond_cu_seqlens)
        else:
            out = routing.attention(query, key, value).flatten(-2)
        out = self.out_layer(out)
        return out


class FeedForward(nn.Module):

    def __init__(self, dim, ff_dim):
        super().__init__()
        self.in_layer = nn.Linear(dim, ff_dim, bias=False)
        self.activation = nn.GELU()
        self.out_layer = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x):
        return self.out_layer(self.activation(self.in_layer(x)))


class OutLayer(nn.Module):

    def __init__(self, model_dim, time_dim, visual_dim, patch_size, video_mod=True):
        super().__init__()
        self.patch_size = patch_size
        self.modulation = Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = nn.Linear(model_dim, math.prod(patch_size) * visual_dim, bias=True)
        self.video_mod = video_mod

    def forward(self, visual_embed, text_embed, time_embed, visual_cu_seqlens, time_embed_idx):
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        if self.video_mod:
            scale = scale[:, None, None]
            shift = shift[:, None, None]
        visual_embed = apply_scale_shift_norm(
            self.norm, visual_embed, scale, shift, time_embed_idx
        ).type_as(visual_embed)
        x = self.out_layer(visual_embed)
        if len(x.shape) == 4:
            duration, height, width, dim = x.shape
            x = x.view(
                duration, height, width, # t, h, w, c, p1, p2, p3 => t p1 h p2 w p3 c
                -1, self.patch_size[0], self.patch_size[1], self.patch_size[2]
            ).permute(0, 4, 1, 5, 2, 6, 3).flatten(0, 1).flatten(1, 2).flatten(2, 3)
        elif len(x.shape) == 2:
            duration, dim = x.shape
            x = x.view(
                duration,  # t, c, p1,  => t p1 c
                -1, self.patch_size[0],
            ).permute(0, 2, 1).flatten(0, 1)
        visual_cu_seqlens = visual_cu_seqlens * self.patch_size[0]

        if self.patch_size[0] > 1:
            idxs = torch.ones(duration * self.patch_size[0], dtype=torch.int32, device=visual_cu_seqlens.device)
            idxs[visual_cu_seqlens[:-1]] -= self.patch_size[0] - 1
            x = torch.repeat_interleave(x, idxs, dim=0)
        return x
