"""Exact token sharding over native persistent causal KV, no parameter conversion."""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import types
from .stream import NativeStream


def local_caption_routing(sparse, text_cu):
    """Use one cross-attention sequence on a shard that can split a frame."""
    if text_cu.numel() != 2:
        raise ValueError("AR token sharding supports one caption per solver call")
    if sparse is not None and "rf_text_run_frames" in sparse:
        return {key: value for key, value in sparse.items() if key != "rf_text_run_frames"}
    return sparse


class TokenGroup:
    def __init__(self, group):
        self.group = group
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)

    def bounds(self, total):
        width = (total + self.world - 1) // self.world
        start = min(total, self.rank * width)
        return start, min(total, start + width), width

    def gather(self, value, total, axis=0):
        if self.world == 1:
            return value
        _, _, width = self.bounds(total)
        x = value.movedim(axis, 0).contiguous()
        if len(x) < width:
            x = torch.cat([x, x.new_zeros((width - len(x), *x.shape[1:]))])
        out = x.new_empty((width * self.world, *x.shape[1:]))
        dist.all_gather_into_tensor(out, x, group=self.group)
        return out[:total].movedim(0, axis)

    def kv(self, k, v, total, axis):
        both = self.gather(torch.stack([k, v], -1), total, axis)
        return both[..., 0].contiguous(), both[..., 1].contiguous()


class ParallelNativeStream(NativeStream):
    def __init__(self, *args, group, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokens = TokenGroup(group)
        self.parallel_enabled = True
        self.local_cu = {}
        for index, block in enumerate(self.blocks):
            original = block.forward

            def forward(
                module,
                x,
                text,
                time,
                rope,
                vcu,
                tcu,
                indices,
                sparse,
                geometry,
                *rest,
                _original=original,
                _first=index == 0,
                _last=index == len(self.blocks) - 1,
            ):
                if not self.parallel_enabled:
                    return _original(
                        x, text, time, rope, vcu, tcu, indices, sparse, geometry, *rest
                    )
                if torch.is_grad_enabled():
                    raise RuntimeError("SP stream is inference only")
                total = len(indices)
                start, stop, _ = self.tokens.bounds(total)
                self.sp_span = (start, stop, total)
                geo = geometry
                if geo is not None and geo.get("t_gate") is not None:
                    geo = {**geo, "t_gate": geo["t_gate"][start:stop]}
                cu_key = (stop - start, x.device)
                if cu_key not in self.local_cu:
                    self.local_cu[cu_key] = vcu.new_tensor([0, stop - start])
                value = _original(
                    x[start:stop] if _first else x,
                    text,
                    time,
                    rope,
                    self.local_cu[cu_key],
                    tcu,
                    indices[start:stop],
                    local_caption_routing(sparse, tcu),
                    geo,
                    *rest,
                )
                # Each block consumes the same local token interval. Attention
                # gathers global KV separately. The output head sees all tokens.
                return self.tokens.gather(value, total) if _last else value

            block.forward = types.MethodType(forward, block)
            self.originals.append((block, original))

    def reset_all(self):
        self.clear()
        self.gauge_id = 0
        self.generation_chunks = 0
        self.calls = {"read": 0, "commit": 0}

    def borrow_read_context(self, base):
        for name in (
            "ids",
            "visual",
            "ucpe",
            "positions",
            "context_positions",
            "context_rope",
            "current_geometry",
            "context_geometry",
            "context_owner",
            "spatial",
            "h",
            "w",
            "device",
        ):
            setattr(self, name, getattr(base, name))
        self.functions = {}

    def visual_attention(self, module, layer, x, rope):
        if not self.parallel_enabled:
            return super().visual_attention(module, layer, x, rope)
        start, stop, total = self.sp_span
        q, k, v = module.get_qkv(x)
        q, k = module.norm_qk(q, k)
        k, v = self.tokens.kv(k, v, total, 0)
        raw_k = k
        rotate = getattr(module, "_speedup_rotary", self.rotate)
        q = rotate(q, rope[start:stop]).type_as(q)
        k = rotate(k, rope).type_as(k)
        cache = self.visual[self.slot]
        if cache.placed[layer] is None and (raw := cache.readable(layer)) is not None:
            cache.place(
                layer, rotate(raw[0], self.context_rope).type_as(raw[0]), raw[1]
            )
        keys, values = (
            cache.join(layer, k, v) if cache.placed[layer] is not None else (k, v)
        )
        out = (
            self.attention(
                q.permute(1, 0, 2)[None],
                keys.permute(1, 0, 2)[None],
                values.permute(1, 0, 2)[None],
                self.backend,
            )[0]
            .permute(1, 0, 2)
            .flatten(1)
        )
        if self.mode == "commit":
            cache.append(layer, raw_k.detach().clone(), v.detach().clone())
        return module.out_layer(out)

    def ucpe_attention(self, module, layer, x):
        if not self.parallel_enabled:
            return super().ucpe_attention(module, layer, x)
        from .native.model.prope import _prepare_apply_fns

        start, stop, total = self.sp_span
        norm = module.input_norm(x).type_as(x)
        project = (
            lambda p: p(norm)
            .view(-1, module.num_heads, module.head_dim)
            .permute(1, 0, 2)[None]
        )
        q, k, v = [project(p) for p in (module.q_proj, module.k_proj, module.v_proj)]
        q, k = module.q_norm(q).type_as(q), module.k_norm(k).type_as(k)
        k, v = self.tokens.kv(k, v, total, 2)
        if module.head_dim not in self.functions:

            def make(geometry):
                if geometry is None:
                    return None
                mats, ks = geometry
                return _prepare_apply_fns(
                    head_dim=module.head_dim,
                    viewmats=mats[None],
                    Ks=ks[None],
                    patches_x=self.w // 2,
                    patches_y=self.h // 2,
                    image_width=self.w * 8,
                    image_height=self.h * 8,
                )

            self.functions[module.head_dim] = (
                make(self.current_geometry),
                make(self.context_geometry),
            )
        (qfn, kvfn, ofn), past = self.functions[module.head_dim]
        transform = lambda value, fn: module._downscale_to_reference_rms(
            value, fn(value).to(value.dtype)
        )

        def local(value, fn):
            padded = F.pad(value, (0, 0, start, total - stop))
            result = fn(padded)[:, :, start:stop].to(value.dtype)
            return module._downscale_to_reference_rms(value, result)

        qt, kt, vt = local(q, qfn), transform(k, kvfn), transform(v, kvfn)
        cache = self.ucpe[self.slot]
        if cache.placed[layer] is None and (raw := cache.readable(layer)) is not None:
            cache.place(layer, transform(raw[0], past[1]), transform(raw[1], past[1]))
        keys, values = (
            cache.join(layer, kt, vt) if cache.placed[layer] is not None else (kt, vt)
        )
        out = local(self.attention(qt, keys, values, self.backend), ofn)
        if self.mode == "commit":
            cache.append(layer, k.detach().clone(), v.detach().clone())
        return module.out_proj(out[0].permute(1, 0, 2).flatten(1))
