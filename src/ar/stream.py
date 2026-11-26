"""Native causal attention and camera-conditioned KV caching."""

import types
import torch
from .cache import StreamCache


def pack(
    model,
    frames,
    types_,
    positions,
    table,
    indices,
    txt,
    layout,
    geometry=None,
    *,
    use_ucpe=True,
    single_timestep=False,
):
    n, h, w, _ = frames.shape
    dev = frames.device
    if single_timestep:
        mask = (types_ == 1).to(frames.dtype).view(n, 1, 1, 1).expand(n, h, w, 1)
    else:
        mask = torch.zeros(n, h, w, 1, device=dev, dtype=frames.dtype)
        mask[types_ == 1] = 1
    x = torch.cat((frames, torch.zeros_like(frames), mask), -1)
    sp = dict(layout, visual_shape=(n, h, w))
    if single_timestep:
        text_indices = indices[:1]
        sp["carla_single_timestep"] = True
    else:
        boundaries = torch.cat(
            (
                indices.new_zeros(1, dtype=torch.long),
                (indices[1:] != indices[:-1]).nonzero().flatten() + 1,
            )
        )
        ends = torch.cat((boundaries[1:], boundaries.new_tensor([n])))
        text_indices = indices[boundaries]
        sp["rf_text_run_frames"] = (ends - boundaries).int()
    sp["rf_time_idx"] = {"video": indices.int(), "text": text_indices.int()}
    rope = [
        positions,
        torch.arange(h // 2, device=dev),
        torch.arange(w // 2, device=dev),
    ]
    kwargs = dict(scale_factor=(1.0, 2.0, 2.0), sparse_params=sp)
    if geometry is None:
        kwargs["visual_token_type_ids"] = types_
    else:
        mats, ks = geometry
        kwargs["navio_inputs"] = {
            "type_ids": types_,
            "view_ids": torch.zeros_like(types_),
        }
        if use_ucpe:
            kwargs["ucpe_inputs"] = {
                "viewmats": [mats],
                "Ks": [ks],
                "latent_shapes": [(n, h, w)],
                "frame_wh": [(w * 8, h * 8)],
                "trans_scale": [0.0],
            }
    result = model(
        x,
        txt["text"],
        txt["pooled"],
        table * 1000,
        torch.tensor([0, n], device=dev, dtype=torch.int32),
        txt["cu"],
        rope,
        txt["rope"],
        **kwargs,
    )
    return result[0] if isinstance(result, (tuple, list)) else result


class NativeStream:
    def __init__(
        self,
        model,
        history=9,
        backend="flash3",
        poses=None,
        intrinsics=None,
        slots=1,
        gauge_chunks=10,
    ):
        from .native.model.nn import apply_rotary
        from .native.model.attention import attention

        self.model, self.limit, self.backend = model, history, backend
        self.rotate, self.attention = apply_rotary, attention
        self.poses, self.intrinsics = poses, intrinsics
        self.gauge_chunks = gauge_chunks
        self.ids, self.current_ids = [], []
        self.gauge_id, self.generation_chunks = 0, 0
        self.blocks = getattr(model, "visual_transformer_blocks", None)
        if self.blocks is None:
            self.blocks = model.visual_blocks
        self.visual, self.ucpe = {}, {}
        for slot in range(slots):
            self.visual[slot] = StreamCache(len(self.blocks))
            self.ucpe[slot] = StreamCache(len(self.blocks))
            self.ucpe[slot].token_axis = 2
        self.calls = {"read": 0, "commit": 0}
        self.originals = []
        for layer, block in enumerate(self.blocks):
            module = block.self_attention
            original = module.forward
            self.originals.append((module, original))

            def visual_forward(
                module,
                x,
                rope,
                cu_seqlens,
                sparse_params=None,
                _layer=layer,
                _original=original,
            ):
                if not (sparse_params or {}).get("carla_stream"):
                    return _original(x, rope, cu_seqlens, sparse_params)
                return self.visual_attention(module, _layer, x, rope)

            module.forward = types.MethodType(visual_forward, module)
            if hasattr(block, "ucpe_branch"):
                branch = block.ucpe_branch
                for flag in (
                    "use_scale_token",
                    "use_scale_rope",
                    "use_scale_gate",
                    "use_scale_sca",
                    "use_se_2_3_5x5",
                    "use_se_2_3_6x6",
                    "use_se_diff",
                    "use_se_2_3_vnorm",
                    "use_se_2_3_wnorm",
                ):
                    if getattr(branch, flag, False):
                        raise ValueError(f"Unsupported learned UCPE option: {flag}")
                original_u = branch.forward
                self.originals.append((branch, original_u))

                def ucpe_forward(
                    module,
                    x_packed,
                    visual_cu_seqlens,
                    viewmats_per_sample,
                    Ks_per_sample,
                    latent_shapes,
                    frame_wh,
                    trans_scale=None,
                    causal_layout=None,
                    _layer=layer,
                    _original=original_u,
                ):
                    if not (causal_layout or {}).get("carla_stream"):
                        return _original(
                            x_packed,
                            visual_cu_seqlens,
                            viewmats_per_sample,
                            Ks_per_sample,
                            latent_shapes,
                            frame_wh,
                            trans_scale,
                            causal_layout,
                        )
                    return self.ucpe_attention(module, _layer, x_packed)

                branch.forward = types.MethodType(ucpe_forward, branch)

    def geometry(self, ids):
        if self.poses is None:
            return None
        # Move the camera gauge to the last committed pose every ten chunks.
        with torch.autocast(device_type=self.device.type, enabled=False):
            relative = torch.linalg.inv(self.poses[self.gauge_id]) @ self.poses[ids]
            relative = relative.clone()
            centers = relative[:, :3, 3]
            relative[:, :3, 3] = (
                centers
                * (150.0 / centers.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(
                    max=1.0
                )
                / 2.0
            )
            return torch.linalg.inv(relative).float(), self.intrinsics[ids]

    def begin(self, ids, h, w, device, advance_gauge=True):
        self.current_ids, self.h, self.w, self.device = list(ids), h, w, device
        if (
            advance_gauge
            and self.generation_chunks
            and self.generation_chunks % self.gauge_chunks == 0
        ):
            self.gauge_id = self.ids[-1]
        count = len(self.ids)
        start = 0 if getattr(self, "infr_rope", False) else 1
        self.positions = torch.arange(
            count + start, count + start + len(ids), device=device
        )
        self.context_positions = torch.arange(start, count + start, device=device)
        self.spatial = (h // 2) * (w // 2)
        self.context_rope = None
        if count:
            rope = self.model.visual_rope_embeddings(
                (count, h // 2, w // 2),
                [
                    self.context_positions,
                    torch.arange(h // 2, device=device),
                    torch.arange(w // 2, device=device),
                ],
                (1.0, 2.0, 2.0),
            )
            self.context_rope = rope.flatten(0, 2)
        self.current_geometry = self.geometry(self.current_ids)
        self.context_geometry = self.geometry(self.ids) if self.ids else None
        self.functions = {}
        self.context_owner = object()
        for cache in [*self.visual.values(), *self.ucpe.values()]:
            cache.place_for(self.context_owner)

    def visual_attention(self, module, layer, x, rope):
        q, k, v = module.get_qkv(x)
        q, k = module.norm_qk(q, k)
        rotate = getattr(module, "_speedup_rotary", self.rotate)
        raw_k = k
        q, k = rotate(q, rope).type_as(q), rotate(k, rope).type_as(k)
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
        from .native.model.prope import _prepare_apply_fns

        norm = module.input_norm(x).type_as(x)
        project = (
            lambda p: p(norm)
            .view(-1, module.num_heads, module.head_dim)
            .permute(1, 0, 2)[None]
        )
        q, k, v = [project(p) for p in (module.q_proj, module.k_proj, module.v_proj)]
        q, k = module.q_norm(q).type_as(q), module.k_norm(k).type_as(k)
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
        qt, kt, vt = transform(q, qfn), transform(k, kvfn), transform(v, kvfn)
        cache = self.ucpe[self.slot]
        if cache.placed[layer] is None and (raw := cache.readable(layer)) is not None:
            cache.place(layer, transform(raw[0], past[1]), transform(raw[1], past[1]))
        keys, values = (
            cache.join(layer, kt, vt) if cache.placed[layer] is not None else (kt, vt)
        )
        out = self.attention(qt, keys, values, self.backend)
        out = transform(out, ofn)
        if self.mode == "commit":
            cache.append(layer, k.detach().clone(), v.detach().clone())
        return module.out_proj(out[0].permute(1, 0, 2).flatten(1))

    def call(
        self,
        value,
        level,
        text,
        slot=0,
        mode="read",
        anchor=False,
        *,
        differentiable=False,
        use_ucpe=True,
    ):
        if torch.is_grad_enabled() and not (
            differentiable and mode == "read" and not anchor
        ):
            raise RuntimeError(
                "Only an explicit read with immutable KV may enter autograd"
            )
        self.mode, self.slot = mode, slot
        self.calls[mode] += 1
        dev = value.device
        types_ = torch.full((len(value),), int(anchor), device=dev, dtype=torch.long)
        single_timestep = getattr(self, "single_timestep", False)
        if single_timestep and isinstance(level, torch.Tensor):
            level = level.to(device=dev, dtype=torch.float32)
            table = torch.stack((level, level.new_zeros(())))
        else:
            table = torch.as_tensor([level, 0.0], device=dev, dtype=torch.float32)
        return pack(
            self.model,
            value,
            types_,
            self.positions,
            table,
            torch.zeros(len(value), device=dev, dtype=torch.int32),
            text,
            {
                "attention_type": "span_causal",
                "prefix_frames": 0,
                "chunk_frames": len(value),
                "backend": self.backend,
                "carla_stream": True,
            },
            self.current_geometry,
            use_ucpe=use_ucpe,
            single_timestep=single_timestep,
        ).float()

    def committed(self, anchor=False):
        if getattr(self, "infr_rope", False):
            from .rope import first_tail, retain_first_tail

            self.ids = first_tail(self.ids + self.current_ids, self.limit)
            retain_first_tail(
                [*self.visual.values(), *self.ucpe.values()], self.spatial, self.limit
            )
            if not anchor:
                self.generation_chunks += 1
            return
        self.ids = (self.ids + self.current_ids)[-self.limit :]
        for cache in [*self.visual.values(), *self.ucpe.values()]:
            cache.evict(self.limit * self.spatial)
        if not anchor:
            self.generation_chunks += 1
        for slot, cache in self.visual.items():
            assert all(
                cache.context_tokens(i) == len(self.ids) * self.spatial
                for i in range(len(self.blocks))
            )

    def clear(self):
        for cache in [*self.visual.values(), *self.ucpe.values()]:
            cache.context = [[] for _ in range(cache.layers)]
            cache._changed()
        self.ids = []
