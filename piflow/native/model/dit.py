import torch
from torch import nn
from ..navio_conditioning import reorder_tokens

from .logger import ModelLogger
from .nan_debug import begin_forward as _nan_begin_forward, check as _nan_check, trace_stats as _nan_trace
from . import act_collect
from .nn import (
    FeedForward,
    Modulation,
    MultiheadCrossAttention,
    MultiheadSelfAttention,
    OutLayer,
    PluckerEmbeddings,
    RoPE1D,
    RoPE3D,
    TextEmbeddings,
    TimeEmbeddings,
    VisualEmbeddings,
    apply_gate_sum,
    apply_scale_shift_norm,
)
from .ucpe import (
    UcpeSideBranch,
    expand_per_frame,
    normalize_se_pose,
    validate_ucpe_attention_config,
)
from .rayrope import RayRoPESideBranch
from .raymap import RayMapSideBranch, build_raymap_input
from .crepe import CRePESideBranch
from .utils import fractal_flatten, fractal_unflatten

_INHERIT_RAYMAP = object()


def _expand_rf_text_runs(text_embed, text_rope_pos, text_cu_seqlens, sparse_params):
    """Duplicate one caption for each contiguous video timestep run."""
    runs = (sparse_params or {}).get("rf_text_run_frames")
    if runs is None:
        return text_embed, text_rope_pos, text_cu_seqlens
    runs = runs.to(device=text_embed.device, dtype=torch.int32)
    if text_cu_seqlens.numel() != 2:
        raise ValueError("RF per-level text routing requires exactly one input caption")
    n = int(runs.numel())
    length = int((text_cu_seqlens[1] - text_cu_seqlens[0]).item())
    if n <= 0 or length <= 0:
        raise ValueError("RF per-level text routing requires non-empty runs and caption")
    text_embed = text_embed.repeat((n,) + (1,) * (text_embed.ndim - 1))
    text_rope_pos = text_rope_pos.repeat(n)
    text_cu_seqlens = torch.arange(
        n + 1, device=text_cu_seqlens.device, dtype=torch.int32
    ) * length
    return text_embed, text_rope_pos, text_cu_seqlens


def _rf_cross_cu_seqlens(total_tokens, sparse_params, device):
    """Visual query segments paired with per-level text segments for cross-attention."""
    runs = (sparse_params or {}).get("rf_text_run_frames")
    if runs is None:
        return None
    runs = runs.to(device=device, dtype=torch.int64)
    total_frames = int(runs.sum().item())
    if total_frames <= 0 or total_tokens % total_frames:
        raise ValueError(
            f"invalid RF text routing: tokens={total_tokens}, frames={total_frames}"
        )
    tokens_per_frame = total_tokens // total_frames
    return torch.cat([
        torch.zeros(1, device=device, dtype=torch.int32),
        torch.cumsum(runs * tokens_per_frame, dim=0).to(torch.int32),
    ])


class TransformerEncoderBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim):
        super().__init__()
        self.text_modulation = Modulation(time_dim, model_dim, 6)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttention(model_dim, head_dim)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def forward(self, x, time_embed, rope, cu_seqlens, time_embed_idx):
        apply_scale_shift_norm = getattr(self, "_speedup_modulate", globals()["apply_scale_shift_norm"])
        apply_gate_sum = getattr(self, "_speedup_gate", globals()["apply_gate_sum"])
        self_attn_params, ff_params = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)

        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = apply_scale_shift_norm(self.self_attention_norm, x, scale, shift, time_embed_idx).type_as(x)
        out = self.self_attention(out, rope, cu_seqlens)
        x = apply_gate_sum(x, out, gate, time_embed_idx).type_as(x)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = apply_scale_shift_norm(self.feed_forward_norm, x, scale, shift, time_embed_idx).type_as(x)
        out = self.feed_forward(out)
        x = apply_gate_sum(x, out, gate, time_embed_idx).type_as(x)
        return x

    def reset_parameters(self):
        self.text_modulation.reset_parameters()
        self.self_attention_norm.reset_parameters()
        self.self_attention.reset_parameters()
        self.feed_forward_norm.reset_parameters()
        self.feed_forward.reset_parameters()


@ModelLogger.log_transformer_block
class TransformerDecoderBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim, use_ucpe: bool = True, use_plucker_mixing: bool = False,
                 use_rayrope: bool = False, rayrope_num_rays: int = 1, rayrope_freq_base: float = 3.0,
                 use_crepe: bool = False, crepe_num_rays: int = 1, crepe_freq_base: float = 3.0,
                 crepe_num_breakpoints: int = 8, use_scale_token: bool = False,
                 use_scale_rope: bool = False, use_scale_gate: bool = False,
                 use_scale_sca: bool = False, use_se_2_3_5x5: bool = False,
                 use_se_2_3_6x6: bool = False, use_se_2_3_vnorm: bool = False,
                 use_se_2_3_wnorm: bool = False, use_se_diff: bool = False,
                 ucpe_attn_mask: str = None, ucpe_chunk_len: int = 3,
                 use_raymap: bool = False, raymap_attn_mask: str = None, raymap_chunk_len: int = 3):
        super().__init__()
        # Optional SE_2(3)/6x6 velocity + angular-velocity normalization (feature-gated,
        # UCPE-only). Stored on the block because the normalize + output-scale happen in
        # this block's forward. use_se_diff = differentiated-extrinsic velocity method (4x4;
        # viewmats built in the dataloader, magnitudes re-injected here).
        self.use_se_2_3_5x5 = use_se_2_3_5x5
        self.use_se_2_3_6x6 = use_se_2_3_6x6
        self.use_se_2_3_vnorm = use_se_2_3_vnorm
        self.use_se_2_3_wnorm = use_se_2_3_wnorm
        self.use_se_diff = use_se_diff
        self.visual_modulation = Modulation(time_dim, model_dim, 9)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttention(model_dim, head_dim)

        self.cross_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.cross_attention = MultiheadCrossAttention(model_dim, head_dim)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

        # UCPE side branch: ray-local PRoPE Q/K/V transform with zero-init out_proj.
        # Step-0 residual is identically zero, so the frozen base model passes
        # through unchanged until UCPE learns a non-zero geometric mix. Built
        # conditionally on use_ucpe so the yaml can toggle UCPE off (e.g.
        # plucker-only ablation) without dead-weight params on the GPU.
        if use_ucpe:
            self.ucpe_branch = UcpeSideBranch(model_dim=model_dim, head_dim=head_dim,
                                              use_scale_token=use_scale_token,
                                              use_scale_rope=use_scale_rope,
                                              use_scale_gate=use_scale_gate,
                                              use_scale_sca=use_scale_sca,
                                              use_se_2_3_5x5=use_se_2_3_5x5,
                                              use_se_2_3_6x6=use_se_2_3_6x6,
                                              use_se_2_3_vnorm=use_se_2_3_vnorm,
                                              use_se_2_3_wnorm=use_se_2_3_wnorm,
                                              ucpe_attn_mask=ucpe_attn_mask,
                                              ucpe_chunk_len=ucpe_chunk_len)
        if use_raymap:
            self.raymap_branch = RayMapSideBranch(
                model_dim, head_dim, attn_mask=raymap_attn_mask,
                chunk_len=raymap_chunk_len,
            )

        # RayRoPE side branch: depth-free global-ray rotary Q/K/V encoding with
        # zero-init out_proj. Same residual contract as ucpe_branch (step-0
        # output is identically zero). Mutually exclusive with UCPE in practice
        # (use_rayrope: true => use_ucpe: false); both consume the same
        # per-frame `ucpe_inputs` trajectory (viewmats + Ks).
        if use_rayrope:
            self.rayrope_branch = RayRoPESideBranch(
                model_dim=model_dim,
                head_dim=head_dim,
                num_rays_per_patch=rayrope_num_rays,
                freq_base=rayrope_freq_base,
                use_scale_token=use_scale_token,
            )

        # CRePE side branch: RayRoPE-style ray rotary + a learned per-token
        # radial-distance head whose expected RoPE places points along the ray
        # (arXiv 2605.12938). Same zero-init residual contract. Built only on the
        # blocks the DiT selects (paper applies CRePE to middle layers only), so
        # use_crepe is passed per-block by DiffusionTransformer3D.
        if use_crepe:
            self.crepe_branch = CRePESideBranch(
                model_dim=model_dim,
                head_dim=head_dim,
                num_rays_per_patch=crepe_num_rays,
                freq_base=crepe_freq_base,
                num_breakpoints=crepe_num_breakpoints,
                use_scale_token=use_scale_token,
            )

        # Plücker mixing fine-rate camera residual (SANA-WM §3.3 Fine Branch).
        # Sums `plucker_proj(plucker_emb)` into the running stream right after
        # the self-attention residual — same insertion site as ucpe_branch.
        # Zero-init weight + bias=False ensures step-0 forward is identical to
        # the baseline regardless of `plucker_emb`. SANA-WM uses bias=True with
        # zero bias; we drop bias entirely because a zero-weight + zero-bias
        # linear would still let the bias drift off zero on step 0 (its grad is
        # dL/d(out), independent of the camera input), learning a per-block
        # scalar shift that carries no geometric signal.
        if use_plucker_mixing:
            self.plucker_proj = nn.Linear(model_dim, model_dim, bias=False)
            nn.init.zeros_(self.plucker_proj.weight)

    def forward(
        self,
        visual_embed,
        text_embed,
        time_embed,
        rope,
        visual_cu_seqlens,
        text_cu_seqlens,
        time_embed_idx,
        sparse_params,
        ucpe_inputs,
        plucker_emb=None,
        external_visual_residual_controller=None,
        raymap_emb=None,
        raymap_gate=None,
        raymap_shapes=None,
    ):
        apply_scale_shift_norm = getattr(self, "_speedup_modulate", globals()["apply_scale_shift_norm"])
        apply_gate_sum = getattr(self, "_speedup_gate", globals()["apply_gate_sum"])
        self_attn_params, cross_attn_params, ff_params = torch.chunk(self.visual_modulation(time_embed), 3, dim=-1)
        # _nan_check(self_attn_params, "block.modulation.self_attn_params")
        # _nan_check(cross_attn_params, "block.modulation.cross_attn_params")
        # _nan_check(ff_params, "block.modulation.ff_params")

        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(self.self_attention_norm, visual_embed, scale, shift, time_embed_idx).type_as(visual_embed)
        # _nan_check(visual_out, "block.after_self_attention_norm")
        visual_out = self.self_attention(visual_out, rope, visual_cu_seqlens, sparse_params)
        # _nan_check(visual_out, "block.after_self_attention")
        _sa_in = visual_embed if act_collect.enabled() else None
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate, time_embed_idx).type_as(visual_embed)
        if _sa_in is not None:
            act_collect.record("self_attn", self._block_idx, visual_embed - _sa_in)
        # _nan_check(visual_embed, "block.after_self_attention_residual")

        # UCPE side-branch residual. Mirrors the plucker `hasattr` guard so
        # use_ucpe=False yaml drops the branch entirely; passing ucpe_inputs
        # on such a model is a no-op (same shape but no projection).
        if (
            ucpe_inputs is not None
            and hasattr(self, "ucpe_branch")
            and getattr(self, "_ucpe_runtime_enabled", True)
        ):
            # Optional velocity/angular-velocity-norm mode: feed direction-only columns to
            # the branch (unit-normalized v [and w for 6x6]) and rescale the branch output
            # by the re-injected per-frame magnitude. Symmetric over 5x5/6x6:
            #   wnorm            -> normalize v, add ||w||; scale by v_coef*||v|| + w_coef*||w||
            #                       (5x5 has no w column, so ||w|| is computed from the rotation
            #                        block in normalize_se_pose; matrix stays 5x5)
            #   vnorm (no wnorm) -> normalize v; scale by v_coef*||v||
            #   neither          -> byte-identical no-op.
            _viewmats = ucpe_inputs["viewmats"]
            _v_scale, _w_scale = None, None
            if self.use_se_diff:
                # Velocity method: viewmats already carry the (normalized) T_diff from the
                # dataloader, so no normalize_se_pose here. The per-frame ||p'||/||R'||_F
                # magnitudes come precomputed in ucpe_inputs; expand them to per-token.
                _v_scale = expand_per_frame(
                    ucpe_inputs.get("se_diff_vnorm"), ucpe_inputs["latent_shapes"],
                    device=visual_embed.device, dtype=visual_embed.dtype)
                _w_scale = expand_per_frame(
                    ucpe_inputs.get("se_diff_wnorm"), ucpe_inputs["latent_shapes"],
                    device=visual_embed.device, dtype=visual_embed.dtype)
            else:
                _ext = self.use_se_2_3_5x5 or self.use_se_2_3_6x6
                _norm_v = _ext and (self.use_se_2_3_vnorm or self.use_se_2_3_wnorm)
                _norm_w = _ext and self.use_se_2_3_wnorm
                if _norm_v or _norm_w:
                    _viewmats, _v_scale, _w_scale = normalize_se_pose(
                        ucpe_inputs["viewmats"],
                        ucpe_inputs["latent_shapes"],
                        device=visual_embed.device,
                        dtype=visual_embed.dtype,
                        norm_v=_norm_v,
                        norm_w=_norm_w,
                    )
            navio = ucpe_inputs.get("navio")
            if (sparse_params or {}).get("attention_type") in {"doubled_causal", "span_causal"}:
                navio = None  # RF/TF geometry and tokens already share the packed order.
            branch_input = (
                reorder_tokens(visual_embed, navio["inverse"])
                if navio is not None else visual_embed
            )
            ucpe_out = self.ucpe_branch(
                branch_input,
                visual_cu_seqlens=visual_cu_seqlens,
                viewmats_per_sample=_viewmats,
                Ks_per_sample=ucpe_inputs["Ks"],
                latent_shapes=ucpe_inputs["latent_shapes"],
                frame_wh=ucpe_inputs["frame_wh"],
                trans_scale=ucpe_inputs.get("trans_scale"),
                causal_layout=sparse_params if (sparse_params or {}).get("attention_type") in {"doubled_causal", "span_causal"} else None,
            ).type_as(visual_embed)
            # Re-inject magnitude with trainable per-block coefs: v_coef*||v|| (+ w_coef*||w||).
            _scale = None
            if _v_scale is not None:
                _scale = self.ucpe_branch.v_coef * _v_scale
            if _w_scale is not None:
                _wpart = self.ucpe_branch.w_coef * _w_scale
                _scale = _wpart if _scale is None else _scale + _wpart
            if _scale is not None:
                ucpe_out = ucpe_out * _scale
            if navio is not None:
                ucpe_out = reorder_tokens(ucpe_out, navio["permutation"])
            # Timestep gate (per-token, precomputed in DiT.forward from ucpe_t): zeroes the
            # UCPE residual outside the noise window => pure self-attention baseline there.
            _tg = ucpe_inputs.get("t_gate")
            if _tg is not None:
                ucpe_out = ucpe_out * _tg
            # _nan_check(ucpe_out, "block.ucpe_branch.output")
            if act_collect.enabled():
                act_collect.record("ucpe", self._block_idx, ucpe_out)
            visual_embed = visual_embed + ucpe_out
            # _nan_check(visual_embed, "block.after_ucpe_residual")

        # Script-scoped plug-and-play residual. Normal training and inference do
        # not attach a controller, so this branch is a strict no-op by default.
        if external_visual_residual_controller is not None:
            external_residual = external_visual_residual_controller.residual_for_block(
                self._block_idx,
                visual_embed,
                visual_cu_seqlens,
            )
            if external_residual is not None:
                if external_residual.shape != visual_embed.shape:
                    raise ValueError(
                        f"external residual shape {tuple(external_residual.shape)} != "
                        f"visual_embed shape {tuple(visual_embed.shape)} at block {self._block_idx}"
                    )
                visual_embed = visual_embed + external_residual.type_as(visual_embed)

        # RayRoPE side-branch residual. Same insertion site + same inputs as
        # the UCPE residual; guarded by hasattr so use_rayrope=False configs
        # drop the branch entirely. UCPE and RayRoPE are mutually exclusive in
        # the yaml, so at most one of these two blocks fires.
        if ucpe_inputs is not None and hasattr(self, "rayrope_branch"):
            rayrope_out = self.rayrope_branch(
                visual_embed,
                visual_cu_seqlens=visual_cu_seqlens,
                viewmats_per_sample=ucpe_inputs["viewmats"],
                Ks_per_sample=ucpe_inputs["Ks"],
                latent_shapes=ucpe_inputs["latent_shapes"],
                frame_wh=ucpe_inputs["frame_wh"],
                trans_scale=ucpe_inputs.get("trans_scale"),
            ).type_as(visual_embed)
            visual_embed = visual_embed + rayrope_out

        # CRePE side-branch residual. Same site + inputs as RayRoPE; present only
        # on the middle blocks selected at build time (hasattr guard).
        if ucpe_inputs is not None and hasattr(self, "crepe_branch"):
            crepe_out = self.crepe_branch(
                visual_embed,
                visual_cu_seqlens=visual_cu_seqlens,
                viewmats_per_sample=ucpe_inputs["viewmats"],
                Ks_per_sample=ucpe_inputs["Ks"],
                latent_shapes=ucpe_inputs["latent_shapes"],
                frame_wh=ucpe_inputs["frame_wh"],
                trans_scale=ucpe_inputs.get("trans_scale"),
            ).type_as(visual_embed)
            visual_embed = visual_embed + crepe_out

        # Plücker fine-rate residual: pre-patched, shared across blocks. Same
        # insertion site as the UCPE residual; SANA-WM §3.3 places this
        # "immediately after each self-attention output, preserving the
        # pretrained model at initialization".
        if plucker_emb is not None and hasattr(self, "plucker_proj"):
            visual_embed = visual_embed + self.plucker_proj(plucker_emb).type_as(visual_embed)

        if raymap_emb is not None and hasattr(self, "raymap_branch"):
            navio = ucpe_inputs.get("navio") if ucpe_inputs is not None else None
            raymap_visual = visual_embed
            raymap_tokens = raymap_emb
            if navio is not None:
                raymap_visual = reorder_tokens(raymap_visual, navio["inverse"])
                raymap_tokens = reorder_tokens(raymap_tokens, navio["inverse"])
            raymap_out = self.raymap_branch(
                raymap_visual, raymap_tokens, visual_cu_seqlens, raymap_shapes
            ).type_as(visual_embed)
            if navio is not None:
                raymap_out = reorder_tokens(raymap_out, navio["permutation"])
            if raymap_gate is not None:
                raymap_out = raymap_out * raymap_gate
            visual_embed = visual_embed + raymap_out

        shift, scale, gate = torch.chunk(cross_attn_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(self.cross_attention_norm, visual_embed, scale, shift, time_embed_idx).type_as(visual_embed)
        # _nan_check(visual_out, "block.after_cross_attention_norm")
        cross_cu = _rf_cross_cu_seqlens(visual_out.shape[0], sparse_params, visual_out.device)
        visual_out = self.cross_attention(visual_out, text_embed, cross_cu if cross_cu is not None else visual_cu_seqlens, text_cu_seqlens)
        # _nan_check(visual_out, "block.after_cross_attention")
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate, time_embed_idx).type_as(visual_embed)
        # _nan_check(visual_embed, "block.after_cross_attention_residual")

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(self.feed_forward_norm, visual_embed, scale, shift, time_embed_idx).type_as(visual_embed)
        # _nan_check(visual_out, "block.after_ffn_norm")
        visual_out = self.feed_forward(visual_out)
        # _nan_check(visual_out, "block.after_ffn")
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate, time_embed_idx).type_as(visual_embed)
        # _nan_check(visual_embed, "block.after_ffn_residual")
        return visual_embed

    def reset_parameters(self):
        self.visual_modulation.reset_parameters()
        self.self_attention_norm.reset_parameters()
        self.self_attention.reset_parameters()
        self.cross_attention_norm.reset_parameters()
        self.cross_attention.reset_parameters()
        self.feed_forward_norm.reset_parameters()
        self.feed_forward.reset_parameters()
        if hasattr(self, "ucpe_branch"):
            self.ucpe_branch.reset_parameters()
        if hasattr(self, "rayrope_branch"):
            self.rayrope_branch.reset_parameters()
        if hasattr(self, "raymap_branch"):
            self.raymap_branch.reset_parameters()
        if hasattr(self, "crepe_branch"):
            self.crepe_branch.reset_parameters()
        if hasattr(self, "plucker_proj"):
            nn.init.zeros_(self.plucker_proj.weight)


@ModelLogger.log_transformer
class DiffusionTransformer3D(nn.Module):
    def __init__(
        self,
        in_visual_dim=4,
        in_text_dim=3584,
        in_text_dim2=768,
        time_dim=512,
        out_visual_dim=4,
        patch_size=(1, 2, 2),
        model_dim=2048,
        ff_dim=5120,
        num_text_blocks=2,
        num_visual_blocks=32,
        axes_dims=(16, 24, 24),
        visual_cond=False,
        instruct_type=None,
        attention_params=None,
        use_motion_score=False,
        use_ucpe=True,
        use_plucker_mixing=False,
        plucker_raw_frames_per_latent=4,
        use_rayrope=False,
        rayrope_num_rays=1,
        rayrope_freq_base=3.0,
        use_crepe=False,
        crepe_layers=None,
        crepe_num_rays=1,
        crepe_freq_base=3.0,
        crepe_num_breakpoints=8,
        use_scale_token=False,
        use_scale_rope=False,
        use_scale_gate=False,
        use_scale_sca=False,
        use_se_2_3_5x5=False,
        use_se_2_3_6x6=False,
        use_se_2_3_vnorm=False,
        use_se_2_3_wnorm=False,
        use_se_diff=False,
        use_se_2_3=None,   # legacy alias for use_se_2_3_5x5 (old saved configs)
        ucpe_layers=None,
        ucpe_t=None,
        ucpe_attn_mask=None,
        ucpe_chunk_len=3,
        use_raymap=False,
        raymap_layers=_INHERIT_RAYMAP,
        raymap_t=_INHERIT_RAYMAP,
        raymap_attn_mask=_INHERIT_RAYMAP,
        raymap_chunk_len=_INHERIT_RAYMAP,
        i2v_mode="k5",
        navio_use_token_type_embeddings=True,
        navio_num_camera_embeddings=7,
    ):
        super().__init__()
        validate_ucpe_attention_config(ucpe_attn_mask, ucpe_chunk_len)
        if raymap_layers is _INHERIT_RAYMAP:
            raymap_layers = ucpe_layers
        if raymap_t is _INHERIT_RAYMAP:
            raymap_t = ucpe_t
        if raymap_attn_mask is _INHERIT_RAYMAP:
            raymap_attn_mask = ucpe_attn_mask
        if raymap_chunk_len is _INHERIT_RAYMAP:
            raymap_chunk_len = ucpe_chunk_len
        validate_ucpe_attention_config(raymap_attn_mask, raymap_chunk_len)
        head_dim = sum(axes_dims)
        self.in_visual_dim = in_visual_dim
        self.instruct_type = instruct_type
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.visual_cond = visual_cond
        self.i2v_mode = i2v_mode
        if i2v_mode == "tail":
            if patch_size[0] != 1 or not visual_cond:
                raise ValueError("tail requires patch_t=1 and visual_cond=true")
            self.visual_token_type_embeddings = nn.Embedding(2, model_dim)
            self.camera_embeddings = None
        if i2v_mode == "navio":
            if patch_size[0] != 1 or not visual_cond:
                raise ValueError("Navio requires patch_t=1 and visual_cond=true")
            self.visual_token_type_embeddings = (
                nn.Embedding(2, model_dim) if navio_use_token_type_embeddings else None
            )
            if navio_num_camera_embeddings not in (0, 1, 7):
                raise ValueError("Navio camera embedding count must be 0, 1, or 7")
            self.camera_embeddings = (
                nn.Embedding(navio_num_camera_embeddings, model_dim)
                if navio_num_camera_embeddings else None
            )
        self.attention_params = attention_params
        self.use_motion_score = use_motion_score
        self.use_ucpe = use_ucpe
        # UCPE gating ("borders of UCPE"): restrict the UCPE side-branch to a contiguous
        # block span [n,m] inclusive (construction-time, mirrors crepe_layers) and to a
        # noise-level window (runtime, applied in forward). None => no gating on that axis
        # (all blocks / all t), i.e. current behavior.
        if use_ucpe and ucpe_layers is not None:
            self.ucpe_layers = set(range(int(ucpe_layers[0]), int(ucpe_layers[1]) + 1))
        else:
            self.ucpe_layers = set(range(num_visual_blocks)) if use_ucpe else set()
        self.ucpe_t = (float(ucpe_t[0]), float(ucpe_t[1])) if ucpe_t is not None else None
        self.use_raymap = bool(use_raymap)
        if self.use_raymap and raymap_layers is not None:
            self.raymap_layers = set(range(int(raymap_layers[0]), int(raymap_layers[1]) + 1))
        else:
            self.raymap_layers = set(range(num_visual_blocks)) if self.use_raymap else set()
        self.raymap_t = (float(raymap_t[0]), float(raymap_t[1])) if raymap_t is not None else None
        self.use_plucker_mixing = use_plucker_mixing
        self.plucker_raw_frames_per_latent = plucker_raw_frames_per_latent
        self.use_rayrope = use_rayrope
        self.rayrope_num_rays = rayrope_num_rays
        self.rayrope_freq_base = rayrope_freq_base
        self.use_crepe = use_crepe
        # Scale token: feed the per-clip translation magnitude (discarded by
        # normalize_trans) back into the active side-branch (UCPE/RayRoPE/CRePE)
        # so the model can produce absolute motion magnitude. Off => byte-
        # identical scale-blind model (no extra params).
        self.use_scale_token = use_scale_token
        self.use_scale_rope = use_scale_rope
        self.use_scale_gate = use_scale_gate
        self.use_scale_sca = use_scale_sca
        # Renamed use_se_2_3 -> use_se_2_3_5x5; accept the legacy key so old saved configs
        # still build (get_dit does DiffusionTransformer3D(**conf)).
        if use_se_2_3 is not None:
            use_se_2_3_5x5 = bool(use_se_2_3_5x5) or bool(use_se_2_3)
        self.use_se_2_3_5x5 = bool(use_se_2_3_5x5)
        self.use_se_2_3_6x6 = bool(use_se_2_3_6x6)
        self.use_se_2_3_vnorm = bool(use_se_2_3_vnorm)
        self.use_se_2_3_wnorm = bool(use_se_2_3_wnorm)
        self.use_se_diff = bool(use_se_diff)
        # 5x5, 6x6 and se_diff are mutually exclusive UCPE extrinsic modes (each is a different
        # matrix fed to the branch); enable at most one.
        if sum([self.use_se_2_3_5x5, self.use_se_2_3_6x6, self.use_se_diff]) > 1:
            raise ValueError(
                "use_se_2_3_5x5, use_se_2_3_6x6 and use_se_diff are mutually exclusive UCPE "
                "extrinsic modes. Enable at most one."
            )
        # CRePE is applied only to the middle blocks (paper: blocks [10,20) of a
        # 30-block backbone). Default to the centered window of `num_visual_blocks`
        # sized like the paper (≈1/3 of depth, 10/30); override via crepe_layers.
        if use_crepe and crepe_layers is None:
            n_mid = max(1, round(num_visual_blocks * 10 / 30))
            lo = (num_visual_blocks - n_mid) // 2
            crepe_layers = list(range(lo, lo + n_mid))
        self.crepe_layers = set(int(i) for i in crepe_layers) if (use_crepe and crepe_layers is not None) else set()

        visual_embed_dim = 2 * in_visual_dim + 1 if visual_cond or instruct_type == "channel" else in_visual_dim
        self.time_embeddings = TimeEmbeddings(model_dim, time_dim)

        if use_motion_score:
            self.motion_embeddings = TimeEmbeddings(model_dim, time_dim)

        self.text_embeddings = TextEmbeddings(in_text_dim, model_dim)
        self.pooled_text_embeddings = TextEmbeddings(in_text_dim2, time_dim)
        self.visual_embeddings = VisualEmbeddings(visual_embed_dim, model_dim, patch_size)

        if use_plucker_mixing:
            # 6 Plücker channels per raw frame, packed across `R` raw frames
            # within one VAE temporal stride. Same patch_size as visual so the
            # token grid lines up byte-for-byte after fractal_flatten.
            self.plucker_embeddings = PluckerEmbeddings(
                plucker_in_dim=6 * plucker_raw_frames_per_latent,
                model_dim=model_dim,
                patch_size=patch_size,
            )
        if self.use_raymap:
            self.raymap_embeddings = PluckerEmbeddings(6, model_dim, patch_size)

        self.text_rope_embeddings = RoPE1D(head_dim)
        self.text_transformer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(model_dim, time_dim, ff_dim, head_dim) for _ in range(num_text_blocks)]
        )

        self.visual_rope_embeddings = RoPE3D(axes_dims)
        self.visual_transformer_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    model_dim,
                    time_dim,
                    ff_dim,
                    head_dim,
                    # UCPE only on blocks in the [n,m] span (mirror CRePE gating).
                    use_ucpe=use_ucpe and (i in self.ucpe_layers),
                    use_raymap=self.use_raymap and (i in self.raymap_layers),
                    raymap_attn_mask=raymap_attn_mask,
                    raymap_chunk_len=raymap_chunk_len,
                    use_plucker_mixing=use_plucker_mixing,
                    use_rayrope=use_rayrope,
                    rayrope_num_rays=rayrope_num_rays,
                    rayrope_freq_base=rayrope_freq_base,
                    # CRePE only on the selected (middle) blocks.
                    use_crepe=use_crepe and (i in self.crepe_layers),
                    crepe_num_rays=crepe_num_rays,
                    crepe_freq_base=crepe_freq_base,
                    crepe_num_breakpoints=crepe_num_breakpoints,
                    use_scale_token=use_scale_token,
                    use_scale_rope=use_scale_rope,
                    use_scale_gate=use_scale_gate,
                    use_scale_sca=use_scale_sca,
                    use_se_2_3_5x5=self.use_se_2_3_5x5,
                    use_se_2_3_6x6=self.use_se_2_3_6x6,
                    use_se_2_3_vnorm=self.use_se_2_3_vnorm,
                    use_se_2_3_wnorm=self.use_se_2_3_wnorm,
                    use_se_diff=self.use_se_diff,
                    ucpe_attn_mask=ucpe_attn_mask,
                    ucpe_chunk_len=ucpe_chunk_len,
                )
                for i in range(num_visual_blocks)
            ]
        )
        # Stamp every visual block with its index so the activation collector
        # (act_collect) can key captured residuals by depth (self-attn tap fires
        # on all blocks, not just UCPE-bearing ones).
        for i, block in enumerate(self.visual_transformer_blocks):
            block._block_idx = i
        # Annotate each block's UCPE branch with its index so the value-range
        # tracer can filter to specific blocks via UCPE_TRACE_BLOCKS. Guarded
        # so use_ucpe=False configs (plucker-only) don't AttributeError.
        if use_ucpe:
            for i, block in enumerate(self.visual_transformer_blocks):
                if hasattr(block, "ucpe_branch"):
                    block.ucpe_branch._block_idx = i
            self.set_active_ucpe_blocks(
                i for i, block in enumerate(self.visual_transformer_blocks)
                if hasattr(block, "ucpe_branch")
            )
        if use_rayrope:
            for i, block in enumerate(self.visual_transformer_blocks):
                block.rayrope_branch._block_idx = i
        if use_crepe:
            for i, block in enumerate(self.visual_transformer_blocks):
                if hasattr(block, "crepe_branch"):
                    block.crepe_branch._block_idx = i

        self.out_layer = OutLayer(model_dim, time_dim, out_visual_dim, patch_size)

    def get_ucpe_blocks(self):
        return [
            i for i, block in enumerate(self.visual_transformer_blocks)
            if hasattr(block, "ucpe_branch")
        ]

    def get_active_ucpe_blocks(self):
        return [
            i for i, block in enumerate(self.visual_transformer_blocks)
            if hasattr(block, "ucpe_branch")
            and getattr(block, "_ucpe_runtime_enabled", True)
        ]

    def set_active_ucpe_blocks(self, block_ids):
        available = set(self.get_ucpe_blocks())
        requested = {int(block_id) for block_id in block_ids}
        invalid = requested.difference(available)
        if invalid:
            raise ValueError(f"UCPE blocks are not present in this model: {sorted(invalid)}")
        for index, block in enumerate(self.visual_transformer_blocks):
            if hasattr(block, "ucpe_branch"):
                block._ucpe_runtime_enabled = index in requested

    def forward(
        self,
        x,
        text_embed,
        pooled_text_embed,
        time,
        visual_cu_seqlens,
        text_cu_seqlens,
        visual_rope_pos,
        text_rope_pos,
        ucpe_inputs=None,
        scale_factor=(1.0, 1.0, 1.0),
        sparse_params=None,
        motion_score=None,
        plucker_input=None,
        navio_inputs=None,
        raymap_input=None,
    ):
        # _nan_begin_forward()
        # _nan_check(x, "dit.input.x")
        # _nan_check(text_embed, "dit.input.text_embed")
        # _nan_check(pooled_text_embed, "dit.input.pooled_text_embed")
        # _nan_check(time, "dit.input.time")
        text_embed, text_rope_pos, text_cu_seqlens = _expand_rf_text_runs(text_embed, text_rope_pos, text_cu_seqlens, sparse_params)
        text_embed = self.text_embeddings(text_embed)
        # _nan_check(text_embed, "dit.after_text_embeddings")
        time_embed, time_embed_idx = self.time_embeddings(time)
        # _nan_check(time_embed, "dit.after_time_embeddings")
        if motion_score is not None and self.use_motion_score:
            ms_embed, _ = self.motion_embeddings(motion_score)
            time_embed = time_embed + ms_embed

        time_embed = time_embed + self.pooled_text_embeddings(pooled_text_embed)
        # _nan_check(time_embed, "dit.after_pooled_text_addition")
        _orig_visual_cu_seqlens = visual_cu_seqlens
        visual_embed, visual_cu_seqlens = self.visual_embeddings(x, visual_cu_seqlens)
        if self.i2v_mode in {"navio", "tail"}:
            if navio_inputs is None:
                navio_inputs = (ucpe_inputs or {}).get("navio")
            if navio_inputs is None:
                raise ValueError("Navio forward requires token types and physical camera IDs")
            if any(navio_inputs[key].shape != (visual_embed.shape[0],) for key in ("type_ids", "view_ids")):
                raise ValueError("Navio token types and camera IDs must match the visual frame count")
            if ucpe_inputs is not None:
                ucpe_inputs = {**ucpe_inputs, "navio": navio_inputs}
            if self.visual_token_type_embeddings is not None and not (
                self.i2v_mode == "tail" and navio_inputs.get("skip_type_embeddings", False)
            ):
                visual_embed = visual_embed + self.visual_token_type_embeddings(
                    navio_inputs["type_ids"].to(x.device)
                )[:, None, None, :]
            if self.camera_embeddings is not None:
                visual_embed = visual_embed + self.camera_embeddings(
                    navio_inputs["view_ids"].to(x.device)
                )[:, None, None, :]
        # _nan_check(visual_embed, "dit.after_visual_embeddings")

        # Plücker fine-rate camera embeddings — patchified once here, then
        # broadcast to every block. Called with the ORIGINAL pre-patch
        # cu_seqlens (the embedder repeats the same patch_size[0] // step
        # internally). Output shape mirrors visual_embed; both go through
        # the same fractal_flatten below.
        if plucker_input is not None and self.use_plucker_mixing:
            plucker_emb, _ = self.plucker_embeddings(plucker_input, _orig_visual_cu_seqlens)
        else:
            plucker_emb = None

        text_rope = self.text_rope_embeddings(text_rope_pos)
        text_indices = ((sparse_params or {}).get("rf_time_idx") or {}).get("text")
        text_time_embed_idx = (text_indices if text_indices is not None else time_embed_idx).repeat_interleave(torch.diff(text_cu_seqlens), dim=0)
        for text_transformer_block in self.text_transformer_blocks:
            text_embed = text_transformer_block(text_embed, time_embed, text_rope, text_cu_seqlens, text_time_embed_idx)

        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(visual_shape, visual_rope_pos, scale_factor)
        to_fractal = sparse_params.get("to_fractal", False) if sparse_params is not None else False

        # UCPE side-branch assumes (latent_t, latent_h, latent_w) row-major
        # token order to align with PRoPE's cameras*patches_y*patches_x layout.
        # Fractal/Morton reordering would break this; refuse to mix the two.
        assert not to_fractal or ucpe_inputs is None, "UCPE side-branch is incompatible with fractal token order"

        raymap_camera_inputs = ucpe_inputs
        if ucpe_inputs is not None:
            # visual_shape = (sum_T_post, H_post, W_post) — i.e., spatial dims
            # post patch-embedding. Per-sample latent_t_post = cu_seqlens delta.
            # Override whatever pre-patch shape the dataloader stored so PRoPE
            # gets the patches_y/x of the actual token grid.
            _, H_post, W_post = visual_shape
            cu = visual_cu_seqlens
            new_latent_shapes = []
            for i in range(len(ucpe_inputs["viewmats"])):
                lt = int(cu[i + 1].item() - cu[i].item())
                new_latent_shapes.append((lt, int(H_post), int(W_post)))
            ucpe_inputs = dict(ucpe_inputs)
            ucpe_inputs["latent_shapes"] = new_latent_shapes

        visual_embed, visual_rope, visual_cu_seqlens = fractal_flatten(
            visual_embed,
            visual_rope,
            visual_cu_seqlens,
            visual_shape,
            fractal=to_fractal,
        )

        # Match the flatten plucker_emb undergoes to visual_embed's flatten.
        # PRoPE / fractal ordering invariants apply identically. We piggyback
        # on the no-fractal case (the only path the UCPE/Plücker branches
        # support) — fractal+plucker is asserted out below.
        if plucker_emb is not None:
            assert not to_fractal, "Plücker mixing is incompatible with fractal token order"
            plucker_emb = plucker_emb.flatten(0, 2)
            assert plucker_emb.shape[0] == visual_embed.shape[0], (
                f"plucker_emb token count {plucker_emb.shape[0]} != "
                f"visual_embed token count {visual_embed.shape[0]} — check "
                f"that plucker_input shape matches x shape (sum_T, H, W, *)."
            )

        raymap_emb = None
        if self.use_raymap and ucpe_inputs is not None:
            if to_fractal:
                raise ValueError("RayMap is incompatible with fractal token order")
            if raymap_input is None:
                raymap_input = build_raymap_input(
                    raymap_camera_inputs, device=x.device, dtype=x.dtype
                )
                navio = raymap_camera_inputs.get("navio")
                if navio is not None:
                    raymap_input = raymap_input[navio["permutation"].to(raymap_input.device)]
            raymap_emb, _ = self.raymap_embeddings(raymap_input, _orig_visual_cu_seqlens)
            raymap_emb = raymap_emb.flatten(0, 2)
            if raymap_emb.shape != visual_embed.shape:
                raise ValueError("RayMap token grid must match visual token grid")

        frame_indices = ((sparse_params or {}).get("rf_time_idx") or {}).get("video")
        if frame_indices is None:
            visual_time_embed_idx = time_embed_idx.repeat_interleave(torch.diff(visual_cu_seqlens), dim=0)
        else:
            visual_time_embed_idx = frame_indices.repeat_interleave(visual_embed.shape[0] // frame_indices.numel()).to(device=x.device, dtype=torch.int32)

        external_visual_residual_controller = getattr(
            self, "_external_visual_residual_controller", None
        )
        if external_visual_residual_controller is not None:
            external_visual_residual_controller.prepare_forward(
                time=time,
                visual_shape=visual_shape,
                visual_cu_seqlens=visual_cu_seqlens,
                device=visual_embed.device,
                dtype=visual_embed.dtype,
                camera_conditioned=ucpe_inputs is not None,
            )

        # UCPE timestep gate: apply the UCPE residual only in a noise-level window. `time` is
        # per-sample (= 1000*t_code, t_code = noise fraction of x_t; 0 clean .. 1 noise). Gate
        # is active (per token) when (1 - t_code) in [ucpe_t_lo, ucpe_t_hi]. Computed once and
        # threaded via ucpe_inputs so the block just multiplies ucpe_out by it. None => no gate.
        if ucpe_inputs is not None and self.ucpe_t is not None:
            t_code = time.index_select(0, visual_time_embed_idx).to(visual_embed.dtype) / 1000.0
            u = 1.0 - t_code
            t_gate = ((u >= self.ucpe_t[0]) & (u <= self.ucpe_t[1])).to(visual_embed.dtype).unsqueeze(-1)
            ucpe_inputs = {**ucpe_inputs, "t_gate": t_gate}

        raymap_gate = None
        if raymap_emb is not None and self.raymap_t is not None:
            t_code = time.index_select(0, visual_time_embed_idx).to(visual_embed.dtype) / 1000.0
            u = 1.0 - t_code
            raymap_gate = ((u >= self.raymap_t[0]) & (u <= self.raymap_t[1])).to(visual_embed.dtype).unsqueeze(-1)

        if act_collect.enabled():
            act_collect.set_seqlens(visual_cu_seqlens)
        for block_idx, visual_transformer_block in enumerate(self.visual_transformer_blocks):
            visual_embed = visual_transformer_block(
                visual_embed,
                text_embed,
                time_embed,
                visual_rope,
                visual_cu_seqlens,
                text_cu_seqlens,
                visual_time_embed_idx,
                sparse_params,
                ucpe_inputs,
                plucker_emb,
                external_visual_residual_controller,
                raymap_emb,
                raymap_gate,
                ucpe_inputs["latent_shapes"] if raymap_emb is not None else None,
            )
            # _nan_check(visual_embed, f"dit.after_visual_block[{block_idx}]")
        visual_embed, visual_cu_seqlens = fractal_unflatten(visual_embed, visual_cu_seqlens, visual_shape, fractal=to_fractal)
        # _nan_check(visual_embed, "dit.after_fractal_unflatten")

        frame_indices = ((sparse_params or {}).get("rf_time_idx") or {}).get("video")
        if frame_indices is None:
            visual_time_embed_idx = time_embed_idx.repeat_interleave(torch.diff(visual_cu_seqlens), dim=0)
        else:
            visual_time_embed_idx = frame_indices.repeat_interleave(visual_embed.shape[0] // frame_indices.numel()).to(device=x.device, dtype=torch.int32)
        x = self.out_layer(visual_embed, text_embed, time_embed, visual_cu_seqlens, visual_time_embed_idx)
        # _nan_check(x, "dit.after_out_layer")
        return x

    def enable_explicit_forward_prefetch(self, factor=1):
        if factor <= 0 or getattr(self, "forward_prefetch", False):
            return
        for i, layer in enumerate(self.visual_transformer_blocks):
            if not hasattr(layer, "set_modules_to_forward_prefetch"):
                raise RuntimeError("wrap DiT with FSDP first to use forward_prefetch")
            start, end = i, i + factor
            prefetch_modules = self.visual_transformer_blocks[start:end]
            layer.set_modules_to_forward_prefetch(prefetch_modules)
        self.forward_prefetch = True

    def reset_parameters(self):
        self.time_embeddings.reset_parameters()
        if self.i2v_mode in {"navio", "tail"}:
            if self.visual_token_type_embeddings is not None:
                nn.init.zeros_(self.visual_token_type_embeddings.weight)
            if self.camera_embeddings is not None:
                nn.init.zeros_(self.camera_embeddings.weight)
        self.text_embeddings.reset_parameters()
        self.pooled_text_embeddings.reset_parameters()
        self.visual_embeddings.reset_parameters()
        if hasattr(self, "plucker_embeddings"):
            self.plucker_embeddings.reset_parameters()
        if hasattr(self, "raymap_embeddings"):
            self.raymap_embeddings.reset_parameters()

        self.text_rope_embeddings.reset_parameters()
        for module in self.text_transformer_blocks:
            module.reset_parameters()

        self.visual_rope_embeddings.reset_parameters()
        for module in self.visual_transformer_blocks:
            module.reset_parameters()

        self.out_layer.reset_parameters()


def get_dit(conf):
    return DiffusionTransformer3D(**conf)
