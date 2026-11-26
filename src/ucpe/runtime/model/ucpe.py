"""Per-block UCPE side-branch attention for Kandinsky-5.

Geometric side branch where Q, K, V are rotated by per-token SE(3) matrices
anchored at each token's camera ray before the softmax dot product (and rotated
back at the output). Adds to the residual stream with a zero-initialized
output projection, so step-0 behavior matches the pretrained baseline and
training learns how much geometric content to mix in.

References:
- UCPE (Zhang et al., arXiv 2512.07237)
- PRoPE (arXiv 2507.10496) — the underlying ray-local Q/K transform engine
  vendored at ``kandinsky/model/prope.py`` (MIT).
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..camera_control import _se6_angular_velocity_from_rotations

from .nan_debug import check as _nan_check, trace_stats as _nan_trace
from .prope import _prepare_apply_fns
from .scale_gate import ScaleGate, apply_scale_gate
from .scale_sca import ScaleSCA, apply_scale_sca
from .scale_token import ScaleEmbedding, apply_scale_token


def validate_ucpe_attention_config(ucpe_attn_mask, ucpe_chunk_len):
    if ucpe_attn_mask not in (None, "A", "B", "C"):
        raise NotImplementedError(
            f"Unsupported ucpe_attn_mask={ucpe_attn_mask!r}; expected one of None, A, B, C"
        )
    if not isinstance(ucpe_chunk_len, int) or isinstance(ucpe_chunk_len, bool) or ucpe_chunk_len <= 0:
        raise ValueError(
            f"ucpe_chunk_len must be a positive integer, got {ucpe_chunk_len!r}"
        )


def normalize_se_pose(
    viewmats_list: List[torch.Tensor],
    latent_shapes: List[Tuple[int, int, int]],
    *,
    device,
    dtype,
    norm_v: bool = False,
    norm_w: bool = False,
    eps: float = 1e-6,
):
    """Unit-normalize the velocity (col 3) column per frame and collect per-frame ||v||, ||w||.

    Backs ``use_se_2_3_vnorm`` (``norm_v``) and ``use_se_2_3_wnorm`` (``norm_w``): the UCPE branch
    then sees the DIRECTION-only velocity column, and the caller re-injects magnitude by weighting
    the two per-token norms with trainable coefficients (``v_coef*||v|| + w_coef*||w||``).

    ``norm_w`` works for both 6x6 and 5x5:
      - 6x6: unit-normalize the angular-velocity column (col 4); ``||w||`` = its per-frame norm.
      - 5x5: there is NO w column, so the matrix is left unchanged and ``||w||`` is computed from the
        rotation block (per-frame ``Log_SO3(R_f^T R_{f+1})``, i.e. the per-latent-step rotation angle).
        This equals the 6x6 column's ||w|| up to the per-clip dt=1/latent_fps factor, which the
        trainable ``w_coef`` absorbs.

    Returns ``(normalized_viewmats_list, v_scale, w_scale)`` where each of ``v_scale``/``w_scale``
    has shape ``(sum_T, 1)`` in frame-major token order (t = frame*H*W + y*W + x), matching the
    packed ``(sum_T, D)`` layout of ``UcpeSideBranch.forward``, or is ``None`` when that component was
    not requested. Zero columns (clamp_min avoids NaN) map to a zero norm for that component.
    """
    norm_list = []
    v_chunks = []
    w_chunks = []
    for viewmats, (lt, h, w) in zip(viewmats_list, latent_shapes):
        pose_dim = int(viewmats.shape[-1])
        if norm_w and pose_dim not in (5, 6):
            raise ValueError(
                f"normalize_se_pose(norm_w=True) requires 5x5 or 6x6 matrices, got "
                f"{tuple(viewmats.shape[-2:])}"
            )
        vm = viewmats.clone()                                        # don't mutate shared ucpe_inputs
        if norm_v:
            v = vm[..., :3, 3]                                       # velocity column, (latent_t, 3)
            vnorm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
            vm[..., :3, 3] = v / vnorm.clamp_min(eps)
            v_chunks.append(vnorm.squeeze(-1).to(device=device, dtype=dtype).repeat_interleave(h * w))
        if norm_w:
            if pose_dim == 6:
                wv = vm[..., :3, 4]                                  # angular-velocity column
                wnorm = torch.linalg.vector_norm(wv, dim=-1, keepdim=True)
                vm[..., :3, 4] = wv / wnorm.clamp_min(eps)
                wnorm = wnorm.squeeze(-1)                            # (latent_t,)
            else:  # pose_dim == 5: no w column -> derive ||w|| from the rotation block (matrix unchanged)
                w_vec = _se6_angular_velocity_from_rotations(vm[..., :3, :3], dt=1.0)  # (latent_t, 3)
                wnorm = torch.linalg.vector_norm(w_vec, dim=-1)      # (latent_t,)
            w_chunks.append(wnorm.to(device=device, dtype=dtype).repeat_interleave(h * w))
        norm_list.append(vm)
    v_scale = torch.cat(v_chunks).unsqueeze(-1) if norm_v else None   # (sum_T, 1) or None
    w_scale = torch.cat(w_chunks).unsqueeze(-1) if norm_w else None   # (sum_T, 1) or None
    return norm_list, v_scale, w_scale


def expand_per_frame(per_sample, latent_shapes, *, device, dtype):
    """Expand a per-sample list of per-frame ``(latent_t,)`` tensors into a packed per-token
    ``(sum_T, 1)`` scale in frame-major order (each frame value repeated over its ``h*w`` tokens),
    matching the packed ``(sum_T, D)`` layout of ``UcpeSideBranch.forward``. Used by the use_se_diff
    velocity method to re-inject the data-computed per-frame ||p'||/||R'||_F magnitudes. Returns
    ``None`` if ``per_sample`` is ``None`` or any sample's entry is ``None`` (norm off)."""
    if per_sample is None:
        return None
    chunks = []
    for val, (lt, h, w) in zip(per_sample, latent_shapes):
        if val is None:
            return None
        chunks.append(val.to(device=device, dtype=dtype).reshape(-1).repeat_interleave(int(h) * int(w)))
    if not chunks:
        return None
    return torch.cat(chunks).unsqueeze(-1)


class UcpeSideBranch(nn.Module):
    """Per-block softmax side-branch attention with PRoPE Q/K/V transforms.

    Per-sample dispatch: visual tokens arrive packed as ``(sum_T, D)`` with
    ``visual_cu_seqlens`` describing batch boundaries. We slice per sample,
    run a (1, num_heads, T_i, head_dim) PRoPE attention, and write back into
    the same packed layout. Acceptable for our 8 GPU / micro_batch=1 setting.
    """

    def __init__(self, model_dim: int, head_dim: int, attn_compress: int = 8,
                 use_scale_token: bool = False, use_scale_rope: bool = False,
                 use_scale_gate: bool = False, use_scale_sca: bool = False,
                 use_se_2_3_5x5: bool = False, use_se_2_3_6x6: bool = False,
                 use_se_2_3_vnorm: bool = False, use_se_2_3_wnorm: bool = False,
                 ucpe_attn_mask: str = None, ucpe_chunk_len: int = 3):
        super().__init__()
        validate_ucpe_attention_config(ucpe_attn_mask, ucpe_chunk_len)
        # Reference UCPE compresses the side-branch attention dim by 8×
        # (camera_control.py line 65: attn_dim = dim // attn_compress, default
        # attn_compress=8). With model_dim=1792 this gives attn_dim=224 and
        # ~64× fewer side-branch parameters per block. Crucial for matching
        # the paper's <1% trainable-params claim AND for stability: the
        # smaller param surface means smaller gradients per step, less
        # opportunity for the random-direction adapter problem to run away.
        attn_dim = model_dim // attn_compress
        # Pick num_heads so that head_dim_attn % 4 == 0 (PRoPE requirement)
        # and head_dim_attn is as large as possible for stable attention.
        # Default sequence: try head_dim, then halve / scan candidates.
        head_dim_attn = None
        for candidate in (head_dim, head_dim // 2, 56, 32, 16, 8):
            if candidate > 0 and candidate % 4 == 0 and attn_dim % candidate == 0:
                head_dim_attn = candidate
                break
        if head_dim_attn is None:
            raise ValueError(
                f"Could not find a valid head_dim_attn for attn_dim={attn_dim}"
            )
        num_heads_attn = attn_dim // head_dim_attn
        self.model_dim = model_dim
        self.attn_dim = attn_dim
        self.head_dim = head_dim_attn
        self.num_heads = num_heads_attn
        self.attn_compress = attn_compress
        self.use_scale_rope = use_scale_rope
        self.use_se_2_3_5x5 = use_se_2_3_5x5
        self.use_se_2_3_6x6 = use_se_2_3_6x6
        self.ucpe_attn_mask = ucpe_attn_mask
        self.ucpe_chunk_len = ucpe_chunk_len

        # Pre-attn LayerNorm: matches base TransformerDecoderBlock, which
        # always feeds self_attention_norm(visual_embed) into self_attention.
        # Without this, UCPE sees the post-residual stream raw — empirically
        # block-15 input hits abs_max≈3.5e4 in our traces, which then drives
        # q_proj output to O(1e4) and SDPA logits past softmax saturation.
        self.input_norm = nn.LayerNorm(model_dim, elementwise_affine=False)

        self.q_proj = nn.Linear(model_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, attn_dim, bias=False)
        # QK RMSNorm: matches base MultiheadSelfAttention.query_norm/key_norm
        # (RMSNorm over head_dim). Also matches SANA-WM's q_norm_cam/k_norm_cam.
        # Required before PRoPE because the per-camera SE3 matrix has spectral
        # norm > 1 on arbitrary 4-D feature groups, so Q/K can drift unbounded
        # across training without it.
        self.q_norm = nn.RMSNorm(head_dim_attn)
        self.k_norm = nn.RMSNorm(head_dim_attn)
        self.out_proj = nn.Linear(attn_dim, model_dim, bias=True)
        self._zero_init_out_proj()

        # Scale token (optional; dit.params.use_scale_token): FiLM-modulate the
        # zero-init geometric residual by the per-clip translation magnitude that
        # normalize_trans discards, so UCPE can learn ABSOLUTE motion magnitude.
        # Zero-init out_layer => step-0 output unchanged. None when off (no extra
        # params, state_dict identical to the scale-blind branch).
        self.scale_embed = ScaleEmbedding(model_dim) if use_scale_token else None
        self.scale_gate = ScaleGate(model_dim) if use_scale_gate else None
        self.scale_sca = ScaleSCA(model_dim) if use_scale_sca else None

        # Trainable per-block multipliers for the vnorm/wnorm magnitude re-injection:
        # UCPE-out scale = v_coef*||v|| (+ w_coef*||w|| for 6x6+wnorm). Init 1.0 => step-0
        # identical to the previous fixed (1.0) scale. Shape (1,) not 0-dim for FSDP2.
        # v is normalized under vnorm OR wnorm; w only under wnorm.
        self.v_coef = nn.Parameter(torch.ones(1)) if (use_se_2_3_vnorm or use_se_2_3_wnorm) else None
        self.w_coef = nn.Parameter(torch.ones(1)) if use_se_2_3_wnorm else None

    def _zero_init_out_proj(self):
        """Zero-init out_proj so step-0 residual is identically zero; the
        pretrained backbone passes through unchanged until UCPE learns a
        non-zero output."""
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def reset_parameters(self):
        self.input_norm.reset_parameters()
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        self.q_norm.reset_parameters()
        self.k_norm.reset_parameters()
        self.out_proj.reset_parameters()
        self._zero_init_out_proj()
        if self.scale_embed is not None:
            self.scale_embed.reset_parameters()
        if self.scale_gate is not None:
            self.scale_gate.reset_parameters()
        if self.scale_sca is not None:
            self.scale_sca.reset_parameters()
        # vnorm/wnorm magnitude multipliers: init 1.0. MUST live here (not just __init__)
        # because meta-init + to_empty() leaves them uninitialized until reset_parameters().
        if self.v_coef is not None:
            nn.init.ones_(self.v_coef)
        if self.w_coef is not None:
            nn.init.ones_(self.w_coef)

    @torch.no_grad()
    def init_from_base_attn(self, base_self_attn, num_heads_base: int, head_dim_base: int):
        """Copy a per-head-truncated slice of the pretrained block's
        ``to_query / to_key / to_value`` weights into the UCPE branch's
        ``q_proj / k_proj / v_proj``. Out_proj stays zero-init (preserves
        the step-0 ``UCPE residual == 0`` invariant).

        NOT part of the reference UCPE recipe (the paper uses default
        kaiming for Q/K/V). This is a custom experiment to test whether
        a base-aligned init helps an adapter-only setup converge on a
        frozen Kandinsky-5 backbone.

        Mapping: base weight (num_heads_base, head_dim_base, in_dim) →
        ucpe weight (num_heads_ucpe, head_dim_ucpe, in_dim) takes the
        first ``num_heads_ucpe`` base heads, first ``head_dim_ucpe`` dims
        of each head.

        FSDP-aware: gathers ``full_tensor()`` on base, slices, then
        re-shards via ``distribute_tensor`` to match the UCPE weight's
        device_mesh + placements.
        """
        # Inputs are torch.distributed DTensors under FSDP.
        try:
            from torch.distributed.tensor import distribute_tensor
        except ImportError:  # older torch
            from torch.distributed._tensor import distribute_tensor

        for ucpe_lin, base_lin in (
            (self.q_proj, base_self_attn.to_query),
            (self.k_proj, base_self_attn.to_key),
            (self.v_proj, base_self_attn.to_value),
        ):
            ucpe_w = ucpe_lin.weight
            base_w = base_lin.weight
            if hasattr(base_w, "full_tensor"):
                base_full = base_w.full_tensor()
            else:
                base_full = base_w.detach()
            in_dim = base_full.shape[1]
            base_4d = base_full.view(num_heads_base, head_dim_base, in_dim)
            sliced = base_4d[: self.num_heads, : self.head_dim, :].reshape(
                self.attn_dim, in_dim
            ).to(ucpe_w.dtype).contiguous()
            if hasattr(ucpe_w, "device_mesh") and hasattr(ucpe_w, "placements"):
                new_dt = distribute_tensor(
                    sliced.to(ucpe_w.device),
                    device_mesh=ucpe_w.device_mesh,
                    placements=ucpe_w.placements,
                )
                ucpe_w.copy_(new_dt)
            else:
                ucpe_w.copy_(sliced.to(ucpe_w.device))

    @staticmethod
    def _downscale_to_reference_rms(ref, transformed, eps: float = 1e-6):
        """Stabilizer borrowed from SANA-WM. PRoPE's SE(3) projmat is only
        norm-preserving for true 3-D homogeneous coords; on arbitrary 4-D
        learned-feature groups its spectral norm can exceed 1 for certain
        camera configurations, amplifying attention-space magnitudes
        unboundedly over training. Clamp per-(head, token) RMS so the
        transformed tensor can never exceed the pre-transform reference.

        Tensor layout: (B, num_heads, T, head_dim) — RMS reduces over the
        last axis (head_dim), shape SANA uses `(B, H, D, N)` and reduces
        over dim=2 (their D = head_dim).
        """
        ref_rms = ref.float().square().mean(dim=-1, keepdim=True).add(eps).sqrt()
        tr_rms = transformed.float().square().mean(dim=-1, keepdim=True).add(eps).sqrt()
        scale = (ref_rms / tr_rms.clamp_min(eps)).clamp(max=1.0).to(transformed.dtype)
        return transformed * scale

    def _apply_attention(self, q, k, v, tokens_per_frame: int):
        """Apply the configured UCPE-only block mask without materializing an NxN mask."""
        attention = getattr(self, "_speedup_attention", F.scaled_dot_product_attention)
        if self.ucpe_attn_mask is None:
            return attention(q, k, v)

        sequence_len = q.shape[-2]
        if k.shape[-2] != sequence_len or v.shape[-2] != sequence_len:
            raise ValueError("UCPE self-attention requires equal query, key, and value lengths")
        if tokens_per_frame <= 0 or sequence_len % tokens_per_frame != 0:
            raise ValueError(
                f"UCPE sequence length {sequence_len} is not divisible by "
                f"tokens_per_frame={tokens_per_frame}"
            )

        chunk_tokens = self.ucpe_chunk_len * tokens_per_frame
        outputs = []
        for query_start in range(0, sequence_len, chunk_tokens):
            query_end = min(query_start + chunk_tokens, sequence_len)
            if self.ucpe_attn_mask == "A":
                key_start = query_start
            elif self.ucpe_attn_mask == "B":
                key_start = max(0, query_start - chunk_tokens)
            else:  # C: all completed chunks plus the full current chunk.
                key_start = 0
            outputs.append(
                attention(
                    q[..., query_start:query_end, :],
                    k[..., key_start:query_end, :],
                    v[..., key_start:query_end, :],
                )
            )
        return torch.cat(outputs, dim=-2)

    def forward(
        self,
        x_packed: torch.Tensor,
        visual_cu_seqlens: torch.Tensor,
        viewmats_per_sample: List[torch.Tensor],
        Ks_per_sample: List[torch.Tensor],
        latent_shapes: List[Tuple[int, int, int]],
        frame_wh: List[Tuple[int, int]],
        trans_scale=None,
    ) -> torch.Tensor:
        """
        Args:
            x_packed: (sum_T, D) packed visual tokens across the batch.
            visual_cu_seqlens: (B+1,) cumulative seqlen boundaries.
            viewmats_per_sample: list of B tensors, each (latent_t, 4, 4),
                world-to-camera (`camera<-world`).
            Ks_per_sample: list of B tensors, each (latent_t, 3, 3).
            latent_shapes: list of B (latent_t, latent_h, latent_w) tuples.
            frame_wh: list of B (frame_w, frame_h) tuples — original pixel
                size used to normalize intrinsics inside PRoPE.

        Returns:
            (sum_T, D) zero-init residual contribution; add it to the
            block's running ``visual_embed``.
        """
        out = torch.zeros_like(x_packed)
        cu = visual_cu_seqlens
        batch = len(viewmats_per_sample)
        assert cu.numel() == batch + 1, f"cu_seqlens has {cu.numel()-1} samples, expected {batch}"

        b_idx = getattr(self, "_block_idx", None)
        # _nan_check(x_packed, f"ucpe[{b_idx}].input.x_packed")
        # _nan_trace(x_packed, f"ucpe[{b_idx}].input.x_packed", block_idx=b_idx)
        # Param magnitudes (independent of forward but useful to watch drift).
        # _nan_trace(self.q_proj.weight, f"ucpe[{b_idx}].q_proj.weight", block_idx=b_idx)
        # _nan_trace(self.k_proj.weight, f"ucpe[{b_idx}].k_proj.weight", block_idx=b_idx)
        # _nan_trace(self.v_proj.weight, f"ucpe[{b_idx}].v_proj.weight", block_idx=b_idx)
        # _nan_trace(self.out_proj.weight, f"ucpe[{b_idx}].out_proj.weight", block_idx=b_idx)
        # _nan_trace(self.out_proj.bias, f"ucpe[{b_idx}].out_proj.bias", block_idx=b_idx)
        # _nan_check(self.q_proj.weight, f"ucpe[{b_idx}].q_proj.weight")
        # _nan_check(self.k_proj.weight, f"ucpe[{b_idx}].k_proj.weight")
        # _nan_check(self.v_proj.weight, f"ucpe[{b_idx}].v_proj.weight")
        # _nan_check(self.out_proj.weight, f"ucpe[{b_idx}].out_proj.weight")
        # _nan_check(self.out_proj.bias, f"ucpe[{b_idx}].out_proj.bias")

        # Pre-attn LayerNorm so q/k/v see RMS≈1 instead of the raw post-residual
        # stream (which by mid-network is O(1e4) in Kandinsky-5). nn.LayerNorm
        # is in autocast's fp32 op list — it auto-promotes the input internally,
        # so an explicit .float() before the call is redundant and only adds
        # a (sum_T, model_dim) fp32 allocation. .type_as casts back to bf16.
        x_normed = self.input_norm(x_packed).type_as(x_packed)
        # _nan_check(x_normed, f"ucpe[{b_idx}].input_normed")
        # _nan_trace(x_normed, f"ucpe[{b_idx}].input_normed", block_idx=b_idx)

        # Project q/k/v on the full packed tensor once (avoids B linear calls).
        q_packed = self.q_proj(x_normed).view(-1, self.num_heads, self.head_dim)
        k_packed = self.k_proj(x_normed).view(-1, self.num_heads, self.head_dim)
        v_packed = self.v_proj(x_normed).view(-1, self.num_heads, self.head_dim)
        # _nan_check(q_packed, f"ucpe[{b_idx}].q_proj_output")
        # _nan_check(k_packed, f"ucpe[{b_idx}].k_proj_output")
        # _nan_check(v_packed, f"ucpe[{b_idx}].v_proj_output")
        # _nan_trace(q_packed, f"ucpe[{b_idx}].q_proj_output", block_idx=b_idx)
        # _nan_trace(k_packed, f"ucpe[{b_idx}].k_proj_output", block_idx=b_idx)
        # _nan_trace(v_packed, f"ucpe[{b_idx}].v_proj_output", block_idx=b_idx)

        # QK RMSNorm before PRoPE — bounds Q/K so the SE3/intrinsics matrix
        # in PRoPE has a known-scale input. Like nn.LayerNorm above,
        # nn.RMSNorm auto-promotes its input to fp32 under autocast
        # (verified on torch 2.6.0). type_as casts the fp32 output back to bf16
        # so SDPA stays on the flash kernel.
        q_packed = self.q_norm(q_packed).type_as(q_packed)
        k_packed = self.k_norm(k_packed).type_as(k_packed)
        # _nan_check(q_packed, f"ucpe[{b_idx}].q_after_qk_norm")
        # _nan_check(k_packed, f"ucpe[{b_idx}].k_after_qk_norm")
        # _nan_trace(q_packed, f"ucpe[{b_idx}].q_after_qk_norm", block_idx=b_idx)
        # _nan_trace(k_packed, f"ucpe[{b_idx}].k_after_qk_norm", block_idx=b_idx)

        # Note: out_packed has attn_dim (not model_dim) per row — view dims
        # match the q/k/v output before reshape.
        out_packed = torch.zeros(
            x_packed.shape[0], self.num_heads, self.head_dim,
            device=x_packed.device, dtype=x_packed.dtype,
        )

        for i in range(batch):
            start = int(cu[i].item())
            end = int(cu[i + 1].item())
            T_i = end - start
            latent_t, latent_h, latent_w = latent_shapes[i]
            assert T_i == latent_t * latent_h * latent_w, (
                f"sample {i}: packed T={T_i} != latent_t*h*w={latent_t}*{latent_h}*{latent_w}="
                f"{latent_t*latent_h*latent_w}"
            )
            frame_w, frame_h = frame_wh[i]

            # Keep q/k/v in their native autocast dtype (bf16) so SDPA can
            # dispatch to a flash backend. The fp32 cost is paid only on the
            # small SE3/intrinsics matrices and on the matrix-times-feats
            # einsum inside _apply_to_q/kv/o (which promotes to fp32). We then
            # immediately downcast the transformed q/k/v back to bf16 before
            # the actual attention — that's where the heavy compute lives.
            attn_dtype = q_packed.dtype  # bf16 under autocast
            q_i = q_packed[start:end].permute(1, 0, 2).unsqueeze(0).contiguous()
            k_i = k_packed[start:end].permute(1, 0, 2).unsqueeze(0).contiguous()
            v_i = v_packed[start:end].permute(1, 0, 2).unsqueeze(0).contiguous()

            tag = f"ucpe[{b_idx}].sample[{i}]"
            # _nan_check(q_i, f"{tag}.q_i_bf16")
            # _nan_check(k_i, f"{tag}.k_i_bf16")
            # _nan_check(v_i, f"{tag}.v_i_bf16")

            # SE3 setup in fp32: viewmats inverse, intrinsics inverse, lift_K,
            # P/P_T/P_inv.
            viewmats_i = viewmats_per_sample[i].to(q_i.device, torch.float32).unsqueeze(0)
            Ks_i = Ks_per_sample[i].to(q_i.device, torch.float32).unsqueeze(0)
            expected_pose_dim = 6 if self.use_se_2_3_6x6 else (5 if self.use_se_2_3_5x5 else 4)
            if viewmats_i.shape[-2:] != (expected_pose_dim, expected_pose_dim):
                raise ValueError(
                    f"UCPE expected {expected_pose_dim}x{expected_pose_dim} viewmats for "
                    f"use_se_2_3_5x5={self.use_se_2_3_5x5}/use_se_2_3_6x6={self.use_se_2_3_6x6}, "
                    f"got {tuple(viewmats_i.shape[-2:])}"
                )
            rope_scale_i = None
            if self.use_scale_rope and trans_scale is not None:
                rope_scale_i = torch.as_tensor(
                    [trans_scale[i]], device=q_i.device, dtype=torch.float32
                )
            # _nan_check(viewmats_i, f"{tag}.viewmats_fp32")
            # _nan_check(Ks_i, f"{tag}.Ks_fp32")
            # _nan_trace(viewmats_i, f"{tag}.viewmats_fp32", block_idx=b_idx)
            # _nan_trace(Ks_i, f"{tag}.Ks_fp32", block_idx=b_idx)
            apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns(
                head_dim=self.head_dim,
                viewmats=viewmats_i, Ks=Ks_i,
                patches_x=latent_w, patches_y=latent_h,
                image_width=frame_w, image_height=frame_h,
                rope_scale=rope_scale_i,
            )

            # Transform q/k/v.
            q_t_fp32 = apply_fn_q(q_i)
            k_t_fp32 = apply_fn_kv(k_i)
            v_t_fp32 = apply_fn_kv(v_i)
            # _nan_check(q_t_fp32, f"{tag}.q_after_apply_fn_q")
            # _nan_check(k_t_fp32, f"{tag}.k_after_apply_fn_kv")
            # _nan_check(v_t_fp32, f"{tag}.v_after_apply_fn_kv")
            # _nan_trace(q_t_fp32, f"{tag}.q_after_apply_fn_q", block_idx=b_idx)
            # _nan_trace(k_t_fp32, f"{tag}.k_after_apply_fn_kv", block_idx=b_idx)
            # _nan_trace(v_t_fp32, f"{tag}.v_after_apply_fn_kv", block_idx=b_idx)

            q_t = q_t_fp32.to(attn_dtype)
            k_t = k_t_fp32.to(attn_dtype)
            v_t = v_t_fp32.to(attn_dtype)
            # RMS clamp so PRoPE's projmat can never amplify Q/K/V vs the
            # pre-transform reference. Without this the side branch grows
            # unboundedly over ~300+ steps (see SANA-WM Branch-1 fix).
            q_t = self._downscale_to_reference_rms(q_i, q_t)
            k_t = self._downscale_to_reference_rms(k_i, k_t)
            v_t = self._downscale_to_reference_rms(v_i, v_t)
            # _nan_check(q_t, f"{tag}.q_downcast_bf16")
            # _nan_check(k_t, f"{tag}.k_downcast_bf16")
            # _nan_check(v_t, f"{tag}.v_downcast_bf16")
            # _nan_trace(q_t, f"{tag}.q_post_rms_clamp", block_idx=b_idx)
            # _nan_trace(k_t, f"{tag}.k_post_rms_clamp", block_idx=b_idx)

            attn_out = self._apply_attention(
                q_t,
                k_t,
                v_t,
                tokens_per_frame=latent_h * latent_w,
            )
            # _nan_check(attn_out, f"{tag}.sdpa_output")
            # _nan_trace(attn_out, f"{tag}.sdpa_output", block_idx=b_idx)
            sdpa_ref = attn_out

            attn_out_fp32 = apply_fn_o(attn_out)
            # _nan_check(attn_out_fp32, f"{tag}.after_apply_fn_o")
            attn_out = attn_out_fp32.to(attn_dtype)
            # Same RMS clamp on the inverse rotation — apply_fn_o was the
            # second amplification site we saw in traces (3.5 → 19.75 etc).
            attn_out = self._downscale_to_reference_rms(sdpa_ref, attn_out)
            # _nan_check(attn_out, f"{tag}.attn_out_downcast_bf16")
            # _nan_trace(attn_out, f"{tag}.attn_out_post_rms_clamp", block_idx=b_idx)

            # NaN guard.
            attn_out = torch.where(torch.isfinite(attn_out), attn_out, torch.zeros_like(attn_out))
            # _nan_check(attn_out, f"{tag}.after_nan_guard")

            out_packed[start:end] = attn_out.squeeze(0).permute(1, 0, 2)

        # _nan_check(out_packed, f"ucpe[{b_idx}].out_packed_all_samples")
        # _nan_trace(out_packed, f"ucpe[{b_idx}].out_packed_all_samples", block_idx=b_idx)
        out = self.out_proj(out_packed.reshape(-1, self.attn_dim))
        # _nan_check(out, f"ucpe[{b_idx}].out_proj_output")
        # _nan_trace(out, f"ucpe[{b_idx}].out_proj_output", block_idx=b_idx)
        # Scale token: FiLM-modulate the residual by per-clip translation
        # magnitude (no-op when scale_embed is None / trans_scale is None;
        # zero at init).
        out = apply_scale_token(out, self.scale_embed, trans_scale, cu, batch)
        out = apply_scale_gate(out, self.scale_gate, cu, batch)
        out = apply_scale_sca(out, self.scale_sca, trans_scale, cu, batch)
        return out
