"""Per-block RayRoPE side-branch attention for Kandinsky-5.

A geometric side branch that conditions the visual stream on the camera
*trajectory* (per-frame extrinsics + intrinsics), implemented as the
**depth-free, global single-pass** variant of RayRoPE: each token's query/key
are rotated by a rotary positional encoding built from its ray geometry —
the camera center ``p0`` and the per-patch ray direction at infinity
``pinf`` — expressed in a common (frame-0-anchored) world frame. The relative
camera geometry between any two tokens then emerges in the q·k dot product,
exactly as ordinary RoPE makes absolute positions act relatively.

This is the direct analog of ``UcpeSideBranch`` (kandinsky/model/ucpe.py):
the branch adds to the residual stream through a **zero-initialized output
projection**, so step-0 behaviour matches the pretrained backbone and training
learns how much geometric content to mix in. The only structural difference is
the geometric engine — RayRoPE's ray-rotary (norm-preserving) instead of
PRoPE's ray-local SE(3) projmats.

The branch consumes the *same* per-sample inputs the UCPE branch already
receives (``viewmats`` = camera<-world per latent frame, ``Ks`` = pinhole
intrinsics, latent shapes, raw frame size), so no dataloader changes are
needed: ``use_rayrope: true`` simply routes ``ucpe_inputs`` here instead.

References:
- RayRoPE: Projective Ray Positional Encoding for Multi-view Attention
  (Wu et al., arXiv 2601.15275; code https://github.com/Lucas-707/RayRoPE).
  The depth-free global-ray formulation here is adapted from that repo's
  ``pos_enc/rope_global_ray.py`` (``pos_enc_type='global-0+inf'``).
- PRoPE / "Cameras as Relative Positional Encoding" (arXiv 2507.10496, MIT) —
  the projection-matrix conventions (``normalize_K``/``P_inv``) are shared.

Depth: the paper's headline ``d_pj+0_3d`` config feeds *predicted scene depth*
into the encoding. That is an NVS feature, not part of a camera trajectory, and
would require a learned depth head; we deliberately drop it (use_pd=False) and
encode only the trajectory (p0 + pinf).
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .scale_token import ScaleEmbedding, apply_scale_token


# ---------------------------------------------------------------------------
# Geometry helpers (shared conventions with kandinsky/model/prope.py).
# ---------------------------------------------------------------------------


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a (..., 4, 4) SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Lift (..., 3, 3) intrinsics to homogeneous (..., 4, 4)."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device, dtype=Ks.dtype)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert (..., 3, 3) intrinsics. Assumes no skew."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out


def normalize_K(Ks: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
    """Normalize pinhole intrinsics to the [-0.5, 0.5] image box (principal
    point at 0). Matches PRoPE's ``Ks_norm``."""
    Ks_norm = torch.zeros_like(Ks)
    Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_width
    Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_height
    Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_width - 0.5
    Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_height - 0.5
    Ks_norm[..., 2, 2] = 1.0
    return Ks_norm


def _ray_offsets(num_rays_per_patch: int) -> List[List[float]]:
    if num_rays_per_patch == 3:
        return [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    if num_rays_per_patch == 2:
        return [[0.0, 0.0], [1.0, 1.0]]
    return [[0.5, 0.5]]  # single center ray


def _pinf_local_coords(
    patches_x: int,
    patches_y: int,
    offsets: List[List[float]],
    device,
    dtype,
) -> torch.Tensor:
    """Per-patch point-at-infinity homogeneous coords ``[u, v, 1, 0]`` in the
    normalized-pixel frame (principal point at 0), shape
    ``(num_patches, num_rays, 4)``.

    Patch order is row-major ``(y, x)`` = ``(latent_h, latent_w)`` to match the
    packed visual-token order ``(t, h, w)`` used by the rest of the model
    (same convention as PRoPE in ``UcpeSideBranch``: patches_x=latent_w,
    patches_y=latent_h).
    """
    num_patches = patches_x * patches_y
    u_base, v_base = torch.meshgrid(
        torch.arange(patches_x, device=device, dtype=dtype),
        torch.arange(patches_y, device=device, dtype=dtype),
        indexing="xy",
    )  # both (patches_y, patches_x)
    coords = []
    for ox, oy in offsets:
        u = ((u_base + ox) / patches_x) - 0.5
        v = ((v_base + oy) / patches_y) - 0.5
        coords.append(torch.stack([u, v], dim=-1).reshape(-1, 2))  # (num_patches, 2)
    coords = torch.stack(coords, dim=1)  # (num_patches, num_rays, 2)
    ones = torch.ones_like(coords[..., :1])
    zeros = torch.zeros_like(coords[..., :1])
    return torch.cat([coords, ones, zeros], dim=-1)  # (num_patches, num_rays, 4)


def _get_frequency(num_freqs: int, max_period: float, min_period: float, device) -> torch.Tensor:
    """Log-spaced angular frequencies in ``[2π/max_period, 2π/min_period]``."""
    two_pi = 2.0 * torch.pi
    log_min = torch.log(torch.tensor(two_pi / max_period))
    log_max = torch.log(torch.tensor(two_pi / min_period))
    log_freqs = torch.linspace(log_min, log_max, num_freqs)
    return torch.exp(log_freqs).to(device)


# Per-coordinate-type base periods. Camera centers (p0) live in the
# translation-normalized world (max ‖t‖ ≤ 1), ray directions (pinf) are unit
# vectors; both in roughly [-1, 1]. Values follow the RayRoPE reference's
# global-ray defaults.
_MAX_PERIOD = {"p0": 4.0, "pinf": 8.0}


def _compute_cos_sin(
    *,
    viewmats: torch.Tensor,  # (cameras, 4, 4) camera<-world
    Ks: torch.Tensor,        # (cameras, 3, 3) pinhole
    patches_x: int,
    patches_y: int,
    image_width: int,
    image_height: int,
    num_rays_per_patch: int,
    num_freqs: int,
    freq_base: float,
    use_p0: bool,
    use_pinf: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build per-token RoPE cos/sin for one sample.

    Returns ``(cos, sin)`` each shaped ``(seqlen, head_dim // 2)`` where
    ``seqlen = cameras * patches_x * patches_y`` and
    ``head_dim // 2 = num_freqs * coord_dim``, in packed ``(t, h, w)`` order.
    All geometry runs in fp32 for stability.
    """
    device = viewmats.device
    wd = torch.float32
    cameras = viewmats.shape[0]
    num_patches = patches_x * patches_y
    offsets = _ray_offsets(num_rays_per_patch)

    w2cs = viewmats.to(wd)
    c2ws = _invert_SE3(w2cs)  # (cameras, 4, 4)
    Ks_norm = normalize_K(Ks.to(wd), image_width, image_height)
    # P_inv = c2w @ lift(K^-1): maps normalized-pixel homogeneous coords to
    # world. For a point at infinity ([u, v, 1, 0]) this yields the world-space
    # ray direction (w=0 ⇒ translation drops out).
    P_inv = torch.einsum("cij,cjk->cik", c2ws, _lift_K(_invert_K(Ks_norm)))

    positions = []  # list of (max_period, tensor (cameras, num_patches, d))

    if use_p0:
        # Camera center in world coords = c2w translation column, broadcast to
        # every patch in the frame.
        p0 = c2ws[:, :3, 3]  # (cameras, 3)
        p0 = p0[:, None, :].expand(cameras, num_patches, 3).contiguous()
        positions.append(("p0", p0))

    if use_pinf:
        pinf_local = _pinf_local_coords(patches_x, patches_y, offsets, device, wd)
        # (num_patches, num_rays, 4) -> broadcast over cameras -> world dirs.
        pinf_world = torch.einsum(
            "cij,prj->cpri", P_inv, pinf_local
        )  # (cameras, num_patches, num_rays, 4)
        pinf_dir = pinf_world[..., :3]
        pinf_dir = pinf_dir / (pinf_dir.norm(dim=-1, keepdim=True) + 1e-6)
        pinf_dir = pinf_dir.reshape(cameras, num_patches, num_rays_per_patch * 3)
        positions.append(("pinf", pinf_dir))

    # Build rope angles per coordinate type, then concat along the coord axis.
    angles = []  # each (cameras, num_patches, num_freqs, d)
    for name, pos in positions:
        max_period = _MAX_PERIOD[name]
        min_period = max_period / (freq_base ** (num_freqs - 1))
        freqs = _get_frequency(num_freqs, max_period, min_period, device)  # (num_freqs,)
        # outer product over freqs × coords -> (cameras, num_patches, num_freqs, d)
        angles.append(torch.einsum("f,cpd->cpfd", freqs, pos))
    rope_angle = torch.cat(angles, dim=-1)  # (cameras, num_patches, num_freqs, coord_dim)
    # -> (seqlen, num_freqs * coord_dim) = (seqlen, head_dim // 2)
    rope_angle = rope_angle.reshape(cameras * num_patches, -1)
    return torch.cos(rope_angle), torch.sin(rope_angle)


def _apply_rope_coeffs(
    feats: torch.Tensor,  # (num_heads, seqlen, head_dim)
    cos: torch.Tensor,    # (seqlen, head_dim // 2)
    sin: torch.Tensor,    # (seqlen, head_dim // 2)
    inverse: bool = False,
) -> torch.Tensor:
    """Split-convention rotary. Pairs feature dim ``i`` with ``i+head_dim//2``
    and rotates by the per-token angle. Exactly norm-preserving (a rotation),
    so no RMS clamp is needed (unlike PRoPE's non-orthogonal projmats)."""
    cos = cos.unsqueeze(0)  # (1, seqlen, head_dim // 2)
    sin = sin.unsqueeze(0)
    x1, x2 = torch.chunk(feats, 2, dim=-1)
    if inverse:
        r1 = x1 * cos + x2 * sin
        r2 = -x1 * sin + x2 * cos
    else:
        r1 = x1 * cos - x2 * sin
        r2 = x1 * sin + x2 * cos
    return torch.cat([r1, r2], dim=-1)


def pick_attn_dims(
    model_dim: int,
    coord_dim: int,
    attn_compress: int = 8,
    target_head_dim: int = 48,
) -> Tuple[int, int, int, int]:
    """Choose ``(attn_dim, num_heads, head_dim, num_freqs)`` for the side branch.

    RayRoPE requires ``head_dim % (2 * coord_dim) == 0`` so every feature dim
    is covered by exactly one 2-D rotation (norm-preserving). We pick a
    ``head_dim`` near ``target_head_dim`` that satisfies this, then enough heads
    to land the branch's attention width near ``model_dim // attn_compress``
    (the 8× compression UCPE uses, ~<1% trainable params/block).
    """
    step = 2 * coord_dim
    num_freqs = max(1, round(target_head_dim / step))
    head_dim = step * num_freqs
    target_attn = max(head_dim, model_dim // attn_compress)
    num_heads = max(1, round(target_attn / head_dim))
    attn_dim = num_heads * head_dim
    return attn_dim, num_heads, head_dim, num_freqs


class RayRoPESideBranch(nn.Module):
    """Per-block softmax side-branch attention with depth-free global-ray RoPE.

    Per-sample dispatch (mirrors ``UcpeSideBranch``): visual tokens arrive
    packed as ``(sum_T, D)`` with ``visual_cu_seqlens`` batch boundaries. We
    slice per sample, run a single ``(1, num_heads, T_i, head_dim)`` ray-rotary
    attention, and write back into the packed layout.
    """

    def __init__(
        self,
        model_dim: int,
        head_dim: int,  # base model head_dim (sum of axes_dims) — unused for sizing, kept for signature parity
        num_rays_per_patch: int = 1,
        freq_base: float = 3.0,
        attn_compress: int = 8,
        use_p0: bool = True,
        use_pinf: bool = True,
        use_scale_token: bool = False,
    ):
        super().__init__()
        if not (use_p0 or use_pinf):
            raise ValueError("RayRoPE needs at least one of use_p0 / use_pinf")
        self.num_rays_per_patch = int(num_rays_per_patch)
        self.freq_base = float(freq_base)
        self.use_p0 = bool(use_p0)
        self.use_pinf = bool(use_pinf)

        # Coordinate budget: 3 for the camera center + 3·num_rays for the ray
        # directions. head_dim of the side-branch attention must be a multiple
        # of 2·coord_dim.
        coord_dim = 3 * int(use_p0) + self.num_rays_per_patch * 3 * int(use_pinf)
        self.coord_dim = coord_dim
        attn_dim, num_heads_attn, head_dim_attn, num_freqs = pick_attn_dims(
            model_dim, coord_dim, attn_compress=attn_compress
        )
        assert head_dim_attn == 2 * coord_dim * num_freqs

        self.model_dim = model_dim
        self.attn_dim = attn_dim
        self.head_dim = head_dim_attn
        self.num_heads = num_heads_attn
        self.num_freqs = num_freqs
        self.attn_compress = attn_compress

        # Pre-attn LayerNorm: same rationale as UcpeSideBranch — without it the
        # side branch sees the raw post-residual stream (O(1e4) by mid-network
        # in Kandinsky-5), saturating the softmax.
        self.input_norm = nn.LayerNorm(model_dim, elementwise_affine=False)

        self.q_proj = nn.Linear(model_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, attn_dim, bias=False)
        # QK RMSNorm over head_dim — required item (5): added "as we made it for
        # ucpe before" (matches UcpeSideBranch.q_norm/k_norm and the base
        # MultiheadSelfAttention). Bounds Q/K scale before the rotary so the
        # encoding sees a known input magnitude.
        self.q_norm = nn.RMSNorm(head_dim_attn)
        self.k_norm = nn.RMSNorm(head_dim_attn)
        self.out_proj = nn.Linear(attn_dim, model_dim, bias=True)
        self._zero_init_out_proj()

        # Scale token (optional; dit.params.use_scale_token): see scale_token.py.
        # Zero-init => step-0 output unchanged; None when off (no extra params).
        self.scale_embed = ScaleEmbedding(model_dim) if use_scale_token else None

    def _zero_init_out_proj(self):
        """Zero-init out_proj so the step-0 residual is identically zero; the
        pretrained backbone passes through unchanged until RayRoPE learns a
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
            frame_wh: list of B (frame_w, frame_h) tuples — raw pixel size used
                to normalize intrinsics.

        Returns:
            (sum_T, D) zero-init residual contribution; add it to the block's
            running ``visual_embed``.
        """
        cu = visual_cu_seqlens
        batch = len(viewmats_per_sample)
        assert cu.numel() == batch + 1, (
            f"cu_seqlens has {cu.numel() - 1} samples, expected {batch}"
        )

        # Pre-norm + project q/k/v on the full packed tensor once. nn.LayerNorm
        # / nn.RMSNorm auto-promote to fp32 under autocast; type_as casts back
        # so SDPA stays on the flash kernel.
        x_normed = self.input_norm(x_packed).type_as(x_packed)
        q_packed = self.q_proj(x_normed).view(-1, self.num_heads, self.head_dim)
        k_packed = self.k_proj(x_normed).view(-1, self.num_heads, self.head_dim)
        v_packed = self.v_proj(x_normed).view(-1, self.num_heads, self.head_dim)
        q_packed = self.q_norm(q_packed).type_as(q_packed)
        k_packed = self.k_norm(k_packed).type_as(k_packed)

        out_packed = torch.zeros(
            x_packed.shape[0], self.num_heads, self.head_dim,
            device=x_packed.device, dtype=x_packed.dtype,
        )
        attn_dtype = q_packed.dtype  # bf16 under autocast

        for i in range(batch):
            start = int(cu[i].item())
            end = int(cu[i + 1].item())
            T_i = end - start
            latent_t, latent_h, latent_w = latent_shapes[i]
            assert T_i == latent_t * latent_h * latent_w, (
                f"sample {i}: packed T={T_i} != latent_t*h*w="
                f"{latent_t}*{latent_h}*{latent_w}={latent_t * latent_h * latent_w}"
            )
            frame_w, frame_h = frame_wh[i]

            viewmats_i = viewmats_per_sample[i].to(x_packed.device, torch.float32)
            Ks_i = Ks_per_sample[i].to(x_packed.device, torch.float32)
            # latent_t may be < viewmats length after patch-time downsampling;
            # trust latent_shapes (dit.py recomputes it from cu_seqlens).
            viewmats_i = viewmats_i[:latent_t]
            Ks_i = Ks_i[:latent_t]

            cos, sin = _compute_cos_sin(
                viewmats=viewmats_i,
                Ks=Ks_i,
                patches_x=latent_w,
                patches_y=latent_h,
                image_width=int(frame_w),
                image_height=int(frame_h),
                num_rays_per_patch=self.num_rays_per_patch,
                num_freqs=self.num_freqs,
                freq_base=self.freq_base,
                use_p0=self.use_p0,
                use_pinf=self.use_pinf,
            )  # (T_i, head_dim // 2) in fp32
            cos = cos.to(attn_dtype)
            sin = sin.to(attn_dtype)

            # (T_i, num_heads, head_dim) -> (num_heads, T_i, head_dim)
            q_i = q_packed[start:end].permute(1, 0, 2).contiguous()
            k_i = k_packed[start:end].permute(1, 0, 2).contiguous()
            v_i = v_packed[start:end].permute(1, 0, 2).contiguous()

            # Global-ray rotary (split convention). Apply the same per-token
            # rotation to q/k/v (inverse=True); un-rotate the output (the V·O
            # rotary trick, matching the RayRoPE reference global-ray apply_fns).
            q_t = _apply_rope_coeffs(q_i, cos, sin, inverse=True)
            k_t = _apply_rope_coeffs(k_i, cos, sin, inverse=True)
            v_t = _apply_rope_coeffs(v_i, cos, sin, inverse=True)

            attn_out = F.scaled_dot_product_attention(
                q_t.unsqueeze(0), k_t.unsqueeze(0), v_t.unsqueeze(0)
            ).squeeze(0)  # (num_heads, T_i, head_dim)
            attn_out = _apply_rope_coeffs(attn_out, cos, sin, inverse=False)

            # NaN guard (degenerate poses / intrinsics).
            attn_out = torch.where(
                torch.isfinite(attn_out), attn_out, torch.zeros_like(attn_out)
            )
            out_packed[start:end] = attn_out.permute(1, 0, 2)

        out = self.out_proj(out_packed.reshape(-1, self.attn_dim))
        out = apply_scale_token(out, self.scale_embed, trans_scale, cu, batch)
        return out
