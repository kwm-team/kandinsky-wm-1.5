"""Per-block CRePE side-branch attention for Kandinsky-5.

CRePE (Camera Ray Positional Encoding with radial distance, arXiv 2605.12938)
extends the UCPE/RayRoPE ray-only encoding with a learned **per-token radial
distance** so the rotary encoding knows *where along the ray* the scene lies,
not just the ray direction. A small geometry head predicts a log radial-distance
center ``mu`` and interval ``sigma`` per token; CRePE then integrates the rotary
positional encoding analytically over the predicted (uniform-in-log-r) distance
distribution — the *expected* phasor — instead of placing RoPE at a single
position. Uncertain tokens (large sigma) get an attenuated (shrunk) phasor.

This is implemented in the same shape as ``RayRoPESideBranch`` (zero-init
``out_proj`` so the step-0 residual is identically zero, QK-RMSNorm, per-sample
single SDPA), and reuses the RayRoPE depth-free geometry (camera center ``p0`` +
ray direction ``pinf``) plus the new radial point ``pd``. It consumes the same
per-frame ``ucpe_inputs`` trajectory as UCPE/RayRoPE.

Deviations from the paper (deliberate, scoped — CRePE is wired but OFF by
default in the current config; training runs on RayRoPE):
- **Global single-pass** (like our RayRoPE branch) rather than the paper's
  per-query-frame projective-relative form: geometry is encoded in a common
  frame-0 world frame and relativity emerges in the q·k dot product. The
  expected pd phasor is applied to Q and K only (apply_vo=False), matching the
  paper's "modulate K before computing attention scores".
- **Pinhole** intrinsics only (Kandinsky data is rectified): the UCM curved
  projected path collapses to a straight ray, so the radial point is
  ``X(r) = o + r·d`` in world coords. The radial coordinate fed to RoPE is this
  3-D point (vs the paper's bounded ``(u, v, ‖X_q‖)`` query-frame coordinate).
- **No radial supervision**: the head is trained end-to-end by the diffusion
  loss only; the paper's optional pseudo-depth (UniK3D) supervision and Radial
  MixForcing control path are left as follow-ups.

References:
- CRePE, arXiv 2605.12938 (the geometry head + expected-RoPE recipe).
- RayRoPE, arXiv 2601.15275 (the expected-phasor-over-an-interval engine and the
  global-ray geometry reused here from ``kandinsky/model/rayrope.py``).
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .scale_token import ScaleEmbedding, apply_scale_token
from .rayrope import (
    _MAX_PERIOD,
    _apply_rope_coeffs,
    _get_frequency,
    _invert_SE3,
    _invert_K,
    _lift_K,
    _pinf_local_coords,
    _ray_offsets,
    normalize_K,
    pick_attn_dims,
)

# Radial point shares the camera-center frequency band (world-space 3-D coords).
_MAX_PERIOD_PD = _MAX_PERIOD["p0"]


def _expected_phasor(theta_k: torch.Tensor, eps: float = 1e-6):
    """Analytic expected (cos, sin) of a phase that varies piecewise-linearly
    across ``K`` breakpoints along axis ``-3``.

    ``theta_k`` shape ``(..., K, F, C)``. For each adjacent breakpoint pair the
    segment-mean of cos/sin is ``(sin θ_{k+1} − sin θ_k)/Δθ`` and
    ``(cos θ_k − cos θ_{k+1})/Δθ``; averaging over the ``K−1`` equal-mass
    segments (uniform-in-log-r) gives the expectation. Degenerate Δθ≈0 segments
    fall back to the exact ``(cos θ_k, sin θ_k)``.
    """
    th0 = theta_k[..., :-1, :, :]
    th1 = theta_k[..., 1:, :, :]
    d = th1 - th0
    small = d.abs() < eps
    d_safe = torch.where(small, torch.ones_like(d), d)
    c_seg = torch.where(small, torch.cos(th0), (torch.sin(th1) - torch.sin(th0)) / d_safe)
    s_seg = torch.where(small, torch.sin(th0), (torch.cos(th0) - torch.cos(th1)) / d_safe)
    return c_seg.mean(dim=-3), s_seg.mean(dim=-3)  # (..., F, C)


def _compute_crepe_cos_sin(
    *,
    viewmats: torch.Tensor,  # (cameras, 4, 4) camera<-world
    Ks: torch.Tensor,        # (cameras, 3, 3) pinhole
    mu: torch.Tensor,        # (cameras*num_patches,) log radial-distance center
    sigma: torch.Tensor,     # (cameras*num_patches,) log radial interval
    patches_x: int,
    patches_y: int,
    image_width: int,
    image_height: int,
    num_rays_per_patch: int,
    num_freqs: int,
    freq_base: float,
    num_breakpoints: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token CRePE cos/sin coeffs for one sample.

    Returns ``(cos, sin)`` each ``(seqlen, head_dim // 2)`` in packed
    ``(t, h, w)`` order, with coordinate blocks ``[p0 | pinf | pd]``. ``p0`` and
    ``pinf`` use exact rotary (norm-preserving); ``pd`` uses the expected phasor
    over the predicted radial-distance distribution (magnitude ≤ 1).
    """
    device = viewmats.device
    wd = torch.float32
    cameras = viewmats.shape[0]
    num_patches = patches_x * patches_y
    offsets = _ray_offsets(num_rays_per_patch)
    R = num_rays_per_patch
    K = int(num_breakpoints)

    w2cs = viewmats.to(wd)
    c2ws = _invert_SE3(w2cs)
    Ks_norm = normalize_K(Ks.to(wd), image_width, image_height)
    P_inv = torch.einsum("cij,cjk->cik", c2ws, _lift_K(_invert_K(Ks_norm)))

    # Camera center per frame, broadcast to patches.
    o_world = c2ws[:, :3, 3]  # (cameras, 3)
    p0 = o_world[:, None, :].expand(cameras, num_patches, 3)  # (cameras, P, 3)

    # World ray directions per (patch, ray).
    pinf_local = _pinf_local_coords(patches_x, patches_y, offsets, device, wd)  # (P, R, 4)
    pinf_world = torch.einsum("cij,prj->cpri", P_inv, pinf_local)  # (cameras, P, R, 4)
    d_world = pinf_world[..., :3]
    d_world = d_world / (d_world.norm(dim=-1, keepdim=True) + 1e-6)  # (cameras, P, R, 3)
    pinf_flat = d_world.reshape(cameras, num_patches, R * 3)

    # Radial breakpoints from (mu, sigma): z_k uniform in log-r, r_k = exp(z_k).
    mu_g = mu.reshape(cameras, num_patches, 1)
    sig_g = sigma.reshape(cameras, num_patches, 1).abs()
    ks = torch.arange(K, device=device, dtype=wd) / max(K - 1, 1)  # (K,)
    z_k = mu_g - sig_g + ks.view(1, 1, K) * (2.0 * sig_g)  # (cameras, P, K)
    r_k = torch.exp(z_k)
    # Radial points X_k = o + r_k · d_world: (cameras, P, R, K, 3).
    X_k = (
        o_world[:, None, None, None, :]
        + r_k[:, :, None, :, None] * d_world[:, :, :, None, :]
    )

    # Frequencies per coordinate type.
    def freqs(max_period):
        min_period = max_period / (freq_base ** (num_freqs - 1))
        return _get_frequency(num_freqs, max_period, min_period, device)

    f_p0 = freqs(_MAX_PERIOD["p0"])
    f_pinf = freqs(_MAX_PERIOD["pinf"])
    f_pd = freqs(_MAX_PERIOD_PD)

    # Exact rotary for p0, pinf.
    ang_p0 = torch.einsum("cpd,f->cpfd", p0, f_p0)            # (cameras, P, F, 3)
    ang_pinf = torch.einsum("cpd,f->cpfd", pinf_flat, f_pinf)  # (cameras, P, F, R*3)
    cos_p0, sin_p0 = torch.cos(ang_p0), torch.sin(ang_p0)
    cos_pinf, sin_pinf = torch.cos(ang_pinf), torch.sin(ang_pinf)

    # Expected rotary for pd (radial point) over the K breakpoints.
    ang_pd = torch.einsum("cprkd,f->cprkfd", X_k, f_pd)  # (cameras, P, R, K, F, 3)
    cbar, sbar = _expected_phasor(ang_pd)  # (cameras, P, R, F, 3)
    # -> (cameras, P, F, R*3)
    cbar = cbar.permute(0, 1, 3, 2, 4).reshape(cameras, num_patches, num_freqs, R * 3)
    sbar = sbar.permute(0, 1, 3, 2, 4).reshape(cameras, num_patches, num_freqs, R * 3)

    cos = torch.cat([cos_p0, cos_pinf, cbar], dim=-1)  # (cameras, P, F, coord_dim)
    sin = torch.cat([sin_p0, sin_pinf, sbar], dim=-1)
    cos = cos.reshape(cameras * num_patches, -1)  # (T, F*coord_dim) = (T, head_dim//2)
    sin = sin.reshape(cameras * num_patches, -1)
    return cos, sin


class CRePESideBranch(nn.Module):
    """Per-block softmax side-branch with CRePE radial expected-RoPE.

    Same per-sample dispatch as ``RayRoPESideBranch`` (zero-init out_proj,
    QK-RMSNorm). Adds a per-token geometry head predicting (mu, sigma); the
    radial coordinate ``pd`` is encoded with the expected phasor over the
    predicted distance distribution. The expected rotary is applied to Q and K
    only (apply_vo=False — modulates attention scores, matching the paper).
    """

    def __init__(
        self,
        model_dim: int,
        head_dim: int,  # base head_dim — kept for signature parity, unused for sizing
        num_rays_per_patch: int = 1,
        freq_base: float = 3.0,
        attn_compress: int = 8,
        num_breakpoints: int = 8,
        sigma_max: float = 3.0,
        use_scale_token: bool = False,
    ):
        super().__init__()
        self.num_rays_per_patch = int(num_rays_per_patch)
        self.freq_base = float(freq_base)
        self.num_breakpoints = int(num_breakpoints)
        self.sigma_max = float(sigma_max)

        # p0 (3) + pinf (3·rays) + pd radial point (3·rays).
        coord_dim = 3 + self.num_rays_per_patch * 3 + self.num_rays_per_patch * 3
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

        self.input_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.q_proj = nn.Linear(model_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, attn_dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim_attn)
        self.k_norm = nn.RMSNorm(head_dim_attn)
        self.out_proj = nn.Linear(attn_dim, model_dim, bias=True)

        # Geometry head g_φ: LayerNorm -> Linear -> SiLU -> Linear, hidden
        # max(16, D/4). Final Linear is zero-init (weight AND bias) so at init
        # mu=0 (radial distance 1); the sigma init offset (=> sigma=sigma_max,
        # paper's max-uncertainty init) is applied as a constant in
        # _geometry_head, NOT as a bias value — copying a plain tensor into an
        # FSDP-sharded DTensor bias is illegal, so the bias must stay zero-init.
        hidden = max(16, model_dim // 4)
        self.geo_norm = nn.LayerNorm(model_dim, elementwise_affine=True)
        self.geo_fc1 = nn.Linear(model_dim, hidden)
        self.geo_act = nn.SiLU()
        self.geo_fc2 = nn.Linear(hidden, 2)
        self._init_weights()

        # Scale token (optional; dit.params.use_scale_token): see scale_token.py.
        # Zero-init => step-0 output unchanged; None when off (no extra params).
        self.scale_embed = ScaleEmbedding(model_dim) if use_scale_token else None

    def _init_weights(self):
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # DTensor/FSDP-safe (zeros_ works on sharded params; copy_ of a plain
        # tensor does not). The sigma_max init is added in _geometry_head.
        nn.init.zeros_(self.geo_fc2.weight)
        nn.init.zeros_(self.geo_fc2.bias)

    def reset_parameters(self):
        self.input_norm.reset_parameters()
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        self.q_norm.reset_parameters()
        self.k_norm.reset_parameters()
        self.out_proj.reset_parameters()
        self.geo_norm.reset_parameters()
        self.geo_fc1.reset_parameters()
        self.geo_fc2.reset_parameters()
        self._init_weights()
        if self.scale_embed is not None:
            self.scale_embed.reset_parameters()

    def _geometry_head(self, x_packed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.geo_norm(x_packed.float()).type_as(x_packed)
        h = self.geo_act(self.geo_fc1(h))
        ms = self.geo_fc2(h).float()  # (sum_T, 2)
        mu = ms[..., 0]
        # +sigma_max constant offset => the zero-init head starts at
        # sigma=sigma_max (paper's max-uncertainty init); the learnable bias then
        # sharpens it. clamp (not tanh) keeps the init exact and the interval
        # bounded to [-sigma_max, sigma_max] (paper: [-3, 3]).
        sigma = (ms[..., 1] + self.sigma_max).clamp(-self.sigma_max, self.sigma_max)
        return mu, sigma

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
        cu = visual_cu_seqlens
        batch = len(viewmats_per_sample)
        assert cu.numel() == batch + 1

        x_normed = self.input_norm(x_packed).type_as(x_packed)
        q_packed = self.q_proj(x_normed).view(-1, self.num_heads, self.head_dim)
        k_packed = self.k_proj(x_normed).view(-1, self.num_heads, self.head_dim)
        v_packed = self.v_proj(x_normed).view(-1, self.num_heads, self.head_dim)
        q_packed = self.q_norm(q_packed).type_as(q_packed)
        k_packed = self.k_norm(k_packed).type_as(k_packed)
        mu_all, sigma_all = self._geometry_head(x_packed)

        out_packed = torch.zeros(
            x_packed.shape[0], self.num_heads, self.head_dim,
            device=x_packed.device, dtype=x_packed.dtype,
        )
        attn_dtype = q_packed.dtype

        for i in range(batch):
            start = int(cu[i].item())
            end = int(cu[i + 1].item())
            T_i = end - start
            latent_t, latent_h, latent_w = latent_shapes[i]
            assert T_i == latent_t * latent_h * latent_w
            frame_w, frame_h = frame_wh[i]

            viewmats_i = viewmats_per_sample[i].to(x_packed.device, torch.float32)[:latent_t]
            Ks_i = Ks_per_sample[i].to(x_packed.device, torch.float32)[:latent_t]

            cos, sin = _compute_crepe_cos_sin(
                viewmats=viewmats_i,
                Ks=Ks_i,
                mu=mu_all[start:end],
                sigma=sigma_all[start:end],
                patches_x=latent_w,
                patches_y=latent_h,
                image_width=int(frame_w),
                image_height=int(frame_h),
                num_rays_per_patch=self.num_rays_per_patch,
                num_freqs=self.num_freqs,
                freq_base=self.freq_base,
                num_breakpoints=self.num_breakpoints,
            )
            cos = cos.to(attn_dtype)
            sin = sin.to(attn_dtype)

            q_i = q_packed[start:end].permute(1, 0, 2).contiguous()
            k_i = k_packed[start:end].permute(1, 0, 2).contiguous()
            v_i = v_packed[start:end].permute(1, 0, 2).contiguous()
            # Modulate Q and K only (apply_vo=False); V/output unrotated so the
            # geometry biases attention scores. p0/pinf rotary is exact;
            # the pd block carries the (contractive) expected phasor.
            q_t = _apply_rope_coeffs(q_i, cos, sin, inverse=True)
            k_t = _apply_rope_coeffs(k_i, cos, sin, inverse=True)

            # CRePE's attention head_dim (= 2*coord_dim*num_freqs, e.g. 54) is
            # frequently NOT a multiple of 8, which disqualifies the fused
            # flash / mem-efficient SDPA kernels and forces the ~8x slower math
            # fallback (the dominant cost of this branch — UCPE/RayRoPE land on
            # multiple-of-8 head_dims and stay on flash). Zero-pad the head dim
            # up to the next multiple of 8 so a fused kernel runs: padding Q/K
            # with zeros leaves every score unchanged (the extra dims add 0 to
            # the dot product) and the padded value columns are sliced back off
            # the output. Pass the TRUE-dim scale so softmax matches the
            # unpadded result exactly (default would use 1/sqrt(padded_dim)).
            head_dim = q_t.shape[-1]
            pad = (-head_dim) % 8
            if pad:
                q_t = F.pad(q_t, (0, pad))
                k_t = F.pad(k_t, (0, pad))
                v_sdpa = F.pad(v_i, (0, pad))
            else:
                v_sdpa = v_i
            attn_out = F.scaled_dot_product_attention(
                q_t.unsqueeze(0), k_t.unsqueeze(0), v_sdpa.unsqueeze(0),
                scale=head_dim ** -0.5,
            ).squeeze(0)
            if pad:
                attn_out = attn_out[..., :head_dim]
            attn_out = torch.where(
                torch.isfinite(attn_out), attn_out, torch.zeros_like(attn_out)
            )
            out_packed[start:end] = attn_out.permute(1, 0, 2)

        out = self.out_proj(out_packed.reshape(-1, self.attn_dim))
        out = apply_scale_token(out, self.scale_embed, trans_scale, cu, batch)
        return out
