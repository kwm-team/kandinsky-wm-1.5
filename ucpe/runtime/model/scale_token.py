"""Scale token: feed back the per-clip translation magnitude that
``camera_control.compute_relative_poses(normalize_trans=True)`` divides out, so a
camera side-branch can produce ABSOLUTE motion magnitude instead of only the
normalized direction/shape.

Shared by ``UcpeSideBranch`` / ``RayRoPESideBranch`` / ``CRePESideBranch`` and
gated per-model by the ``dit.params.use_scale_token`` flag (the branch builds a
``ScaleEmbedding`` only when the flag is on; otherwise ``scale_embed`` is ``None``
and ``apply_scale_token`` is a no-op, so the model is byte-identical to the
scale-blind branch and old checkpoints load unchanged).

Ported from kwm-team/kandinsky5-ucpe@3b1d544 (originally ``_ScaleEmbedding`` in
ucpe.py); generalised to all three side-branches.
"""

import torch
from torch import nn

from .utils import get_freqs


class ScaleEmbedding(nn.Module):
    """Embeds the per-clip translation magnitude (``max_norm`` discarded by
    ``normalize_trans``) into a FiLM ``(gamma, beta)`` over the branch residual.

    Design (from a 121,663-clip ViPE trans-norm audit; see commit 3b1d544):
      * ``max_disp`` spans ~6 orders of magnitude (p1=0.004, p50=5.07, p99=738,
        with a scale-collapse tail up to 3.46e6) -> a linear scalar is hopeless,
        so we ``log1p`` after winsorizing to ``[lo, hi]``.
      * ``s_ref`` defaults to the corpus median (5.0) so the bulk of clips land
        at u in ~[-1, +1], the useful band of the sinusoidal features.
      * ``out_layer`` is zero-init so at step 0 gamma=beta=0 and the branch is
        byte-identical to the scale-blind model -- magnitude is *learned*, never
        injected as noise at init.
    """

    def __init__(self, model_dim, freq_dim=128, hidden_dim=512,
                 s_ref=5.0, lo=1e-2, hi=740.0, max_period=10000.0):
        super().__init__()
        self.s_ref = float(s_ref)
        self.lo = float(lo)
        self.hi = float(hi)
        self.register_buffer("freqs", get_freqs(freq_dim, max_period), persistent=False)
        self.in_layer = nn.Linear(2 * freq_dim, hidden_dim, bias=True)
        self.act = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, 2 * model_dim, bias=True)  # gamma, beta
        self._zero_init()

    def _zero_init(self):
        nn.init.zeros_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    def reset_parameters(self):
        self.in_layer.reset_parameters()
        self.out_layer.reset_parameters()
        self._zero_init()

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, scale: torch.Tensor) -> torch.Tensor:
        # scale: (B,) >= 0 per-clip max displacement (ViPE ~m); 0 == static.
        scale = scale.float().clamp(min=0.0)
        # Winsorize so a few scale-collapse clips can't dominate the band;
        # near-static (< lo) collapses to u=0 just like the static path.
        winsor = torch.where(
            scale < self.lo, torch.zeros_like(scale), scale.clamp(self.lo, self.hi)
        )
        u = torch.log1p(winsor / self.s_ref)                 # (B,)
        args = torch.outer(u, self.freqs.to(u.device))       # (B, freq_dim)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, 2*freq_dim)
        return self.out_layer(self.act(self.in_layer(emb)))  # (B, 2*model_dim)


def apply_scale_token(out, scale_embed, trans_scale, cu, batch):
    """FiLM-modulate a packed side-branch residual ``out`` (sum_T, D) per sample
    by ``trans_scale``:

        out[s:e] = out[s:e] * (1 + gamma[i]) + beta[i]   for sample i's token span

    gamma scales the normalized-direction geometric residual by motion
    magnitude; beta is the additive scale token. Both are zero at init, so the
    branch output is unchanged at step 0.

    No-op (returns ``out`` unchanged) when the branch was built scale-blind
    (``scale_embed is None``) or no scale was provided (``trans_scale is None``),
    so this is safe to call unconditionally at every branch's output.

    Implemented out-of-place (expand per-sample gamma/beta to per-token via
    repeat_interleave, then one fused multiply-add) rather than the in-place
    ``out[s:e] = ...`` loop: gamma's gradient needs the original ``out`` values,
    so overwriting ``out`` in place breaks autograd ("variable modified by an
    inplace operation"). This vectorized form is autograd-safe and faster.
    """
    if scale_embed is None or trans_scale is None:
        return out
    scale_t = torch.as_tensor(trans_scale, device=out.device, dtype=torch.float32)
    gamma, beta = scale_embed(scale_t).chunk(2, dim=-1)  # each (B, D), 0 at init
    counts = (cu[1:] - cu[:-1]).to(device=gamma.device, dtype=torch.long)  # (B,) tokens/sample
    gamma_full = gamma.repeat_interleave(counts, dim=0)  # (sum_T, D)
    beta_full = beta.repeat_interleave(counts, dim=0)     # (sum_T, D)
    return out * (1.0 + gamma_full).type_as(out) + beta_full.type_as(out)
