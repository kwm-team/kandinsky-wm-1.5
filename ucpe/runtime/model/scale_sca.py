"""Scale-conditioned Simplified Channel Attention for UCPE residuals.

NAFNet's SCA is:

    y = x * conv1x1(avg_pool(x))

Here ``x`` is the packed UCPE residual for one sample. We use the same channel
attention form, but concatenate a learned descriptor of ``trans_scale`` with the
pooled channel descriptor, then compress it back to model channels before the
1x1 projection so the branch can learn absolute motion magnitude through SCA.

The final projection is zero-initialized and the multiplier is ``1 + delta``;
therefore enabling this module is an exact identity at initialization.
"""

import torch
from torch import nn

from .utils import get_freqs


class ScaleSCA(nn.Module):
    def __init__(
        self,
        model_dim: int,
        freq_dim: int = 128,
        hidden_dim: int = 512,
        s_ref: float = 5.0,
        lo: float = 1e-2,
        hi: float = 740.0,
        max_period: float = 10000.0,
    ):
        super().__init__()
        self.s_ref = float(s_ref)
        self.lo = float(lo)
        self.hi = float(hi)
        self.register_buffer("freqs", get_freqs(freq_dim, max_period), persistent=False)
        self.scale_in = nn.Linear(2 * freq_dim, hidden_dim, bias=True)
        self.act = nn.SiLU()
        self.scale_out = nn.Linear(hidden_dim, model_dim, bias=True)
        self.descriptor_mlp = nn.Sequential(
            nn.Linear(2 * model_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, model_dim, bias=True),
        )
        self.channel_proj = nn.Conv1d(model_dim, model_dim, kernel_size=1, bias=True)
        self._identity_init()

    def _identity_init(self):
        nn.init.zeros_(self.channel_proj.weight)
        nn.init.zeros_(self.channel_proj.bias)

    def reset_parameters(self):
        self.scale_in.reset_parameters()
        self.scale_out.reset_parameters()
        for module in self.descriptor_mlp:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        self.channel_proj.reset_parameters()
        self._identity_init()

    def scale_descriptor(self, scale: torch.Tensor) -> torch.Tensor:
        scale = scale.float().clamp(min=0.0)
        winsor = torch.where(
            scale < self.lo, torch.zeros_like(scale), scale.clamp(self.lo, self.hi)
        )
        u = torch.log1p(winsor / self.s_ref)
        args = torch.outer(u, self.freqs.to(u.device))
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.scale_out(self.act(self.scale_in(emb)))

    def forward(self, out_i: torch.Tensor, scale_i: torch.Tensor) -> torch.Tensor:
        # out_i: (T, D). SCA descriptor is global average over tokens/channels.
        pooled = out_i.float().mean(dim=0, keepdim=True)
        scale_desc = self.scale_descriptor(scale_i.reshape(1)).to(pooled.device)
        descriptor = torch.cat([pooled, scale_desc], dim=-1)
        descriptor = self.descriptor_mlp(descriptor)
        descriptor = descriptor.to(dtype=self.channel_proj.weight.dtype)
        delta = self.channel_proj(descriptor.unsqueeze(-1)).squeeze(-1)
        gate = 1.0 + delta.type_as(out_i)
        return out_i * gate


def apply_scale_sca(out, scale_sca, trans_scale, cu, batch):
    if scale_sca is None or trans_scale is None:
        return out
    scale_t = torch.as_tensor(trans_scale, device=out.device, dtype=torch.float32)
    chunks = []
    for i in range(batch):
        start = int(cu[i].item())
        end = int(cu[i + 1].item())
        chunks.append(scale_sca(out[start:end], scale_t[i]))
    if not chunks:
        return out
    return torch.cat(chunks, dim=0)
