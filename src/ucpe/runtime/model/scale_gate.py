"""Depthwise scale gate for UCPE side-branch residuals.

The gate is optional and identity-initialized:

    out[s:e] = out[s:e] * conv(out[s:e])

where conv operates over the token axis independently per channel. With zero
weights and bias=1, enabling the module preserves step-0 behavior exactly.
"""

import torch
from torch import nn


class ScaleGate(nn.Module):
    def __init__(self, model_dim: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ScaleGate kernel_size must be odd to preserve sequence length")
        self.conv = nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=model_dim,
            bias=True,
        )
        self._identity_init()

    def _identity_init(self):
        nn.init.zeros_(self.conv.weight)
        nn.init.ones_(self.conv.bias)

    def reset_parameters(self):
        self.conv.reset_parameters()
        self._identity_init()

    def forward(self, out_i: torch.Tensor) -> torch.Tensor:
        # out_i: (T, D). Conv1d expects (N, C, T).
        gate = self.conv(out_i.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
        return out_i * gate.type_as(out_i)


def apply_scale_gate(out, scale_gate, cu, batch):
    if scale_gate is None:
        return out
    chunks = []
    for i in range(batch):
        start = int(cu[i].item())
        end = int(cu[i + 1].item())
        chunks.append(scale_gate(out[start:end]))
    if not chunks:
        return out
    return torch.cat(chunks, dim=0)
