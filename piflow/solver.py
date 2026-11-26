"""PiFlow velocity policy and flow-matching rollout."""

import torch
import torch.nn.functional as F


def shift_timesteps(t, shift):
    return shift * t / (1 + (shift - 1) * t)


def build_solver_segments(
    nfe, final_step_size_scale, shift, eps, device, dtype=torch.float32
):
    """Build the descending DX solver grid used for policy inference."""
    if nfe < 1:
        raise ValueError(f"nfe must be positive, got {nfe}")
    final_step_size_scale = max(float(final_step_size_scale), eps)
    one_minus_final_scale = 1.0 - final_step_size_scale
    base_segment_size = 1.0 / (float(nfe) - one_minus_final_scale)
    final_step_size = final_step_size_scale * base_segment_size

    source_index = torch.arange(1, nfe + 1, device=device, dtype=dtype)
    raw_t_src = ((source_index - one_minus_final_scale) * base_segment_size).clamp(
        min=eps, max=1.0
    )
    segment_size = torch.where(
        source_index == 1,
        torch.full_like(raw_t_src, final_step_size),
        torch.full_like(raw_t_src, base_segment_size),
    )
    raw_t_dst = (raw_t_src - segment_size).clamp(min=0)

    raw_t_src = raw_t_src.flip(0)
    raw_t_dst = raw_t_dst.flip(0)
    segment_size = segment_size.flip(0)
    sigma_t_src = shift_timesteps(raw_t_src, shift)
    sigma_t_dst = shift_timesteps(raw_t_dst, shift)
    return raw_t_src, raw_t_dst, sigma_t_src, sigma_t_dst, segment_size


class PiFlowPolicy:
    """Interpolate a grid of x0 predictions over one solver segment."""

    def __init__(self, velocity_grid, x_t_src, sigma_t_src, segment_size, shift, eps):
        self.x_t_src = x_t_src
        self.ndim = x_t_src.dim()
        self.shift = float(shift)
        self.eps = float(eps)
        self.sigma_t_src = self._reshape_time(sigma_t_src)
        self.raw_t_src = self._unwarp_t(self.sigma_t_src)
        segment_size = self._reshape_time(segment_size)
        self.raw_t_dst = (self.raw_t_src - segment_size).clamp(min=0)
        self.segment_size = (self.raw_t_src - self.raw_t_dst).clamp(min=eps)
        self.x0_grid = (
            x_t_src.unsqueeze(1) - self.sigma_t_src.unsqueeze(1) * velocity_grid
        )

    def _reshape_time(self, value):
        return value.reshape(*value.shape, *((self.ndim - value.dim()) * [1]))

    def _unwarp_t(self, sigma_t):
        return sigma_t / (self.shift + (1 - self.shift) * sigma_t)

    @staticmethod
    def _interpolate(grid, position):
        n_grid = grid.size(1)
        if n_grid < 2:
            return grid.squeeze(1)
        position = position.clamp(0, 1) * (n_grid - 1)
        lower = position.floor().long().clamp(0, n_grid - 2)
        upper = lower + 1
        indices = torch.stack([lower, upper], dim=1)
        endpoints = torch.gather(
            grid, dim=1, index=indices.expand(-1, -1, *grid.shape[2:])
        )
        return (upper - position) * endpoints[:, 0] + (position - lower) * endpoints[
            :, 1
        ]

    def velocity(self, x_t, sigma_t):
        sigma_t = self._reshape_time(sigma_t)
        raw_t = self._unwarp_t(sigma_t)
        position = (raw_t - self.raw_t_dst) / self.segment_size
        x0 = self._interpolate(self.x0_grid, position)
        return (x_t - x0) / sigma_t.clamp(min=self.eps)

    def detached(self):
        detached = PiFlowPolicy.__new__(PiFlowPolicy)
        detached.x_t_src = self.x_t_src
        detached.ndim = self.ndim
        detached.shift = self.shift
        detached.eps = self.eps
        detached.sigma_t_src = self.sigma_t_src
        detached.raw_t_src = self.raw_t_src
        detached.raw_t_dst = self.raw_t_dst
        detached.segment_size = self.segment_size
        detached.x0_grid = self.x0_grid.detach()
        return detached


def rollout_policy(
    x_t_start, sigma_t_start, raw_t_start, raw_t_end, total_substeps, policy
):
    batch_size = x_t_start.size(0)
    ndim = x_t_start.dim()
    raw_t_start = raw_t_start.reshape(batch_size, *((ndim - 1) * [1]))
    raw_t_end = raw_t_end.reshape(batch_size, *((ndim - 1) * [1]))
    delta = raw_t_start - raw_t_end
    num_substeps = (delta * total_substeps).round().long().clamp(min=1)
    substep_size = delta / num_substeps

    raw_t = raw_t_start
    sigma_t = sigma_t_start
    x_t = x_t_start
    for substep_id in range(int(num_substeps.max().item())):
        velocity = policy.velocity(x_t, sigma_t)
        raw_t_next = (raw_t - substep_size).clamp(min=0)
        sigma_t_next = shift_timesteps(raw_t_next, policy.shift)
        x_t_next = x_t + velocity * (sigma_t_next - sigma_t)
        active = num_substeps > substep_id
        x_t = torch.where(active, x_t_next, x_t)
        sigma_t = torch.where(active, sigma_t_next, sigma_t)
        raw_t = torch.where(active, raw_t_next, raw_t)
    return x_t, sigma_t


class GridOutLayer(torch.nn.Module):
    """Runs `grid_points` independent OutLayer copies on the same pre-head features
    and stacks their outputs on a new axis at dim=1, so the result broadcasts against
    PiFlowPolicy's `x_t_src.unsqueeze(1)` convention: [frames, grid, Hlat, Wlat, C]."""

    def __init__(self, heads):
        super().__init__()
        self.heads = torch.nn.ModuleList(heads)

    def forward(
        self, visual_embed, text_embed, time_embed, visual_cu_seqlens, time_embed_idx
    ):
        outputs = [
            head(
                visual_embed, text_embed, time_embed, visual_cu_seqlens, time_embed_idx
            )
            for head in self.heads
        ]
        return torch.stack(outputs, dim=1)

    def reset_parameters(self):
        for head in self.heads:
            head.reset_parameters()
