"""Lie-group helpers used by the camera side branch."""

import numpy as np
import torch


def _log_so3(rot, eps: float = 1e-5):
    """Batched SO(3) log map: rotation matrix (..., 3, 3) -> so(3) vector (..., 3) = axis * angle.

    Reads a rotation-rate out of a relative rotation. Handles the two numerically
    singular regimes: theta ~ 0 (falls back to the skew-symmetric ``vee``, exact to
    first order) and theta ~ pi (axis recovered from ``(R + I)/2 = a a^T``).
    """
    if rot.shape[-2:] != (3, 3):
        raise ValueError(f"_log_so3 expects (..., 3, 3), got {tuple(rot.shape)}")
    eye = torch.eye(3, device=rot.device, dtype=rot.dtype)
    trace = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)  # (...), in [0, pi]
    # vee(R - R^T) = 2 sin(theta) * axis
    vee = torch.stack(
        [
            rot[..., 2, 1] - rot[..., 1, 2],
            rot[..., 0, 2] - rot[..., 2, 0],
            rot[..., 1, 0] - rot[..., 0, 1],
        ],
        dim=-1,
    )
    sin_theta = torch.sin(theta)
    # Default (also the theta->0 limit): w = vee/2, exact to first order since
    # vee -> 2*theta*axis and theta/(2 sin theta) -> 1/2.
    out = 0.5 * vee
    generic = (theta > eps) & (theta < (float(np.pi) - eps))
    if generic.any():
        coeff = (theta / (2.0 * sin_theta)).unsqueeze(-1)
        out = torch.where(generic.unsqueeze(-1), coeff * vee, out)
    near_pi = theta >= (float(np.pi) - eps)
    if near_pi.any():
        # theta ~ pi: R + I = 2 a a^T. Recover axis from the strongest diagonal
        # column, magnitude = theta (sign of the pivot component is free).
        A = 0.5 * (rot + eye)
        diagA = torch.diagonal(A, dim1=-2, dim2=-1).clamp_min(0.0)  # (..., 3)
        k = diagA.argmax(dim=-1)  # (...)
        idx = k.reshape(k.shape + (1, 1)).expand(k.shape + (3, 1))
        col_k = torch.take_along_dim(A, idx, dim=-1).squeeze(
            -1
        )  # A[..., :, k] -> (..., 3)
        pivot = torch.take_along_dim(diagA, k.unsqueeze(-1), dim=-1)  # (..., 1)
        axis = col_k / torch.sqrt(pivot).clamp_min(eps)
        axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(
            eps
        )
        w_pi = theta.unsqueeze(-1) * axis
        out = torch.where(near_pi.unsqueeze(-1), w_pi, out)
    return out


def _se6_angular_velocity_from_rotations(rotations, dt: float):
    """Forward-difference angular velocity w_i = Log_SO3(R_i^T R_{i+1}) / dt (last copies previous)."""
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError(f"SE 6x6 angular-velocity dt must be positive, got {dt}")
    w = torch.zeros(
        rotations.shape[:-2] + (3,), device=rotations.device, dtype=rotations.dtype
    )
    if rotations.shape[0] > 1:
        rel = torch.einsum(
            "...ji,...jk->...ik", rotations[:-1], rotations[1:]
        )  # R_i^T R_{i+1}
        w[:-1] = _log_so3(rel) / dt
        w[-1] = w[-2]
    return w
