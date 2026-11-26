"""Coordinate transforms shared by trajectory preprocessing and camera attention."""

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

_DEGEN_POSE_WARN_COUNT = 0


def _target_indices(source_indices, latent_t):
    if source_indices.shape[0] == 0:
        raise ValueError("source_indices must be non-empty")
    if latent_t <= 1:
        return np.asarray([source_indices[0]], dtype=np.float32)
    return np.linspace(source_indices[0], source_indices[-1], num=latent_t, dtype=np.float32)


def _sanitize_rotation_matrices(src_rot_mat: np.ndarray) -> np.ndarray:
    """Make rotation blocks safe for scipy ``Rotation.from_matrix`` (which runs
    an SVD that raises ``LinAlgError: SVD did not converge`` on NaN/Inf or
    badly-degenerate input). A single bad frame in one merged-dataset pose npz
    must not
    kill training. For each frame, if the 3x3 is non-finite OR far from a proper
    rotation (det not ~1, or not orthonormal), replace it with the previous
    valid rotation (identity for the first). Counts + rate-limited logs."""
    global _DEGEN_POSE_WARN_COUNT
    R = np.ascontiguousarray(src_rot_mat, dtype=np.float64).copy()
    n = R.shape[0]
    eye = np.eye(3, dtype=np.float64)
    last_good = eye.copy()
    n_fixed = 0
    for i in range(n):
        Ri = R[i]
        ok = np.all(np.isfinite(Ri))
        if ok:
            # proper-rotation sanity: R R^T ~ I and det ~ +1
            orth_err = np.abs(Ri @ Ri.T - eye).max()
            det = np.linalg.det(Ri) if np.all(np.isfinite(Ri)) else 0.0
            ok = bool(orth_err < 1e-2 and abs(det - 1.0) < 1e-1)
        if ok:
            last_good = Ri
        else:
            R[i] = last_good
            n_fixed += 1
    if n_fixed:
        _DEGEN_POSE_WARN_COUNT += 1
        if _DEGEN_POSE_WARN_COUNT <= 50 or _DEGEN_POSE_WARN_COUNT % 200 == 0:
            print(
                f"[camera_control] sanitized {n_fixed}/{n} degenerate rotation "
                f"frame(s) before SLERP (warn #{_DEGEN_POSE_WARN_COUNT})",
                flush=True,
            )
    return R


def interpolate_camera_poses(src_indices, poses, tgt_indices):
    src_rot_mat = _sanitize_rotation_matrices(poses[:, :3, :3])
    src_trans_vec = np.nan_to_num(
        np.asarray(poses[:, :3, 3], dtype=np.float64),
        nan=0.0, posinf=0.0, neginf=0.0,
    )

    if len(src_indices) == 1:
        single = np.zeros((1, 4, 4), dtype=np.float32)
        single[0, :3, :3] = src_rot_mat[0]
        single[0, :3, 3] = src_trans_vec[0]
        single[0, 3, 3] = 1.0
        tiled = np.repeat(single, len(tgt_indices), axis=0)
        return torch.from_numpy(tiled).float()

    interp_func_trans = interp1d(
        src_indices,
        src_trans_vec,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    interpolated_trans_vec = interp_func_trans(tgt_indices)

    # Defense-in-depth: _sanitize_rotation_matrices already removed non-finite /
    # degenerate frames, but guard the SVD-backed from_matrix anyway so a missed
    # edge case degrades to identity rotations instead of crashing the dataloader.
    try:
        src_quat_vec = Rotation.from_matrix(src_rot_mat)
    except np.linalg.LinAlgError:
        global _DEGEN_POSE_WARN_COUNT
        _DEGEN_POSE_WARN_COUNT += 1
        print(
            "[camera_control] Rotation.from_matrix SVD failed even after "
            "sanitization; falling back to identity rotations "
            f"(warn #{_DEGEN_POSE_WARN_COUNT})",
            flush=True,
        )
        src_rot_mat = np.repeat(np.eye(3, dtype=np.float64)[None], len(src_indices), axis=0)
        src_quat_vec = Rotation.from_matrix(src_rot_mat)
    quats = src_quat_vec.as_quat().copy()
    for idx in range(1, len(quats)):
        if np.dot(quats[idx], quats[idx - 1]) < 0:
            quats[idx] = -quats[idx]
    slerp_func_rot = Slerp(src_indices, Rotation.from_quat(quats))
    interpolated_rot_mat = slerp_func_rot(tgt_indices).as_matrix()

    out = np.zeros((len(tgt_indices), 4, 4), dtype=np.float32)
    out[:, :3, :3] = interpolated_rot_mat
    out[:, :3, 3] = interpolated_trans_vec
    out[:, 3, 3] = 1.0
    return torch.from_numpy(out).float()


def interpolate_intrinsics(src_indices, intrinsics, tgt_indices):
    if len(src_indices) == 1:
        tiled = np.repeat(intrinsics[:1], len(tgt_indices), axis=0)
        return torch.from_numpy(tiled).float()
    interp_func = interp1d(
        src_indices,
        intrinsics,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return torch.from_numpy(np.asarray(interp_func(tgt_indices), dtype=np.float32)).float()


def se3_inverse(transforms):
    rotation = transforms[:, :3, :3]
    translation = transforms[:, :3, 3:]
    rotation_inv = rotation.transpose(-1, -2)
    translation_inv = -torch.bmm(rotation_inv, translation)
    out = torch.eye(4, device=transforms.device, dtype=transforms.dtype)[None].repeat(
        transforms.shape[0], 1, 1
    )
    out[:, :3, :3] = rotation_inv
    out[:, :3, 3:] = translation_inv
    return out


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
    theta = torch.acos(cos_theta)                                    # (...), in [0, pi]
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
        diagA = torch.diagonal(A, dim1=-2, dim2=-1).clamp_min(0.0)    # (..., 3)
        k = diagA.argmax(dim=-1)                                      # (...)
        idx = k.reshape(k.shape + (1, 1)).expand(k.shape + (3, 1))
        col_k = torch.take_along_dim(A, idx, dim=-1).squeeze(-1)      # A[..., :, k] -> (..., 3)
        pivot = torch.take_along_dim(diagA, k.unsqueeze(-1), dim=-1)  # (..., 1)
        axis = col_k / torch.sqrt(pivot).clamp_min(eps)
        axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(eps)
        w_pi = theta.unsqueeze(-1) * axis
        out = torch.where(near_pi.unsqueeze(-1), w_pi, out)
    return out


def _se6_angular_velocity_from_rotations(rotations, dt: float):
    """Forward-difference angular velocity w_i = Log_SO3(R_i^T R_{i+1}) / dt (last copies previous)."""
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError(f"SE 6x6 angular-velocity dt must be positive, got {dt}")
    w = torch.zeros(rotations.shape[:-2] + (3,), device=rotations.device, dtype=rotations.dtype)
    if rotations.shape[0] > 1:
        rel = torch.einsum("...ji,...jk->...ik", rotations[:-1], rotations[1:])  # R_i^T R_{i+1}
        w[:-1] = _log_so3(rel) / dt
        w[-1] = w[-2]
    return w
