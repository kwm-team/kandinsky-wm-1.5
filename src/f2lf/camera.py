"""Build the metric path between two calibrated camera anchors."""

from dataclasses import dataclass

import numpy as np

WIDTH, HEIGHT = 768, 512
BODY_PLANES = 31


@dataclass(frozen=True)
class CameraPath:
    poses: np.ndarray
    viewmats: np.ndarray
    intrinsics: np.ndarray
    translation_scale: float


def camera_path(camera: dict, offset_m: float = 0.0) -> CameraPath:
    """Build the body and two endpoint camera slots from metric ENU poses."""
    if not isinstance(camera, dict) or camera.get("coordinate_system") != "metric_ENU" or camera.get("resolution") != [WIDTH, HEIGHT]:
        raise ValueError("Expected metric_ENU camera poses calibrated at 768×512")
    if not np.isfinite(offset_m) or abs(offset_m) > 5:
        raise ValueError("Camera offset must be within ±5 metres")
    a, b, intrinsics = (np.asarray(camera.get(key), dtype=float) for key in ("c2w_A", "c2w_B", "K"))
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("K must be a finite 3×3 calibration matrix")
    for pose in (a, b):
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("Both anchors require finite 4×4 camera-to-world matrices")
        rotation = pose[:3, :3]
        if (
            not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
        ):
            raise ValueError("Both camera-to-world matrices must be rigid transforms")
    if not np.allclose(a[:3, :3], b[:3, :3], atol=1e-8):
        raise ValueError("Both anchors must share the same heading and pitch")
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    if not (10 < fx < 10000 and 10 < fy < 10000 and 0 <= cx < WIDTH and 0 <= cy < HEIGHT
            and np.allclose(intrinsics[2], [0, 0, 1])):
        raise ValueError("K must use native pixel units and a principal point inside the image")

    delta = b[:3, 3] - a[:3, 3]
    distance = np.linalg.norm(delta[:2])
    if distance <= 1e-6:
        raise ValueError("F2LF requires nonzero horizontal separation between anchors")
    right = np.array([delta[1], -delta[0], 0]) / distance
    u = np.linspace(0, 1, BODY_PLANES)
    bump = 16 * u * u * (1 - u) * (1 - u)
    tangent = 32 * u * (1 - u) * (1 - 2 * u)
    poses = np.repeat(a[None], BODY_PLANES, axis=0)
    poses[:, :3, 3] = (1 - u[:, None]) * a[:3, 3] + u[:, None] * b[:3, 3] + offset_m * bump[:, None] * right
    for i, angle in enumerate(np.arctan2(offset_m * tangent, distance)):
        cos, sin = np.cos(angle), np.sin(angle)
        yaw = np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
        poses[i, :3, :3] = yaw @ a[:3, :3]
    poses[[0, -1]] = np.stack([a, b])

    relative = np.linalg.inv(a) @ poses
    radius = float(np.linalg.norm(relative[:, :3, 3], axis=1).max())
    if not 0 < radius < 150:
        raise ValueError("The trajectory must stay within 150 metres of the first anchor")
    relative[:, :3, 3] /= 2
    viewmats = np.linalg.inv(relative).astype(np.float32)
    return CameraPath(
        poses, np.concatenate([viewmats, viewmats[[0, -1]]]),
        np.repeat(intrinsics.astype(np.float32)[None], BODY_PLANES + 2, axis=0), radius,
    )
