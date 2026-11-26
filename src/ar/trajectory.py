"""Read metrically calibrated camera poses and select native latent anchors."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    camera_to_world: np.ndarray
    intrinsics: np.ndarray
    image_size: tuple[int, int] | None

    @property
    def frames(self) -> int:
        return len(self.camera_to_world) - 1


@dataclass(frozen=True)
class Window:
    start: int
    frames: int
    anchors: tuple[int, ...]
    reset_after: bool


def load_trajectory(path: Path) -> Trajectory:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("fps") != 24:
        raise ValueError("Expected a trajectory sampled at 24 fps")
    poses = np.asarray(data.get("camera_to_world"), dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 5:
        raise ValueError("camera_to_world must contain at least five 4x4 matrices")
    if (len(poses) - 1) % 4:
        raise ValueError(
            "Trajectory must contain frame zero plus a multiple of four output frames"
        )
    indices = data.get("frame_indices")
    if (
        not isinstance(indices, list)
        or any(type(index) is not int for index in indices)
        or indices != list(range(len(poses)))
    ):
        raise ValueError(
            "frame_indices must enumerate every RGB frame consecutively from zero"
        )
    rotation = poses[:, :3, :3]
    if (
        not np.isfinite(poses).all()
        or not np.allclose(poses[:, 3], (0, 0, 0, 1), atol=1e-6)
        or not np.allclose(rotation @ rotation.swapaxes(-1, -2), np.eye(3), atol=1e-4)
        or not np.allclose(np.linalg.det(rotation), 1, atol=1e-4)
    ):
        raise ValueError(
            "camera_to_world must use finite translations and proper rotations"
        )
    intrinsics = np.asarray(data.get("intrinsics"), dtype=np.float32)
    if intrinsics.shape == (3, 3):
        intrinsics = np.broadcast_to(intrinsics, (len(poses), 3, 3)).copy()
    if (
        intrinsics.shape != (len(poses), 3, 3)
        or not np.isfinite(intrinsics).all()
        or (intrinsics[:, 0, 0] <= 0).any()
        or (intrinsics[:, 1, 1] <= 0).any()
        or not np.allclose(intrinsics[:, 2], (0, 0, 1), atol=1e-6)
        or not np.allclose(intrinsics[:, 1, 0], 0, atol=1e-6)
    ):
        raise ValueError(
            "intrinsics must contain one or one-per-frame calibrated 3x3 matrix"
        )
    size = data.get("image_size")
    if size is not None and (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(value) is not int or value <= 0 for value in size)
    ):
        raise ValueError("image_size must be [width, height]")
    return Trajectory(poses, intrinsics, tuple(size) if size else None)


def rollout_windows(
    frames: int, *, chunk_planes: int = 4, reset_frames: int | None = 120
) -> Iterator[Window]:
    if frames < 4 or frames % 4:
        raise ValueError("Requested output length must be a positive multiple of four")
    if chunk_planes < 1 or (
        reset_frames is not None and (reset_frames < 4 or reset_frames % 4)
    ):
        raise ValueError("Chunk and reset lengths must align to latent planes")

    start = 0
    while start < frames:
        until_reset = reset_frames - start % reset_frames if reset_frames else frames
        visible = min(4 * chunk_planes, frames - start, until_reset)
        anchors = tuple(min(frames, start + 4 * i) for i in range(1, chunk_planes + 1))
        yield Window(
            start, visible, anchors, visible == until_reset and reset_frames is not None
        )
        start += visible
