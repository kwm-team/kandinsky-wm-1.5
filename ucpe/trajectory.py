"""Read camera-to-world trajectories and build the camera attention inputs."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from .runtime.camera_control import (
    _target_indices,
    interpolate_camera_poses,
    interpolate_intrinsics,
    se3_inverse,
)


@dataclass(frozen=True)
class Trajectory:
    camera_to_world: np.ndarray
    intrinsics: np.ndarray
    frame_indices: np.ndarray
    fps: float
    image_size: tuple[int, int] | None = None

    @property
    def frames(self) -> int:
        return int(self.frame_indices[-1]) + 1

    @classmethod
    def from_file(cls, path: str | Path) -> "Trajectory":
        path = Path(path)
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as data:
                return cls._parse(data)
        if path.suffix.lower() == ".json":
            return cls._parse(json.loads(path.read_text(encoding="utf-8")))
        raise ValueError("Trajectory files must use .json or .npz")

    @classmethod
    def _parse(cls, data) -> "Trajectory":
        poses = np.asarray(data["camera_to_world"], dtype=np.float32)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses):
            raise ValueError("camera_to_world must have shape [N, 4, 4]")
        if not np.isfinite(poses).all() or not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5):
            raise ValueError("camera_to_world must contain finite homogeneous transforms")
        rotation = poses[:, :3, :3]
        if not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-3) or not np.allclose(
            np.linalg.det(rotation), 1, atol=1e-3
        ):
            raise ValueError("camera_to_world must contain proper rotation matrices")
        intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)
        if intrinsics.shape == (3, 3):
            intrinsics = np.broadcast_to(intrinsics, (len(poses), 3, 3))
        if intrinsics.shape == (len(poses), 3, 3):
            if not np.allclose(intrinsics[:, 2], [0, 0, 1], atol=1e-6) or not np.allclose(
                intrinsics[:, [0, 1], [1, 0]], 0, atol=1e-6
            ):
                raise ValueError("intrinsics must describe a pinhole camera without skew")
            intrinsics = intrinsics[:, [0, 1, 0, 1], [0, 1, 2, 2]]
        if intrinsics.shape == (4,):
            intrinsics = np.broadcast_to(intrinsics, (len(poses), 4))
        if intrinsics.shape != (len(poses), 4) or not np.isfinite(intrinsics).all() or np.any(intrinsics[:, :2] <= 0):
            raise ValueError("intrinsics must provide positive focal lengths and finite principal points")
        indices = np.asarray(data.get("frame_indices", np.arange(len(poses))), dtype=np.float32)
        if (
            indices.shape != (len(poses),)
            or not np.isfinite(indices).all()
            or np.any(indices != np.floor(indices))
            or indices[0] != 0
            or np.any(np.diff(indices) <= 0)
        ):
            raise ValueError("frame_indices must start at zero and strictly increase as integers")
        fps = float(data.get("fps", 24))
        if fps != 24:
            raise ValueError("This checkpoint expects a 24 fps trajectory")
        image_size = None
        if data.get("image_size") is not None:
            size = np.asarray(data["image_size"])
            if size.shape != (2,) or size.dtype.kind not in "iu" or np.any(size <= 0):
                raise ValueError("image_size must contain positive integer [width, height] values")
            image_size = (int(size[0]), int(size[1]))
        return cls(poses, intrinsics, indices, fps, image_size)

    def camera_inputs(self, latent_shape: tuple[int, int, int], image_size: tuple[int, int]) -> dict:
        if self.image_size is not None and self.image_size != image_size:
            raise ValueError(f"Trajectory calibration image_size {self.image_size} disagrees with {image_size}")
        timestamps = _target_indices(self.frame_indices, latent_shape[0])
        poses = interpolate_camera_poses(self.frame_indices, self.camera_to_world, timestamps)
        calibration = interpolate_intrinsics(self.frame_indices, self.intrinsics, timestamps)

        # Match the trained transform. Express motion relative to the first camera,
        # clamp the translation radius in metres and apply the learned motion scale.
        relative = se3_inverse(poses[0:1]) @ poses
        relative[0] = torch.eye(4, dtype=relative.dtype)
        translations = relative[:, :3, 3]
        norms = torch.norm(translations, dim=-1, keepdim=True)
        scale = float(norms.max())
        translations[:] = translations * torch.clamp(150.0 / norms.clamp_min(1e-12), max=1.0) / 2.0

        focal_x, focal_y, center_x, center_y = calibration.unbind(dim=-1)
        intrinsics = torch.zeros((latent_shape[0], 3, 3), dtype=torch.float32)
        intrinsics[:, 0, 0] = focal_x
        intrinsics[:, 1, 1] = focal_y
        intrinsics[:, 0, 2] = center_x
        intrinsics[:, 1, 2] = center_y
        intrinsics[:, 2, 2] = 1
        return {
            "viewmats": [se3_inverse(relative)],
            "Ks": [intrinsics],
            "latent_shapes": [latent_shape],
            "frame_wh": [image_size],
            "trans_scale": [scale],
        }
