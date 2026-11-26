"""Keyboard trajectories, role placement and RGB framing for the AR demo."""

from dataclasses import dataclass
import math
import struct

import numpy as np


@dataclass(frozen=True)
class Layout:
    dit: tuple[int, ...]
    vae: int
    sr: int

    @classmethod
    def resolve(cls, world, *, dit_ranks=None, vae_rank=None, sr_rank=None):
        vae = world - 1 if vae_rank is None else vae_rank
        sr = vae if sr_rank is None else sr_rank
        count = (1 if world == 1 else vae) if dit_ranks is None else dit_ranks
        if not (0 <= vae < world and 0 <= sr < world):
            raise ValueError("VAE and SR rank must be inside the torchrun world")
        if not 1 <= count <= world or (world > 1 and min(vae, sr) < count):
            raise ValueError("DiT ranks must precede the dedicated VAE and SR ranks")
        return cls(tuple(range(count)), vae, sr)


def local_device_index(layout, rank, local_rank, vae_device):
    if rank == layout.vae and vae_device is not None:
        return vae_device
    return local_rank


@dataclass(frozen=True)
class DriveWindow:
    poses: np.ndarray
    accepted_planes: int
    reset_after: bool


class Drive:
    """Advance in the startup camera's horizontal plane at the video frame rate."""

    def __init__(self, startup_pose, *, reset_frames=120, chunk_planes=4):
        self.pose = np.array(startup_pose, dtype=np.float64, copy=True)
        self.basis = self.pose[:3, :3].copy()
        self.position = self.pose[:3, 3].copy()
        self.heading = self.speed = 0.0
        self.frame = 0
        self.reset_frames = reset_frames
        self.chunk_planes = chunk_planes

    def next_window(self, keys, frame_limit=None):
        visible = min(
            self.chunk_planes * 4,
            self.reset_frames - self.frame % self.reset_frames,
            frame_limit - self.frame if frame_limit is not None else self.chunk_planes * 4,
        )
        if visible < 4 or visible % 4:
            raise ValueError("The next AR window must contain whole latent planes")

        speed, heading, position = self.speed, self.heading, self.position.copy()
        steering = int("right" in keys) - int("left" in keys)
        acceleration = -7.0 if "brake" in keys else 3.2 if "forward" in keys else -1.8
        poses = []
        for index in range(1, self.chunk_planes * 4 + 1):
            speed = max(0.0, min(10.0, speed + acceleration / 24))
            heading += steering * 0.65 * min(1.0, speed / 4) / 24
            c, s = math.cos(heading), math.sin(heading)
            rotation = self.basis @ ((c, 0, s), (0, 1, 0), (-s, 0, c))
            position += rotation[:, 2] * (speed / 24)
            if index == visible:
                self.speed, self.heading, self.position = speed, heading, position.copy()
            if index % 4 == 0:
                pose = self.pose.copy()
                pose[:3, :3], pose[:3, 3] = rotation, position
                poses.append(pose)
        self.frame += visible
        return DriveWindow(
            np.stack(poses), visible // 4, self.frame % self.reset_frames == 0
        )


def resize_intrinsics(intrinsics, source_size, target_size):
    sx, sy = (target / source for source, target in zip(source_size, target_size))
    pixel_resize = np.array(
        ((sx, 0, (sx - 1) / 2), (0, sy, (sy - 1) / 2), (0, 0, 1)),
        dtype=intrinsics.dtype,
    )
    return pixel_resize @ intrinsics


def center_crop(image, intrinsics, target_size):
    width, height = target_size
    if width > image.width or height > image.height:
        raise ValueError("Crop dimensions cannot exceed the source image")
    left, top = (image.width - width) // 2, (image.height - height) // 2
    pixel_crop = np.array(((1, 0, -left), (0, 1, -top), (0, 0, 1)), dtype=intrinsics.dtype)
    return image.crop((left, top, left + width, top + height)), pixel_crop @ intrinsics


_RGB_HEADER = struct.Struct("!III")


def _read_exact(stream, size, *, allow_eof=False):
    data = bytearray(size)
    view = memoryview(data)
    received = 0
    while received < size:
        count = stream.readinto(view[received:])
        if not count:
            if allow_eof and received == 0:
                return None
            raise EOFError("Incomplete RGB message")
        received += count
    return data


def read_rgb(stream):
    header = _read_exact(stream, _RGB_HEADER.size, allow_eof=True)
    if header is None:
        return None
    frames, height, width = _RGB_HEADER.unpack(header)
    if not (0 < frames <= 64 and 0 < height <= 4320 and 0 < width <= 7680):
        raise ValueError("Invalid RGB message dimensions")
    data = _read_exact(stream, frames * height * width * 3)
    return np.frombuffer(data, dtype=np.uint8).reshape(frames, height, width, 3)


def write_rgb(stream, frames):
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected uint8 RGB frames with shape [T,H,W,3]")
    stream.write(_RGB_HEADER.pack(*frames.shape[:3]))
    stream.write(memoryview(np.ascontiguousarray(frames)).cast("B"))
    stream.flush()
