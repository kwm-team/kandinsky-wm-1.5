"""Regenerate the metric OpenCV camera paths used in the AV examples."""

import json
import math
from pathlib import Path

FPS = 24
SIZE = (768, 512)
INTRINSICS = [[580, 0, 384], [0, 580, 256], [0, 0, 1]]


def camera_pose(distance: float, radius: float | None) -> list[list[float]]:
    angle = distance / radius if radius else 0.0
    sin, cos = math.sin(angle), math.cos(angle)
    x = -radius * (1 - cos) if radius else 0.0
    z = radius * sin if radius else distance
    return [[cos, 0, -sin, x], [0, 1, 0, 0], [sin, 0, cos, z], [0, 0, 0, 1]]


def travel(frame: int) -> float:
    seconds = max(0, frame / FPS - 0.5)
    acceleration = 1.5
    cruise_speed = 4.0
    ramp_end = cruise_speed / acceleration
    ramp_time = min(seconds, ramp_end)
    return 0.5 * acceleration * ramp_time**2 + cruise_speed * max(0, seconds - ramp_end)


def write_path(path: Path, frames: int, radius: float | None = None, *, stationary: bool = False) -> None:
    indices = list(range(frames))
    header = {
        "fps": FPS, "image_size": SIZE, "frame_indices": indices, "intrinsics": INTRINSICS,
    }
    poses = [
        [[round(value, 8) for value in row] for row in camera_pose(0 if stationary else travel(index), radius)]
        for index in indices
    ]
    text = json.dumps(header, indent=2).removesuffix("\n}")
    text += ',\n  "camera_to_world": [\n'
    text += ",\n".join("    " + json.dumps(pose, separators=(", ", ": ")) for pose in poses)
    path.write_text(text + "\n  ]\n}\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    write_path(root / "straight.json", 121)
    write_path(root / "left.json", 121, radius=16)
    write_path(root / "right.json", 121, radius=-16)
    write_path(root / "stationary.json", 121, stationary=True)
    write_path(root / "rollout.json", 145)
