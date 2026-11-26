"""Video encoding for RGB inference outputs."""

from itertools import islice
from os import link
from pathlib import Path
from tempfile import NamedTemporaryFile

import av
import numpy as np


def read_video_prefix(path: Path, frames: int, fps: int) -> np.ndarray:
    """Read the first RGB frames at the checkpoint's video rate."""
    with av.open(path, "r") as container:
        if not container.streams.video:
            raise ValueError("The reference file has no video stream")
        stream = container.streams.video[0]
        if stream.average_rate != fps:
            raise ValueError(f"The reference video must be {fps} fps")
        images = [frame.to_ndarray(format="rgb24") for frame in islice(container.decode(stream), frames)]
    if len(images) < frames:
        raise ValueError(f"The reference video must contain at least {frames} frames")
    return np.stack(images)


def write_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    """Encode THWC RGB video and publish the completed file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".mp4", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with av.open(temporary_path, "w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.height, stream.width = frames.shape[1:3]
            stream.pix_fmt = "yuv420p"
            for image in frames:
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
