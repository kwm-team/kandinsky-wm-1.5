import hashlib
from fractions import Fraction
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

TOOL = run_path(str(Path(__file__).resolve().parents[1] / "tools" / "upscale_video.py"))


@pytest.mark.parametrize("frame_count", [120, 121])
def test_upscale_preserves_frames_and_presentation_times(tmp_path, frame_count):
    source = tmp_path / "native.mp4"
    output = tmp_path / "upscaled.mp4"
    native_frames = [Image.fromarray(np.full((64, 64, 3), index, np.uint8)) for index in range(frame_count)]
    TOOL["write_video"](source, native_frames, Fraction(24))

    class Pipeline:
        vae_scale_factor_temporal = 4
        vae_scale_factor_spatial = 16

        def __call__(self, *, video, generator, resolution_scale, num_inference_steps):
            assert len(video) == 121
            assert np.array_equal(np.asarray(video[frame_count - 1]), np.asarray(video[-1]))
            assert resolution_scale == 2.25
            assert num_inference_steps == 2
            assert generator is None
            return SimpleNamespace(frames=[[frame.resize((128, 128)) for frame in video]])

    TOOL["upscale_video"](Pipeline(), source, output, generator=None)
    _, rate, times = TOOL["read_video"](output)
    assert rate == 24
    assert times == tuple(Fraction(index, 24) for index in range(frame_count))
    assert output.with_suffix(".mp4.source.sha256").read_text().splitlines() == [
        hashlib.sha256(source.read_bytes()).hexdigest(), hashlib.sha256(output.read_bytes()).hexdigest(),
    ]
