import hashlib

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from f2lf.camera import camera_path
from f2lf.kernels import reference_matmul_precision
from f2lf.references import load_reference_latents
from f2lf.sampling import pack_latents, sample


def test_camera_arc_moves_sideways_and_preserves_both_anchors():
    first = np.eye(4)
    last = first.copy()
    last[0, 3] = 60
    request = {
        "coordinate_system": "metric_ENU",
        "resolution": [768, 512],
        "c2w_A": first.tolist(),
        "c2w_B": last.tolist(),
        "K": [[548, 0, 383.5], [0, 548, 255.5], [0, 0, 1]],
    }
    path = camera_path(request, offset_m=3.5)
    np.testing.assert_array_equal(path.poses[[0, -1]], [first, last])
    np.testing.assert_allclose(path.poses[15, :3, 3], [30, -3.5, 0])
    np.testing.assert_array_equal(path.viewmats[-2:], path.viewmats[[0, 30]])
    assert path.intrinsics.shape == (33, 3, 3)
    assert path.translation_scale == 60.0
    with pytest.raises(ValueError, match="offset"):
        camera_path(request, offset_m=5.1)


def test_pack_latents_keeps_independent_reference_slots():
    body = torch.zeros(3, 2, 2, 16)
    anchors = torch.stack((torch.ones_like(body[0]), torch.full_like(body[0], 2)))
    packed = pack_latents(body, anchors)
    assert packed.shape == (5, 2, 2, 33)
    torch.testing.assert_close(packed[-2:, ..., :16], anchors)
    assert not packed[:3].count_nonzero()
    assert (packed[-2:, ..., -1] == 1).all()


def test_rf_solver_runs_fifty_cfg_pairs():
    calls = []

    def velocity(current, sigma, *, positive):
        calls.append((float(sigma), positive))
        return torch.ones_like(current) if positive else torch.zeros_like(current)

    result = sample(velocity, torch.zeros(3, 2, 2, 16))
    assert len(calls) == 100
    assert calls[0] == (1.0, True)
    assert all(calls[i][0] == calls[i + 1][0] and calls[i][1] and not calls[i + 1][1] for i in range(0, 100, 2))
    torch.testing.assert_close(result, torch.full_like(result, -5))


def test_reference_precision_restores_callers_matmul_policy_on_error():
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("medium")
        with pytest.raises(RuntimeError, match="sample interrupted"), reference_matmul_precision():
            assert torch.get_float32_matmul_precision() == "highest"
            raise RuntimeError("sample interrupted")
        assert torch.get_float32_matmul_precision() == "medium"
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.fixture
def reference_bundle(tmp_path):
    first, last, latents = (tmp_path / name for name in ("first.png", "last.png", "reference.safetensors"))
    first.write_bytes(b"first")
    last.write_bytes(b"last")
    metadata = {
        "format": "kwm-av-f2lf-reference-latents-v1",
        "first_sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
        "last_sha256": hashlib.sha256(last.read_bytes()).hexdigest(),
    }
    return first, last, latents, metadata


def test_cached_f2lf_anchors_bind_to_both_reference_images(reference_bundle):
    first, last, path, metadata = reference_bundle
    anchors = np.zeros((2, 64, 96, 16), dtype=np.float32)
    save_file({"anchors": anchors}, str(path), metadata)

    np.testing.assert_array_equal(load_reference_latents(path, first, last), anchors)
    with pytest.raises(ValueError, match="reference images"):
        load_reference_latents(path, last, first)


@pytest.mark.parametrize("shape,dtype,fill", [
    ((1, 64, 96, 16), np.float32, 0),
    ((2, 64, 96, 16), np.float16, 0),
    ((2, 64, 96, 16), np.float32, np.nan),
])
def test_cached_f2lf_anchors_validate_tensor(reference_bundle, shape, dtype, fill):
    first, last, path, metadata = reference_bundle
    save_file({"anchors": np.full(shape, fill, dtype=dtype)}, str(path), metadata)

    with pytest.raises(ValueError, match="finite float32"):
        load_reference_latents(path, first, last)
