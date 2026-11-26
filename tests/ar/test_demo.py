import io

import numpy as np
import pytest
import torch

import ar.parallel as parallel
from ar.demo import parse_args
from ar.demo_control import Drive, Layout, read_rgb, resize_intrinsics, write_rgb
from ar.model import SolverSettings
from ar.parallel import local_caption_routing
from ar.runtime import active_camera_levels
from ar.solver import (
    PiFlowPolicy,
    apply_rollout_coefficients,
    build_solver_segments,
    precompute_rollout_coefficients,
    rollout_policy,
)
from ar.stream import pack


def test_caption_routing_for_a_partial_latent_plane():
    source = {"rf_text_run_frames": torch.tensor([4]), "carla_stream": True}
    # The 6144 visual tokens split over seven ranks give 878 queries on rank 0.
    # That local span can attend to the same caption without dividing by four.
    assert local_caption_routing(source, torch.tensor([0, 35])) == {"carla_stream": True}
    assert "rf_text_run_frames" in source
    with pytest.raises(ValueError, match="one caption"):
        local_caption_routing(source, torch.tensor([0, 35, 70]))


def test_visual_stack_gathers_hidden_state_only_before_output(monkeypatch):
    shapes, gathered = [], []

    class Block(torch.nn.Module):
        def forward(self, x, *_):
            shapes.append(len(x))
            return x + 1

    def native_init(self, *args, **kwargs):
        self.blocks = [Block(), Block(), Block()]
        self.originals = []

    class Shard:
        def __init__(self, _):
            pass

        def bounds(self, total):
            assert total == 7
            return 0, 3, 3

        def gather(self, value, total):
            gathered.append(value.clone())
            return torch.cat((value, value.new_zeros(total - len(value))))

    monkeypatch.setattr(parallel.NativeStream, "__init__", native_init)
    monkeypatch.setattr(parallel, "TokenGroup", Shard)
    stream = parallel.ParallelNativeStream(group=None)
    value = torch.arange(7)
    with torch.inference_mode():
        for block in stream.blocks:
            value = block(value, None, None, None, torch.tensor([0, 7]),
                          torch.tensor([0, 2]), torch.zeros(7), {}, None)

    assert shapes == [3, 3, 3]
    assert len(gathered) == 1
    assert value.tolist() == [3, 4, 5, 0, 0, 0, 0]


def test_roles_allow_one_gpu_and_split_across_three_nodes():
    solo = Layout.resolve(1)
    assert (solo.dit, solo.vae, solo.sr) == ((0,), 0, 0)
    shared = Layout.resolve(8)
    assert (shared.dit, shared.vae, shared.sr) == (tuple(range(7)), 7, 7)
    split = Layout.resolve(24, dit_ranks=8, vae_rank=8, sr_rank=16)
    assert (split.dit, split.vae, split.sr) == (tuple(range(8)), 8, 16)
    with pytest.raises(ValueError, match="DiT"):
        Layout.resolve(8, dit_ranks=7, vae_rank=6)
    with pytest.raises(ValueError, match="rank"):
        Layout.resolve(8, sr_rank=8)


def test_drive_stationary_and_right_turn_preserve_opencv_camera_pose():
    drive = Drive(np.eye(4))
    stationary = drive.next_window(set())
    np.testing.assert_array_equal(stationary.poses, np.repeat(np.eye(4)[None], 4, axis=0))
    turn = drive.next_window({"forward", "right"})
    assert turn.poses[-1, 0, 2] > 0
    assert turn.poses[-1, 2, 3] > 0
    for pose in turn.poses:
        np.testing.assert_allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-9)
        assert np.linalg.det(pose[:3, :3]) == pytest.approx(1)


def test_window_at_reset_commits_only_accepted_motion():
    drive = Drive(np.eye(4), reset_frames=24)
    drive.next_window({"forward"})
    reset = drive.next_window({"forward"})
    assert (reset.accepted_planes, reset.reset_after) == (2, True)
    np.testing.assert_allclose(drive.position, reset.poses[1, :3, 3])
    assert reset.poses[3, 2, 3] > drive.position[2]
    next_window = drive.next_window({"forward"}, frame_limit=28)
    assert (next_window.accepted_planes, next_window.reset_after) == (1, False)


def test_demo_can_align_history_resets_to_full_generation_chunks():
    args = parse_args(["--weights", "ar", "--image", "frame.jpg", "--prompt", "Drive",
                       "--calibration", "drive.json", "--reset-frames", "128"])
    drive = Drive(np.eye(4), reset_frames=args.reset_frames)
    windows = [drive.next_window({"forward"}) for _ in range(8)]
    assert all(window.accepted_planes == 4 for window in windows)
    assert not any(window.reset_after for window in windows[:-1])
    assert windows[-1].reset_after


def test_sr_framing_roundtrips_rgb_and_detects_truncated_data():
    source = np.arange(120, dtype=np.uint8).reshape(2, 4, 5, 3)
    stream = io.BytesIO()
    write_rgb(stream, source)
    payload = stream.getvalue()
    np.testing.assert_array_equal(read_rgb(io.BytesIO(payload)), source)
    assert read_rgb(io.BytesIO()) is None
    with pytest.raises(EOFError, match="RGB"):
        read_rgb(io.BytesIO(payload[:-1]))


def test_intrinsics_follow_pixel_centers_when_downsampling():
    calibration = np.array([[[768, 0, 383.5], [0, 512, 255.5], [0, 0, 1]]], dtype=np.float32)
    resized = resize_intrinsics(calibration, (768, 512), (576, 384))
    np.testing.assert_array_equal(resized[0], [[576, 0, 287.5], [0, 384, 191.5], [0, 0, 1]])


def test_linear_rollout_matches_all_eight_reference_solver_segments():
    settings = SolverSettings(8, 8, 4, 8, 10.0, 0.5, 128, 1e-5)
    schedule = build_solver_segments(8, 0.5, 10.0, 1e-5, "cpu")
    coefficients = precompute_rollout_coefficients(schedule, settings)
    rng = torch.Generator().manual_seed(42)
    latent = torch.randn((4, 128), generator=rng)
    grid = torch.randn((4, settings.grid_points, 128), generator=rng)
    for weights, (src, dst, sigma, _, step_size) in zip(coefficients, zip(*schedule)):
        policy = PiFlowPolicy(grid, latent, sigma.expand(4), step_size.expand(4), 10, 1e-5)
        reference, _ = rollout_policy(latent, sigma.expand(4, 1), src.expand(4), dst.expand(4), 128, policy)
        fast = apply_rollout_coefficients(latent, grid, weights)
        torch.testing.assert_close(fast, reference, rtol=3e-6, atol=4e-6)


def test_camera_schedule_retains_boundary_levels_and_skips_uniform_zero_gate():
    sigmas = torch.tensor([1.0, 0.8, 0.6, 0.59, 0.35])
    assert active_camera_levels(sigmas, (0.0, 0.4)) == (True, True, True, False, False)
    assert all(active_camera_levels(sigmas, None))


def test_single_timestep_packing_preserves_inputs_and_labels_optimized_layout():
    captured = []

    def model(*args, **kwargs):
        captured.append((args, kwargs["sparse_params"]))
        return args[0]

    frames = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2, 1)
    inputs = (
        model, frames, torch.tensor([1, 0]), torch.arange(2), torch.tensor([0.8, 0.0]),
        torch.zeros(2, dtype=torch.int32),
        {"text": torch.zeros(2, 1), "pooled": torch.zeros(1), "cu": torch.tensor([0, 2]), "rope": torch.arange(2)},
        {},
    )
    assert torch.equal(pack(*inputs), pack(*inputs, single_timestep=True))
    assert captured[1][1]["carla_single_timestep"]
    assert "rf_text_run_frames" not in captured[1][1]
    assert torch.equal(captured[0][1]["rf_time_idx"]["text"], captured[1][1]["rf_time_idx"]["text"])
