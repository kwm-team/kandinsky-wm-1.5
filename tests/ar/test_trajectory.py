import json

import numpy as np
import pytest

from ar.trajectory import load_trajectory, rollout_windows


def fixture(frames=136):
    poses = np.repeat(np.eye(4)[None], frames + 1, axis=0)
    poses[:, 2, 3] = np.arange(frames + 1) / 24
    return {
        "fps": 24,
        "frame_indices": list(range(frames + 1)),
        "camera_to_world": poses.tolist(),
        "intrinsics": [[500, 0, 384], [0, 500, 256], [0, 0, 1]],
        "image_size": [768, 512],
    }


def test_camera_contract_preserves_coordinates_and_expands_calibration(tmp_path):
    path = tmp_path / "route.json"
    path.write_text(json.dumps(fixture(16)))
    trajectory = load_trajectory(path)
    assert trajectory.camera_to_world.shape == (17, 4, 4)
    assert trajectory.intrinsics.shape == (17, 3, 3)
    assert trajectory.image_size == (768, 512)
    assert trajectory.camera_to_world[4, 2, 3] == 4 / 24


def test_camera_contract_rejects_invalid_world_pose(tmp_path):
    raw = fixture(16)
    raw["camera_to_world"][4][2][2] = -1
    path = tmp_path / "route.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="rotation"):
        load_trajectory(path)


def test_periodic_reset_commits_only_the_emitted_tail():
    windows = list(rollout_windows(136))
    assert len(windows) == 9
    assert [
        (window.start, window.frames, window.reset_after) for window in windows[-2:]
    ] == [
        (112, 8, True),
        (120, 16, False),
    ]
    assert windows[-2].anchors == (116, 120, 124, 128)
    assert windows[-1].anchors == (124, 128, 132, 136)


def test_terminal_partial_window_pads_coordinates():
    last = list(rollout_windows(20))[-1]
    assert last.frames == 4
    assert last.anchors == (20, 20, 20, 20)
