import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from ucpe.runtime.camera_control import _se6_angular_velocity_from_rotations
from ucpe.trajectory import Trajectory


def inputs():
    transforms = np.repeat(np.eye(4)[None], 2, axis=0)
    transforms[1, 0, 3] = 8
    return {
        "camera_to_world": transforms.tolist(),
        "intrinsics": [[500, 0, 384], [0, 510, 256], [0, 0, 1]],
        "frame_indices": [0, 8],
        "fps": 24,
        "image_size": [768, 512],
    }


class TrajectoryTest(unittest.TestCase):
    def test_json_and_npz_have_same_coordinates(self):
        data = inputs()
        with TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            json_path, npz_path = scratch / "trajectory.json", scratch / "trajectory.npz"
            json_path.write_text(json.dumps(data))
            np.savez(npz_path, **data)
            trajectory = Trajectory.from_file(json_path)
            npz = Trajectory.from_file(npz_path)
        self.assertEqual(trajectory.frames, 9)
        first = trajectory.camera_inputs((3, 64, 96), (768, 512))
        second = npz.camera_inputs((3, 64, 96), (768, 512))
        for key in ("viewmats", "Ks"):
            self.assertTrue(torch.equal(first[key][0], second[key][0]))
        torch.testing.assert_close(first["viewmats"][0][:, 0, 3], torch.tensor([0., -2., -4.]))
        torch.testing.assert_close(first["Ks"][0][:, 1, 1], torch.full((3,), 510.))
        self.assertEqual(first["frame_wh"], [(768, 512)])
        self.assertEqual(first["trans_scale"], [8.0])
        with self.assertRaisesRegex(ValueError, "calibration image_size"):
            trajectory.camera_inputs((3, 64, 96), (384, 256))

    def test_rejects_reflection_and_frame_index_ambiguity(self):
        data = inputs()
        data["camera_to_world"][0][0][0] = -1
        with self.assertRaisesRegex(ValueError, "proper rotation"):
            Trajectory._parse(data)
        data = inputs()
        data["frame_indices"] = [1, 8]
        with self.assertRaisesRegex(ValueError, "start at zero"):
            Trajectory._parse(data)

    def test_camera_rotation_rate(self):
        quarter_turn = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rotations = torch.stack((torch.eye(3), quarter_turn))
        rate = _se6_angular_velocity_from_rotations(rotations, 0.5)
        torch.testing.assert_close(rate, torch.tensor([[0.0, 0.0, torch.pi]]).expand(2, -1))


if __name__ == "__main__":
    unittest.main()
