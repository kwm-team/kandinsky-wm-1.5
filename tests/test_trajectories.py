import json
import math
from pathlib import Path
import unittest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class ExampleTrajectoryTests(unittest.TestCase):
    def test_camera_paths_are_dense_and_use_metric_rigid_transforms(self):
        for name, count in (("straight", 121), ("left", 121), ("rollout", 145)):
            with self.subTest(name=name):
                data = json.loads((EXAMPLES / f"{name}.json").read_text())
                self.assertEqual(data["fps"], 24)
                self.assertEqual(data["image_size"], [768, 512])
                self.assertEqual(data["frame_indices"], list(range(count)))
                intrinsics = data["intrinsics"]
                self.assertGreater(intrinsics[0][0], 0)
                self.assertGreater(intrinsics[1][1], 0)
                poses = data["camera_to_world"]
                self.assertEqual(len(poses), count)
                identity = [[int(x == y) for x in range(4)] for y in range(4)]
                self.assertEqual(poses[0], identity)
                for pose in poses:
                    self.assertEqual(pose[3], [0, 0, 0, 1])
                    for i in range(3):
                        for j in range(3):
                            dot = math.fsum(pose[k][i] * pose[k][j] for k in range(3))
                            self.assertAlmostEqual(dot, int(i == j), places=7)
                self.assertGreater(poses[-1][2][3], 0)
                self.assertLessEqual(poses[-1][0][3], 0)

    def test_straight_and_left_examples_have_distinct_endpoints(self):
        straight = json.loads((EXAMPLES / "straight.json").read_text())
        left = json.loads((EXAMPLES / "left.json").read_text())
        self.assertEqual(straight["camera_to_world"][0], left["camera_to_world"][0])
        self.assertEqual(straight["camera_to_world"][-1][0][3], 0)
        self.assertLess(left["camera_to_world"][-1][0][3], -1)


if __name__ == "__main__":
    unittest.main()
