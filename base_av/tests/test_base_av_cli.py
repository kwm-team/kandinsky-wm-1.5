import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from base_av.cli import build_jobs, parse_args
from base_av.inputs import Job


class InputTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def args(self, *args):
        return parse_args(["--model-dir", str(self.root / "weights"), *map(str, args)])

    def test_single_image_keeps_seed_and_prompt(self):
        image = self.root / "dashcam.png"
        image.touch()
        output = self.root / "road.mp4"
        args = self.args("--mode", "i2v", "--image", image, "--prompt", "Traffic continues.",
                         "--seed", 42, "--output", output)
        self.assertEqual(build_jobs(args), [Job(image, "Traffic continues.", 42, output)])

    def test_manifest_inherits_seed_from_filename(self):
        images = self.root / "condition_image"
        images.mkdir()
        image = images / "road.jpg"
        image.touch()
        manifest = self.root / "samples.csv"
        manifest.write_text("filename,prompt\nroad__42.mp4,Traffic continues.\n")
        output = self.root / "results"
        args = self.args("--mode", "i2v", "--input", manifest, "--output-dir", output)
        self.assertEqual(build_jobs(args), [
            Job(image, "Traffic continues.", 42, output / "road__42.mp4")
        ])

    def test_video_prefix_length_must_match_temporal_grid(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args("--mode", "v2v", "--video", self.root / "left.mp4", "--prompt", "Continue.",
                      "--output", self.root / "future.mp4", "--prefix-frames", 6)

    def test_manifest_protects_other_inputs_from_overwrite(self):
        original = self.root / "scene__1137.mp4"
        original.touch()
        other = self.root / "other.mp4"
        other.touch()
        manifest = self.root / "samples.json"
        manifest.write_text(json.dumps([
            {"video_id": "scene", "prompt": "Continue.", "video_path": other.name},
            {"video_id": "another", "prompt": "Continue.", "video_path": original.name},
        ]))
        args = self.args("--mode", "v2v", "--input", manifest, "--output-dir", self.root)
        with self.assertRaisesRegex(ValueError, "overlaps a conditioning"):
            build_jobs(args)


if __name__ == "__main__":
    unittest.main()
