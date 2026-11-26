import contextlib
import io
import json
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from base_av.cli import build_jobs, parse_args
from base_av.temporal import TemporalPrompt


class TemporalPromptTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.image = self.root / "start.jpg"
        self.image.touch()
        self.prompt = {
            "scene": "A drive through town.",
            "captions": [f"Moment {i}." for i in range(5)],
        }
        self.segments = self.root / "segments.json"
        self.segments.write_text(json.dumps(self.prompt))

    def args(self, *args):
        return parse_args(
            ["--mode", "i2v", "--model-dir", str(self.root), *map(str, args)]
        )

    def test_standalone_segments_default_to_hard(self):
        args = self.args(
            "--image",
            self.image,
            "--prompt-segments",
            self.segments,
            "--output",
            self.root / "result.mp4",
        )
        self.assertEqual(args.temporal_text, "hard")
        self.assertEqual(
            build_jobs(args)[0].prompt,
            TemporalPrompt(self.prompt["scene"], tuple(self.prompt["captions"])),
        )

    def test_batch_supports_original_manifest_schema(self):
        manifest = self.root / "samples.json"
        manifest.write_text(
            json.dumps(
                [{**self.prompt, "img_id": self.image.name, "video_id": "sample"}]
            )
        )
        args = self.args(
            "--input", manifest, "--temporal-text", "smooth", "--output-dir", self.root
        )
        self.assertEqual(
            build_jobs(args)[0].prompt.captions, tuple(self.prompt["captions"])
        )

    def test_four_captions_rejected_before_inference(self):
        self.segments.write_text(
            json.dumps({**self.prompt, "captions": self.prompt["captions"][:4]})
        )
        args = self.args(
            "--image",
            self.image,
            "--prompt-segments",
            self.segments,
            "--output",
            self.root / "result.mp4",
        )
        with self.assertRaisesRegex(ValueError, "exactly five"):
            build_jobs(args)

    def test_temporal_rejects_short_output(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args(
                "--image",
                self.image,
                "--prompt-segments",
                self.segments,
                "--output",
                self.root / "result.mp4",
                "--output-frames",
                24,
            )

    @unittest.skipUnless(find_spec("torch"), "PyTorch is required")
    def test_routes_keep_startup_and_reach_final_second(self):
        import torch
        from base_av.model.temporal import frame_priors

        hard = frame_priors("hard")
        self.assertEqual(
            hard.argmax(1).tolist(), [0, *[i for i in range(1, 6) for _ in range(6)], 0]
        )
        smooth = frame_priors("smooth")
        torch.testing.assert_close(smooth.sum(1), torch.ones(32))
        self.assertEqual(smooth[4].nonzero().flatten().tolist(), [1, 2])
        self.assertEqual(smooth[31].argmax().item(), 0)
