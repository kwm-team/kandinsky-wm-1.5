from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from base_av.media import complete_video, save_video


@unittest.skipUnless(find_spec("av") and find_spec("torch"), "PyAV and PyTorch are required")
class MediaTests(unittest.TestCase):
    def test_write_and_decode_mp4(self):
        import torch

        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.mp4"
            video = torch.zeros((3, 4, 32, 32), dtype=torch.uint8)
            save_video(video, path, frames=3, fps=24)
            self.assertTrue(complete_video(path, 3, size=(32, 32), fps=24))
            self.assertFalse(complete_video(path, 4))


if __name__ == "__main__":
    unittest.main()
