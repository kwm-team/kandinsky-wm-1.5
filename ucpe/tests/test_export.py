from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from safetensors import safe_open
import torch

from ucpe.export import export, merge


class ExportTest(unittest.TestCase):
    def test_reassembles_dense_and_applies_ema(self):
        with TemporaryDirectory() as root:
            root = Path(root)
            dense, ema = root / "base", root / "adapter"
            dense.mkdir()
            ema.mkdir()
            name = "visual_transformer_blocks.12.ucpe_branch.q_proj.weight"
            for rank in range(2):
                torch.save({"weight": torch.tensor([rank]), name: torch.tensor([0.0])}, dense / f"{rank}.pt")
                torch.save({name: torch.tensor([2.0 + rank])}, ema / f"{rank}.pt")
            target = root / "model.safetensors"
            export(dense, ema, target)
            with safe_open(target, framework="pt", device="cpu") as result:
                self.assertEqual(result.get_tensor("weight").tolist(), [0, 1])
                self.assertEqual(result.get_tensor(name).tolist(), [2.0, 3.0])
                self.assertEqual(set(result.keys()), set(merge(dense, ema)))


if __name__ == "__main__":
    unittest.main()
