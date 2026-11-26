import unittest

import torch
from torch import nn

from ucpe.runtime.model.dit import TransformerDecoderBlock


class Zero(nn.Module):
    def forward(self, x, *args):
        return torch.zeros_like(x)


class CameraResidual(nn.Module):
    def forward(self, x, *, viewmats_per_sample, **kwargs):
        return torch.ones_like(x) * viewmats_per_sample[0][0, 0, 3]


class CameraBranchTest(unittest.TestCase):
    def test_trajectory_reaches_denoiser_and_respects_timestep_gate(self):
        block = TransformerDecoderBlock(8, 4, 16, 4, use_ucpe=False)
        block.self_attention = Zero()
        block.cross_attention = Zero()
        block.feed_forward = Zero()
        block.ucpe_branch = CameraResidual()
        ends = torch.tensor([0, 1], dtype=torch.int32)
        position = torch.zeros(1, dtype=torch.long)
        args = (torch.zeros(1, 8), torch.zeros(1, 8), torch.zeros(1, 4),
                None, ends, ends, position, None)

        def run(translation, gate):
            pose = torch.eye(4)[None]
            pose[0, 0, 3] = translation
            return block(*args, ucpe_inputs={
                "viewmats": [pose],
                "Ks": [torch.eye(3)[None]],
                "latent_shapes": [(1, 1, 1)],
                "frame_wh": [(1, 1)],
                "navio": {"permutation": position, "inverse": position},
                "t_gate": torch.full((1, 1), gate),
            })

        torch.testing.assert_close(run(0.0, 1.0), torch.zeros(1, 8))
        torch.testing.assert_close(run(2.0, 1.0), torch.full((1, 8), 2.0))
        torch.testing.assert_close(run(2.0, 0.0), torch.zeros(1, 8))


if __name__ == "__main__":
    unittest.main()
