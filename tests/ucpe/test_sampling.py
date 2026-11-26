import unittest

import torch

from ucpe.sampling import sample_video


class Denoiser:
    patch_size = (1, 2, 2)

    def __init__(self):
        self.inputs = []

    def __call__(self, visual, tokens, pooled, time, visual_ends, text_ends, visual_positions, text_positions,
                 *, ucpe_inputs, scale_factor, navio_inputs):
        self.inputs.append((visual.detach().clone(), ucpe_inputs, visual_positions, navio_inputs))
        return torch.ones_like(visual[..., :1])


class SamplerTest(unittest.TestCase):
    def test_camera_geometry_is_duplicated_for_clean_tail(self):
        denoiser = Denoiser()
        noise = torch.zeros(3, 2, 2, 1)
        reference = torch.full((1, 2, 2, 1), 9.0)
        text = (torch.empty(1, 4), torch.empty(1, 2))
        view = torch.eye(4)[None].repeat(3, 1, 1)
        view[2, 0, 3] = 5
        cameras = {"viewmats": [view], "Ks": [torch.eye(3)[None].repeat(3, 1, 1)]}
        result = sample_video(
            denoiser, noise, reference, text, None, cameras,
            num_steps=3, scheduler_scale=1, guidance_weight=1, scale_factor=(1, 2, 2),
        )
        self.assertEqual(len(denoiser.inputs), 2)
        for visual, geometry, positions, navio in denoiser.inputs:
            self.assertTrue(torch.equal(visual[-1:, ..., :1], reference))
            self.assertTrue(torch.equal(geometry["viewmats"][0][-1], view[0]))
            self.assertEqual(positions[0].tolist(), [0, 1, 2, 0])
            self.assertEqual(navio["type_ids"].tolist(), [0, 0, 0, 1])
        torch.testing.assert_close(result, torch.full_like(noise, -1.0))

    def test_video_prefix_retains_time_and_is_excluded_from_euler_updates(self):
        denoiser = Denoiser()
        noise = torch.zeros(4, 2, 2, 1)
        reference = torch.full((2, 2, 2, 1), 9.0)
        text = (torch.empty(1, 4), torch.empty(1, 2))
        cameras = {"viewmats": [torch.eye(4)[None].repeat(4, 1, 1)]}
        result = sample_video(
            denoiser, noise, reference, text, None, cameras,
            num_steps=3, scheduler_scale=1, guidance_weight=1, scale_factor=(1, 2, 2),
        )
        for visual, geometry, positions, navio in denoiser.inputs:
            self.assertIs(geometry, cameras)
            torch.testing.assert_close(visual[-2:, ..., :1], reference)
            self.assertEqual(positions[0].tolist(), [2, 3, 0, 1])
            self.assertEqual(navio["inverse"].tolist(), [2, 3, 0, 1])
            self.assertEqual(navio["type_ids"].tolist(), [0, 0, 1, 1])
        torch.testing.assert_close(result[:2], reference)
        torch.testing.assert_close(result[2:], torch.full_like(noise[2:], -1.0))

    def test_text_to_video_uses_only_generation_slots(self):
        denoiser = Denoiser()
        noise = torch.zeros(3, 2, 2, 1)
        text = (torch.empty(1, 4), torch.empty(1, 2))
        cameras = {}
        result = sample_video(
            denoiser, noise, None, text, None, cameras,
            num_steps=3, scheduler_scale=1, guidance_weight=1, scale_factor=(1, 2, 2),
        )
        for visual, geometry, positions, navio in denoiser.inputs:
            self.assertIs(geometry, cameras)
            self.assertEqual(len(visual), 3)
            self.assertEqual(positions[0].tolist(), [0, 1, 2])
            self.assertTrue(navio["skip_type_embeddings"])
            self.assertEqual(navio["type_ids"].tolist(), [0, 0, 0])
        torch.testing.assert_close(result, torch.full_like(noise, -1.0))


if __name__ == "__main__":
    unittest.main()
