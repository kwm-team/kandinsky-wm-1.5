"""Image-to-video and video-prefix continuation with the Base AV checkpoint."""
import math
from pathlib import Path

import torch

from .model.loading import load_models
from .reference import encode_reference
from .sampler import sample_video
from .temporal import TEMPORAL_OUTPUT_FRAMES, TemporalPrompt


class BaseAVPipeline:
    def __init__(self, bundle: Path, device, mesh=None):
        self.device = torch.device(device)
        self.dit, self.vae, self.text = load_models(bundle, self.device, mesh)

    @torch.inference_mode()
    def encode_prompt(self, prompt: str | TemporalPrompt):
        if isinstance(prompt, TemporalPrompt):
            from .model.temporal import encode_temporal_text

            return encode_temporal_text(self.text, prompt.scene, prompt.captions)
        return self.text.encode(prompt)

    @torch.inference_mode()
    def encode_negative(self, prompt: str, *, temporal: bool = False):
        if temporal:
            from .model.temporal import encode_temporal_text

            return encode_temporal_text(self.text, negative_caption=prompt)
        return self.text.encode(prompt)

    @torch.inference_mode()
    def generate(
        self,
        source: Path,
        prompt: str | TemporalPrompt,
        negative,
        *,
        mode: str,
        prefix_frames: int | None,
        visual_size: int,
        output_frames: int,
        seed: int,
        num_steps: int,
        scheduler_scale: float,
        guidance_weight: float,
        temporal_policy: str | None = None,
    ) -> torch.Tensor:
        """Return a uint8 RGB tensor in CTHW order."""
        if isinstance(prompt, TemporalPrompt) != (temporal_policy is not None):
            raise ValueError("Provide a temporal prompt and policy together")
        if temporal_policy is not None and (mode != "i2v" or output_frames not in TEMPORAL_OUTPUT_FRAMES):
            raise ValueError("Temporal text requires I2V with 120 or 121 output frames")
        latent_frames = math.ceil((output_frames - 1) / 4) + 1
        generator = torch.Generator(device=self.device).manual_seed(seed)
        reference = encode_reference(
            source, mode, prefix_frames, visual_size, self.vae, self.device, generator
        )
        video = sample_video(
            self.dit, self.vae, self.encode_prompt(prompt), negative, reference, mode,
            latent_frames, seed, num_steps, scheduler_scale, guidance_weight, temporal_policy,
        )
        return video[:, :output_frames]
