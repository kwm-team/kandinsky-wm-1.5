"""Independent startup encoding and frame-exact temporal decoder overlap."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from PIL.Image import Image


class VideoCodec:
    def __init__(
        self, vae_dir: Path, device: torch.device, *, turbo_weights: Path | None = None
    ):
        from .native.model.vae import AutoencoderKLHunyuanVideo

        self.device = device
        self.vae = (
            AutoencoderKLHunyuanVideo.from_pretrained(
                vae_dir,
                torch_dtype=torch.float16,
                local_files_only=True,
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.vae.use_framewise_decoding = True
        torch.nn.utils.convert_conv3d_weight_memory_format(
            self.vae, torch.channels_last_3d
        )
        self.scale = self.vae.config.scaling_factor
        self.decoder = self.vae
        if turbo_weights is not None:
            from .native.model.turbo_vae import AutoencoderKLTurboVAED

            decoder = AutoencoderKLTurboVAED()
            weights = torch.load(turbo_weights, map_location="cpu", weights_only=True)
            if not (
                torch.equal(weights.pop("latents_mean"), torch.zeros(16))
                and torch.equal(weights.pop("latents_std"), torch.ones(16))
            ):
                raise ValueError("Unexpected TurboVAE latent normalization")
            decoder.decoder.load_state_dict(
                {
                    name: value
                    for name, value in weights.items()
                    if not name.startswith("aligned_feature_projection_heads.")
                },
                strict=True,
            )
            self.decoder = (
                decoder.to(device=device, dtype=torch.float16)
                .eval()
                .requires_grad_(False)
            )

    @torch.inference_mode()
    def encode_startup(self, image: Image, *, opt_tiling: bool = True) -> torch.Tensor:
        rgb = torch.from_numpy(np.asarray(image.convert("RGB")).copy())
        pixels = rgb.to(self.device, dtype=torch.float16).permute(2, 0, 1)[
            None, :, None
        ]
        pixels = (pixels / 127.5 - 1).contiguous(memory_format=torch.channels_last_3d)
        posterior = self.vae.encode(pixels, opt_tiling=opt_tiling).latent_dist
        # Match the stand transport's FP16 encode and FP32 latent rescaling.
        encoded = (
            (posterior.mode() * self.scale / self.scale)[0].permute(1, 2, 3, 0).float()
        )
        return encoded * self.scale

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        inputs = latents.to(self.device, dtype=torch.float16).permute(3, 0, 1, 2)[None]
        inputs = (inputs / self.scale).contiguous(memory_format=torch.channels_last_3d)
        video = self.decoder.decode(inputs).sample[0]
        return (
            ((video.float().clamp(-1, 1) + 1) * 127.5)
            .round()
            .byte()
            .permute(1, 2, 3, 0)
            .cpu()
        )


class OverlapDecoder:
    def __init__(self, codec: VideoCodec, startup: torch.Tensor):
        self.codec = codec
        self.context = startup.detach().clone()

    def decode(self, generated: torch.Tensor, accepted_planes: int) -> torch.Tensor:
        if not 1 <= accepted_planes <= len(generated):
            raise ValueError("Visible length does not fit the generated chunk")
        prefix = len(self.context)
        latent = torch.cat((self.context.to(generated), generated))
        decoded = self.codec.decode(latent)
        if len(decoded) != 4 * len(latent) - 3:
            raise ValueError("Unexpected video VAE temporal compression")
        self.context = latent[: prefix + accepted_planes][-2:].detach().clone()
        start = 4 * prefix - 3
        return decoded[start : start + 4 * accepted_planes]
