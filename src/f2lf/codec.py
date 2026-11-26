"""Independently encode two images and decode only the generated body."""

from contextlib import nullcontext
import hashlib
from pathlib import Path
from types import MethodType

import cv2
import numpy as np
from PIL import Image
import torch

from ucpe.runtime.model.vae import AutoencoderKLHunyuanVideo

from .camera import HEIGHT, WIDTH

VAE_SCALE = 0.476986
REFERENCE_SEED = 20260914
IMAGE_SIZES = ((WIDTH, HEIGHT), (2 * WIDTH, 2 * HEIGHT))


def use_eager_forwards(model):
    for module in model.modules():
        original = getattr(type(module).forward, "_torchdynamo_orig_callable", None)
        if original is not None:
            module.forward = MethodType(original, module)


class VideoCodec:
    def __init__(self, weights: Path, device: torch.device):
        self.device = device
        self.model = AutoencoderKLHunyuanVideo.from_pretrained(
            weights, torch_dtype=torch.float16, local_files_only=True,
        ).to(device).eval()
        if self.model.config.scaling_factor != VAE_SCALE:
            raise ValueError("Unexpected VAE scaling factor for F2LF")
        use_eager_forwards(self.model)

    @torch.inference_mode()
    def encode_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            if image.size not in IMAGE_SIZES:
                raise ValueError("Reference images must be 768×512 or aligned 1536×1024")
            rgb = np.array(image.convert("RGB"))
        if rgb.shape[:2] == (2 * HEIGHT, 2 * WIDTH):
            rgb = cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(rgb.copy())[None]
        values = image.to(self.device, dtype=torch.float16).permute(3, 0, 1, 2)[None] / 127.5 - 1
        posterior = self.model.encode(values, return_dict=False, opt_tiling=True)[0]
        digest = hashlib.sha256(image.numpy().tobytes()).digest()
        seed = (int.from_bytes(digest[:8], "little") + REFERENCE_SEED) % (2**63 - 1)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(posterior.mean.shape, device=self.device, dtype=posterior.mean.dtype, generator=generator)
        latent = (posterior.mean + posterior.std * noise)[0].permute(1, 2, 3, 0).float().mul(VAE_SCALE).cpu()
        if latent.shape != (1, HEIGHT // 8, WIDTH // 8, 16) or not torch.isfinite(latent).all():
            raise ValueError("Invalid encoded reference latent")
        return latent

    @torch.inference_mode()
    def decode(self, body: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        values = body.to(self.device, dtype=torch.float16).div(VAE_SCALE).permute(3, 0, 1, 2)[None]
        cudnn = torch.backends.cudnn
        policy = cudnn.flags(
            enabled=cudnn.enabled, benchmark=False, deterministic=True, allow_tf32=cudnn.allow_tf32,
        ) if deterministic else nullcontext()
        with policy:
            video = self.model.decode(values, return_dict=False)[0][0]
        if video.shape != (3, 121, HEIGHT, WIDTH) or not torch.isfinite(video).all():
            raise ValueError("Invalid decoded F2LF frame sequence")
        return ((video.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu()
