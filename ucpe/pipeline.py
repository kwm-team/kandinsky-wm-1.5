"""Standalone camera-controlled video generation."""

import json
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from accelerate import init_empty_weights
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms import functional as vision
from transformers import CLIPTextModel, CLIPTokenizer

from .runtime.model.dit import get_dit
from .runtime.model.qwen.qwen_embedder import Qwen2_5_VLTextEmbedder
from .runtime.model.vae import AutoencoderKLHunyuanVideo
from .sampling import sample_video
from .trajectory import Trajectory


NEGATIVE_PROMPT = (
    "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, "
    "low quality, ugly, deformed, walking backwards"
)
VIDEO_PREFIX_FRAMES = 5

@contextmanager
def _source_precision():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("medium")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


class TextEncoder:
    """Encode each prompt with packed Qwen features and a pooled CLIP vector."""

    def __init__(self, config: dict, components: Path, device: torch.device):
        self.qwen = Qwen2_5_VLTextEmbedder(
            OmegaConf.create({
                **config,
                "model_checkpoint_path": str(components / "text_encoder/model.pt"),
                "processor_checkpoint_path": str(components / "tokenizer"),
            }),
            device,
        )
        self.clip = CLIPTextModel.from_pretrained(components / "text_encoder_2", local_files_only=True).to(device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(components / "text_encoder_2", local_files_only=True)
        self.device = device

    @torch.inference_mode()
    def __call__(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, _ = self.qwen([text], type_of_content="video")
        inputs = self.tokenizer(
            [text], max_length=77, truncation=True, add_special_tokens=True,
            padding="max_length", return_tensors="pt",
        ).to(self.device)
        return tokens, self.clip(**inputs).pooler_output


class UCPEPipeline:
    """Generate a camera-controlled video with an optional image or video prefix."""

    def __init__(self, weights: str | Path, components: str | Path | None = None, device: str = "cuda"):
        weights = Path(weights)
        components = Path(components) if components is not None else weights
        config = json.loads((weights / "config.json").read_text(encoding="utf-8"))
        if config.get("format") != "kwm-av-ucpe":
            raise ValueError("Expected a merged AV UCPE bundle")
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("UCPE generation requires a CUDA device")
        self.sampler = config["sampler"]
        with init_empty_weights(include_buffers=False):
            denoiser = get_dit(config["architecture"])
        state = load_file(weights / "model.safetensors", device="cpu")
        denoiser.load_state_dict(
            {name: tensor.to(device=self.device, dtype=torch.bfloat16) for name, tensor in state.items()},
            strict=True,
            assign=True,
        )
        self.denoiser = denoiser.to(device=self.device).eval()
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained(
            components / "vae", torch_dtype=torch.float16, local_files_only=True,
        ).to(self.device).eval()
        self.text_encoder = TextEncoder(config["qwen"], components, self.device)

    @torch.inference_mode()
    @_source_precision()
    def __call__(
        self,
        image: Image.Image | None,
        prompt: str,
        trajectory: Trajectory,
        *,
        seed: int = 42,
        width: int = 768,
        height: int = 512,
        negative_prompt: str = NEGATIVE_PROMPT,
        num_steps: int | None = None,
        guidance_weight: float | None = None,
        video: np.ndarray | None = None,
    ) -> np.ndarray:
        if width <= 0 or height <= 0 or width % 16 or height % 16:
            raise ValueError("Width and height must be positive multiples of 16")
        if not 5 <= trajectory.frames <= 121 or (trajectory.frames - 1) % 4:
            raise ValueError("Trajectory spans must contain 5 to 121 frames and have length 4k+1")
        if image is not None and video is not None:
            raise ValueError("Provide either an image or a video prefix")
        if video is not None and trajectory.frames <= VIDEO_PREFIX_FRAMES:
            raise ValueError("Video continuation requires a trajectory of at least nine frames")
        latent_shape = (1 + (trajectory.frames - 1) // 4, height // 8, width // 8)
        source_size = (width, height)
        if image is not None:
            source_size = image.size
        elif video is not None:
            if (video.ndim != 4 or video.shape[-1] != 3 or video.dtype != np.uint8
                    or len(video) < VIDEO_PREFIX_FRAMES or min(video.shape[1:3]) <= 0):
                raise ValueError("Video input must contain at least five uint8 RGB frames in THWC layout")
            source_size = (video.shape[2], video.shape[1])
        camera_inputs = trajectory.camera_inputs(latent_shape, source_size)
        reference_generator = torch.Generator(device=self.device).manual_seed(seed)
        noise_generator = torch.Generator(device=self.device).manual_seed(seed)
        reference = None
        if image is not None:
            pixels = resize_reference(image, width, height)
            tensor = torch.from_numpy(pixels).permute(2, 0, 1)[None, :, None]
            tensor = (tensor / 127.5 - 1.0).to(self.device, dtype=self.vae.dtype)
            reference = self.vae.encode(tensor).latent_dist.sample(generator=reference_generator)
            reference = reference[0].permute(1, 2, 3, 0).cpu() * self.vae.config.scaling_factor
        elif video is not None:
            prefix = torch.from_numpy(video[:VIDEO_PREFIX_FRAMES]).permute(3, 0, 1, 2)
            scale = min(source_size[1] / height, source_size[0] / width)
            resized = vision.resize(prefix, [int(source_size[1] / scale), int(source_size[0] / scale)])
            top, left = (resized.shape[-2] - height) // 2, (resized.shape[-1] - width) // 2
            prefix = vision.crop(resized, top, left, height, width)
            tensor = prefix[None].to(self.device, dtype=self.vae.dtype) / 127.5 - 1.0
            reference = self.vae.encode(tensor).latent_dist.sample(generator=reference_generator)
            reference = (reference * self.vae.config.scaling_factor)[0].permute(1, 2, 3, 0)
            if len(reference) != 2:
                raise ValueError("Expected two VAE latents from the five-frame video prefix")
        noise = torch.randn(
            *latent_shape, self.denoiser.in_visual_dim, generator=noise_generator,
            device=self.device, dtype=torch.float32,
        )
        positive = self.text_encoder(prompt)
        guidance = self.sampler["guidance_weight"] if guidance_weight is None else guidance_weight
        if not math.isfinite(guidance) or guidance < 0:
            raise ValueError("Guidance weight must be finite and nonnegative")
        negative = self.text_encoder(negative_prompt) if guidance != 1 else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latents = sample_video(
                self.denoiser, noise, reference, positive, negative, camera_inputs,
                num_steps=self.sampler["num_steps"] if num_steps is None else num_steps,
                scheduler_scale=self.sampler["scheduler_scale"],
                guidance_weight=guidance,
                scale_factor=tuple(self.sampler["scale_factor"]),
            )
            decoded = self.vae.decode(
                (latents / self.vae.config.scaling_factor).permute(3, 0, 1, 2)[None],
            ).sample[0]
        return ((decoded.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()


def resize_reference(image: Image.Image, width: int, height: int) -> np.ndarray:
    """Resize and center-crop the RGB reference with the native PIL policy."""
    image = image.convert("RGB")
    src_width, src_height = image.size
    scale = min(src_height / height, src_width / width)
    resized = image.resize((int(src_width / scale), int(src_height / scale)))
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return np.array(resized.crop((left, top, left + width, top + height)))
