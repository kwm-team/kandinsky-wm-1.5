"""Load a self-contained F2LF bundle and render a five-second transition."""

from dataclasses import dataclass
import json
from pathlib import Path
from types import MethodType

from safetensors.torch import load_file
import torch

from ucpe.pipeline import TextEncoder
from ucpe.runtime.model.dit import get_dit

from .camera import BODY_PLANES, CameraPath, HEIGHT, WIDTH
from .codec import VideoCodec
from .kernels import configure
from .sampling import Predictor, sample


@dataclass(frozen=True)
class Generation:
    frames: torch.Tensor
    latents: torch.Tensor


def _text_forward_compatibility(module, inputs, kwargs):
    kwargs.pop("mm_token_type_ids", None)
    return inputs, kwargs


def _text_features(weights, config, prompt, scene, device):
    encoder = TextEncoder(config, weights, device)
    qwen = encoder.qwen
    for module in qwen.model.modules():
        original = getattr(module.forward, "_torchdynamo_orig_callable", None)
        if original is not None:
            module.forward = original if getattr(original, "__self__", None) is not None else MethodType(original, module)
    qwen.model.requires_grad_(False)
    qwen.model.register_forward_pre_hook(_text_forward_compatibility, with_kwargs=True)

    template = "\n".join(qwen.PROMPT_TEMPLATE["template"]["video"])
    crop = qwen.PROMPT_TEMPLATE["crop_start"]["video"]
    count = len(qwen.processor.tokenizer(template.format(prompt), add_special_tokens=False)["input_ids"])
    if count > crop + qwen.max_length or len(encoder.tokenizer(scene, truncation=False)["input_ids"]) > 77:
        raise ValueError("The video prompt or short scene description exceeds the encoder limit")
    tokens, _ = qwen([prompt], type_of_content="video")
    clip_inputs = encoder.tokenizer(
        [scene], max_length=77, truncation=True, add_special_tokens=True, padding="max_length", return_tensors="pt",
    ).to(device)
    pooled = encoder.clip(**clip_inputs).pooler_output
    return tokens.cpu(), pooled.cpu()


def _load_denoiser(weights, architecture, device):
    with torch.device("meta"):
        model = get_dit(architecture)
    model.to_empty(device=device)
    model.load_state_dict(load_file(weights / "transformer.safetensors"), strict=True)
    states = load_file(weights / "buffers.safetensors")
    buffers = dict(model.named_buffers())
    if states.keys() != buffers.keys():
        raise ValueError("Canonical F2LF buffers do not match the model topology")
    for name, target in buffers.items():
        saved = states[name]
        if target.shape != saved.shape or target.dtype != saved.dtype:
            raise ValueError(f"Unexpected shape or dtype for F2LF buffer {name}")
        target.copy_(saved)
    model.eval().requires_grad_(False)
    configure(model)
    return model


class F2LFPipeline:
    """Generate between independently encoded first and last reference frames."""

    def __init__(self, weights: Path, device: str = "cuda"):
        self.weights = Path(weights)
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available() or torch.cuda.get_device_capability(self.device)[0] != 9:
            raise ValueError("The reference F2LF path requires an NVIDIA Hopper GPU")
        self.config = json.loads((self.weights / "config.json").read_text(encoding="utf-8"))
        if self.config.get("format") != "kwm-av-f2lf-v1":
            raise ValueError("Expected an F2LF inference bundle")

    @torch.inference_mode()
    def __call__(
        self, first: Path, last: Path, path: CameraPath, prompt: str, scene: str, seed: int, *, deterministic: bool = False,
    ) -> Generation:
        codec = VideoCodec(self.weights / "vae", self.device)
        anchors = torch.cat([codec.encode_image(first), codec.encode_image(last)]).float()
        del codec
        torch.cuda.empty_cache()

        positive = _text_features(self.weights, self.config["qwen"], prompt, scene, self.device)
        negative = load_file(self.weights / "negative.safetensors")
        texts = (positive, (negative["text_embeds"], negative["pooled_embed"]))
        for tokens, pooled in texts:
            if tokens.ndim != 2 or tokens.shape[-1] != 3584 or pooled.numel() != 768:
                raise ValueError("Unexpected F2LF text feature dimensions")
            if not torch.isfinite(tokens).all() or not torch.isfinite(pooled).all():
                raise ValueError("Nonfinite F2LF text features")
        torch.cuda.empty_cache()

        generator = torch.Generator("cpu").manual_seed(seed)
        noise = torch.randn((BODY_PLANES, HEIGHT // 8, WIDTH // 8, 16), generator=generator)
        model = _load_denoiser(self.weights, self.config["architecture"], self.device)
        predictor = Predictor(model, anchors, texts, path, self.device)
        body = sample(predictor, noise.to(self.device)).cpu()
        del model, predictor
        torch.cuda.empty_cache()

        codec = VideoCodec(self.weights / "vae", self.device)
        return Generation(codec.decode(body, deterministic=deterministic), body)
