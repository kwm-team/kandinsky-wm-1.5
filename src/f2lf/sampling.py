"""Bidirectional F2LF denoising with independent endpoint references."""

import torch

from .camera import CameraPath, HEIGHT, WIDTH


def pack_latents(body: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    if body.ndim != 4 or body.shape[-1] != 16 or anchors.shape != (2, *body.shape[1:]):
        raise ValueError("Expected THWC16 latents and two independent reference planes")
    frames = torch.cat([body, anchors], dim=0)
    mask = frames.new_zeros((*frames.shape[:-1], 1))
    mask[-2:] = 1
    return torch.cat([frames, torch.zeros_like(frames), mask], dim=-1)


class Predictor:
    def __init__(self, model, anchors, texts, path: CameraPath, device: torch.device):
        self.model = model
        self.anchors = anchors.to(device)
        self.frames = len(path.poses)
        self.device = device
        _, height, width, _ = anchors.shape
        total = self.frames + 2
        order = torch.arange(total, device=device)
        types = torch.zeros_like(order)
        types[-2:] = 1
        self.metadata = dict(type_ids=types, view_ids=torch.zeros_like(types), permutation=order, inverse=order)
        temporal_rope = torch.cat([order[:self.frames], torch.tensor([0, self.frames - 1], device=device)])
        self.metadata["rope_t"] = temporal_rope
        self.visual_rope = [temporal_rope, torch.arange(height // 2, device=device), torch.arange(width // 2, device=device)]
        self.visual_ends = torch.tensor([0, total], device=device, dtype=torch.int32)
        self.camera = dict(
            viewmats=[torch.from_numpy(path.viewmats).to(device)],
            Ks=[torch.from_numpy(path.intrinsics).to(device)],
            latent_shapes=[(total, height, width)], frame_wh=[(WIDTH, HEIGHT)],
            trans_scale=[path.translation_scale],
        )
        self.texts = [self._text_inputs(tokens, pooled) for tokens, pooled in texts]

    def _text_inputs(self, tokens, pooled):
        return (
            tokens.to(self.device), pooled.reshape(1, -1).to(self.device),
            torch.tensor([0, len(tokens)], device=self.device, dtype=torch.int32),
            torch.arange(len(tokens), device=self.device),
        )

    def __call__(self, body, sigma, *, positive):
        tokens, pooled, ends, rope = self.texts[0 if positive else 1]
        visual = pack_latents(body, self.anchors)
        level = torch.as_tensor(sigma, device=self.device, dtype=torch.float32).reshape(1) * 1000
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model(
                visual, tokens, pooled, level, self.visual_ends, ends, self.visual_rope, rope,
                scale_factor=(1.0, 2.0, 2.0), ucpe_inputs=self.camera, navio_inputs=self.metadata,
            )
        return output[:self.frames].float()


@torch.no_grad()
def sample(predict, noise: torch.Tensor) -> torch.Tensor:
    levels = torch.linspace(1, 0, 51, device=noise.device)
    levels = 10 * levels / (1 + 9 * levels)
    body = noise
    for current, following in zip(levels[:-1], levels[1:], strict=True):
        positive = predict(body, current, positive=True)
        negative = predict(body, current, positive=False)
        velocity = negative + 5.0 * (positive - negative)
        body = body.float() + (following - current) * velocity
    if not torch.isfinite(body).all():
        raise FloatingPointError("F2LF produced nonfinite body latents")
    return body.detach()
