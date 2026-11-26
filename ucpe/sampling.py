"""Euler sampling with generated latents followed by clean reference latents."""

from collections.abc import Callable

import torch


def sample_video(
    denoiser: Callable,
    noise: torch.Tensor,
    reference: torch.Tensor | None,
    text: tuple[torch.Tensor, torch.Tensor],
    negative_text: tuple[torch.Tensor, torch.Tensor] | None,
    cameras: dict,
    *,
    num_steps: int,
    scheduler_scale: float,
    guidance_weight: float,
    scale_factor: tuple[float, float, float],
) -> torch.Tensor:
    """Sample packed THWC latents using the checkpoint's camera attention path."""
    device = noise.device
    frames, height, width, _ = noise.shape
    if num_steps < 2:
        raise ValueError("The Euler schedule requires at least two points")
    reference_count = 0 if reference is None else len(reference)
    if reference is not None and (
        reference.shape[1:] != noise.shape[1:] or not 1 <= reference_count < frames
    ):
        raise ValueError("The reference must match the latent grid and leave room for generated frames")
    prefix_count = reference_count if reference_count > 1 else 0
    generated_count = frames - prefix_count
    packed_count = generated_count + reference_count
    order = torch.arange(packed_count, device=device)
    positions = order
    if reference_count == 1:
        # The independently encoded image represents time zero in its own slot.
        cameras = {
            **cameras,
            "viewmats": [torch.cat((view, view[:1])) for view in cameras["viewmats"]],
            "Ks": [torch.cat((matrix, matrix[:1])) for matrix in cameras["Ks"]],
            "latent_shapes": [(packed_count, height, width)],
        }
        positions = torch.cat((order[:-1], order[:1]))
    elif prefix_count:
        # Move the video prefix to the packed reference slots, retaining its time.
        order = torch.cat((order[prefix_count:], order[:prefix_count]))
        positions = order
    token_types = torch.zeros(packed_count, device=device, dtype=torch.long)
    token_types[generated_count:] = 1
    metadata = {
        "type_ids": token_types,
        "view_ids": torch.zeros_like(token_types),
        "permutation": order,
        "inverse": torch.argsort(order),
        "skip_type_embeddings": reference_count == 0,
    }
    visual_ends = torch.tensor([0, packed_count], device=device, dtype=torch.int32)
    visual_positions = [
        positions,
        torch.arange(height // denoiser.patch_size[1]),
        torch.arange(width // denoiser.patch_size[2]),
    ]
    mask = token_types.to(noise.dtype)[:, None, None, None].expand(packed_count, height, width, 1)
    t_grid = torch.linspace(1, 0, num_steps, device=device)
    t_grid = scheduler_scale * t_grid / (1 + (scheduler_scale - 1) * t_grid)

    def text_inputs(encoding: tuple[torch.Tensor, torch.Tensor]):
        tokens, pooled = encoding
        ends = torch.tensor([0, len(tokens)], device=device, dtype=torch.int32)
        return tokens, pooled, ends, torch.arange(len(tokens))

    positive = text_inputs(text)
    negative = text_inputs(negative_text) if negative_text is not None else None
    if guidance_weight != 1 and negative is None:
        raise ValueError("Classifier-free guidance requires negative text features")

    def velocity(visual, time, features):
        tokens, pooled, ends, positions = features
        return denoiser(
            visual, tokens, pooled, time, visual_ends, ends, visual_positions, positions,
            ucpe_inputs=cameras, scale_factor=scale_factor, navio_inputs=metadata,
        )[:generated_count]

    sample = noise[prefix_count:].clone()
    tail = sample[:0] if reference is None else reference.to(sample)
    for now, delta in zip(t_grid[:-1], torch.diff(t_grid), strict=True):
        content = torch.cat((sample, tail))
        visual = torch.cat((content, torch.zeros_like(content), mask), dim=-1)
        time = now[None] * 1000
        pred = velocity(visual, time, positive)
        if guidance_weight != 1:
            null = velocity(visual, time, negative)
            pred = null + guidance_weight * (pred - null)
        sample += delta * pred
    output = torch.cat((tail, sample)) if prefix_count else sample
    if not torch.isfinite(output).all():
        raise ValueError("The model produced non-finite latents")
    return output
