"""Visual packing and token reordering for camera-conditioned sampling."""

import torch


def visual_input(latents: torch.Tensor, clean: torch.Tensor, metadata: dict) -> torch.Tensor:
    reference = metadata["type_ids"].to(latents.device).bool()
    x = torch.where(reference.reshape(-1, 1, 1, 1), clean.to(latents), latents)
    mask = reference.to(latents.dtype).reshape(-1, 1, 1, 1).expand(*latents.shape[:-1], 1)
    return torch.cat((x, torch.zeros_like(x), mask), dim=-1)


def reorder_tokens(tokens: torch.Tensor, frame_order: torch.Tensor) -> torch.Tensor:
    order = frame_order.to(tokens.device)
    return tokens.reshape(len(order), -1, tokens.shape[-1]).index_select(0, order).reshape_as(tokens)
