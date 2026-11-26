"""Checkpoint-compatible Navio condV1 packing and physical camera identities."""

import re

import torch


CAMERA_TO_VIEW = {camera: index for index, camera in enumerate((1, 3, 4, 5, 6, 7, 8))}


def camera_view_index(
    camera_id: int | str | float | None = None, identifier: str = ""
) -> int:
    matches = {
        int(value) for value in re.findall(r"__cam(\d+)(?=\D|$)", str(identifier))
    }
    if camera_id is not None and str(camera_id).strip().lower() not in (
        "",
        "nan",
        "none",
    ):
        number = float(camera_id)
        if not number.is_integer():
            raise ValueError(f"Invalid Navio camera ID: {camera_id}")
        matches.add(int(number))
    if len(matches) != 1 or next(iter(matches)) not in CAMERA_TO_VIEW:
        raise ValueError(
            f"Missing, conflicting or unsupported Navio camera: {camera_id!r}, {identifier!r}"
        )
    return CAMERA_TO_VIEW[next(iter(matches))]


def packing_metadata(
    lengths: list[int],
    counts: list[int],
    views: list[int],
    device: str | int | torch.device,
) -> dict:
    if not (len(lengths) == len(counts) == len(views)):
        raise ValueError("Navio lengths, prefix counts and camera IDs must match")
    orders, types, positions, cameras = [], [], [], []
    start = 0
    for length, count, view in zip(lengths, counts, views):
        length, count, view = int(length), int(count), int(view)
        if not 0 <= count < length or not 0 <= view < 7:
            raise ValueError(f"Invalid Navio clip: T={length}, k={count}, view={view}")
        order = torch.cat((torch.arange(count, length), torch.arange(count)))
        orders.append(order + start)
        positions.append(order)
        types.append(
            torch.cat(
                (
                    torch.zeros(length - count, dtype=torch.long),
                    torch.ones(count, dtype=torch.long),
                )
            )
        )
        cameras.append(torch.full((length,), view, dtype=torch.long))
        start += length
    permutation = torch.cat(orders).to(device)
    return {
        "permutation": permutation,
        "inverse": torch.argsort(permutation),
        "rope_t": torch.cat(positions).to(device),
        "type_ids": torch.cat(types).to(device),
        "view_ids": torch.cat(cameras).to(device),
    }


def visual_input(
    latents: torch.Tensor, clean: torch.Tensor, metadata: dict
) -> torch.Tensor:
    reference = metadata["type_ids"].to(latents.device).bool()
    x = torch.where(reference.reshape(-1, 1, 1, 1), clean.to(latents), latents)
    mask = (
        reference.to(latents.dtype).reshape(-1, 1, 1, 1).expand(*latents.shape[:-1], 1)
    )
    return torch.cat((x, torch.zeros_like(x), mask), dim=-1)


def reorder_tokens(tokens: torch.Tensor, frame_order: torch.Tensor) -> torch.Tensor:
    order = frame_order.to(tokens.device)
    return (
        tokens.reshape(len(order), -1, tokens.shape[-1])
        .index_select(0, order)
        .reshape_as(tokens)
    )


def configure_navio(conf: object) -> None:
    """Derive construction and loader settings from the single trainer switch."""
    mode = str(getattr(conf.trainer.params, "i2v_mode", "k5")).lower()
    conf.dit.params.i2v_mode = mode
    conf.data.i2v_mode = mode
    if hasattr(conf, "metrics"):
        conf.metrics.i2v_mode = mode
