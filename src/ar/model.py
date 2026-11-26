"""Load the native DiT architecture and its velocity-grid inference weights."""

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

import torch


@dataclass(frozen=True)
class SolverSettings:
    nfe: int
    grid_points: int
    chunk_planes: int
    history_planes: int
    shift: float
    final_step_size_scale: float
    total_substeps: int
    eps: float


@dataclass(frozen=True)
class StreamSettings:
    history: int
    pinned_reference: bool
    reset_frames: int
    retain_after_reset: int
    ucpe_scale: float


@dataclass(frozen=True)
class Configuration:
    architecture: dict
    solver: SolverSettings
    stream: StreamSettings


def load_configuration(path: Path) -> Configuration:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("format") not in ("kwm-av-ar-v1", "kwm-av-piflow-v1"):
        raise ValueError("Expected an AR weight directory")
    solver = SolverSettings(**raw["solver"])
    stream = StreamSettings(
        **{k: raw["stream"][k] for k in StreamSettings.__dataclass_fields__}
    )
    if (solver.nfe, solver.grid_points, solver.chunk_planes) != (8, 8, 4):
        raise ValueError(
            "This model requires eight evaluations, eight grid points and four-plane chunks"
        )
    if not (
        stream.pinned_reference
        and 1 <= stream.retain_after_reset <= stream.history <= solver.history_planes
    ):
        raise ValueError("Invalid bounded pinned-reference history")
    if (
        stream.reset_frames < 4
        or stream.reset_frames % 4
        or not 0 < stream.ucpe_scale <= 4
    ):
        raise ValueError("Invalid reset interval or UCPE scale")
    return Configuration(raw["architecture"], solver, stream)


def load_model(
    architecture: dict, checkpoint: Path, grid_points: int, device: torch.device
):
    """Retain the donor's FP32 parameters and autocast its inference calls."""
    from safetensors.torch import load_file

    from .native.model.dit import get_dit
    from .solver import GridOutLayer

    with torch.device("meta"):
        model = get_dit(architecture)
    model.to_empty(device=device)
    model.reset_parameters()
    model.out_layer = GridOutLayer(
        [deepcopy(model.out_layer) for _ in range(grid_points)]
    )
    model.load_state_dict(load_file(checkpoint), strict=True)
    # reset_parameters recreates nonpersistent frequency buffers on the CPU.
    return model.to(device).eval().requires_grad_(False)
