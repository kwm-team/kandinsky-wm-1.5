"""Export a complete inference state dictionary from dense and EMA tensor shards."""

from pathlib import Path

import torch
from safetensors.torch import save_file


def read_shards(directory: Path) -> list[dict[str, torch.Tensor]]:
    """Read consecutively numbered dimension-zero shards."""
    paths = sorted(
        (path for path in directory.glob("*.pt") if path.stem.isdecimal()),
        key=lambda path: int(path.stem),
    )
    if not paths or [path.stem for path in paths] != list(map(str, range(len(paths)))):
        raise ValueError(f"Expected consecutive numbered tensor shards in {directory}")
    states = [torch.load(path, map_location="cpu", mmap=True, weights_only=True) for path in paths]
    keys = set(states[0])
    for state in states:
        if state.keys() != keys or not all(isinstance(value, torch.Tensor) for value in state.values()):
            raise ValueError(f"Inconsistent tensor shards in {directory}")
    return states


def merge(dense_dir: Path, adapter_dir: Path) -> dict[str, torch.Tensor]:
    """Select EMA adapter tensors and reconstruct the full denoiser."""
    base = read_shards(dense_dir)
    adapter = read_shards(adapter_dir)
    if not adapter[0] or any(".ucpe_branch." not in name for name in adapter[0]):
        raise ValueError("The adapter must contain UCPE branch parameters")
    result = {}
    for name in sorted(base[0].keys() | adapter[0].keys()):
        source = adapter if name in adapter[0] else base
        pieces = [shard[name] for shard in source]
        if any(x.shape[1:] != pieces[0].shape[1:] or x.dtype != pieces[0].dtype for x in pieces):
            raise ValueError(f"Inconsistent shard dimension or dtype for {name}")
        result[name] = torch.cat(pieces, dim=0).contiguous()
        if name in base[0] and name in adapter[0]:
            expected = sum(shard[name].shape[0] for shard in base)
            if result[name].shape != torch.Size([expected, *base[0][name].shape[1:]]):
                raise ValueError(f"Adapter shape disagrees with dense parameter {name}")
    return result


def export(dense_dir: Path, adapter_dir: Path, destination: Path) -> None:
    """Write the merged state to a new Safetensors file."""
    if destination.exists():
        raise FileExistsError(destination)
    state = merge(dense_dir, adapter_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, destination, metadata={"format": "pt"})
    print(f"Saved {len(state)} tensors, {sum(x.numel() for x in state.values()):,} parameters")
