"""Read cached VAE conditioning bound to an ordered pair of reference images."""

from hashlib import sha256
from pathlib import Path

import numpy as np
from safetensors import SafetensorError, safe_open

from .camera import HEIGHT, WIDTH


def load_reference_latents(path: Path, first: Path, last: Path) -> np.ndarray:
    expected_hashes = {
        "first_sha256": sha256(first.read_bytes()).hexdigest(),
        "last_sha256": sha256(last.read_bytes()).hexdigest(),
    }
    try:
        with safe_open(str(path), framework="np", device="cpu") as bundle:
            metadata = bundle.metadata() or {}
            if metadata.get("format") != "kwm-av-f2lf-reference-latents-v1" or bundle.keys() != ["anchors"]:
                raise ValueError("Expected an F2LF reference latent bundle with one anchors tensor")
            if any(metadata.get(name) != digest for name, digest in expected_hashes.items()):
                raise ValueError("Cached F2LF anchors must match the two ordered reference images")
            anchors = bundle.get_tensor("anchors")
    except SafetensorError as error:
        raise ValueError(f"Invalid F2LF reference latent bundle: {error}") from error

    shape = (2, HEIGHT // 8, WIDTH // 8, 16)
    if anchors.shape != shape or anchors.dtype != np.float32 or not np.isfinite(anchors).all():
        raise ValueError(f"Expected finite float32 F2LF anchors of shape {shape}")
    return anchors
