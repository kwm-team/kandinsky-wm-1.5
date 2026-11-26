"""Optional exact-attention kernels for causal video generation.

Unmasked attention over a preselected span; the caller (block_causal_attention)
supplies each visible prefix/chunk span. Each span is bidirectional, including
rectangular current-to-prefix. causal=True here would silently change visibility.
FA3 is imported from flash_attn_interface (flash_attn.flash_attn_func is FA2 here);
FA3 beta returns (out, lse) -> unwrap first item.
"""
from functools import lru_cache

import torch
from torch.nn import functional as F

BACKENDS = ("sdpa", "flash2", "flash3")


@lru_cache(maxsize=None)
def flash_function(backend: str):
    try:
        if backend == "flash3":
            try:
                from flash_attn_interface import flash_attn_func
            except ModuleNotFoundError as error:
                if error.name != "flash_attn_interface":
                    raise
                from flash_attn_3.flash_attn_interface import flash_attn_func
        elif backend == "flash2":
            from flash_attn import flash_attn_func
        else:
            raise ValueError(f"Unknown FlashAttention backend: {backend}")
    except ImportError as error:
        raise RuntimeError(
            f"{backend} was requested but its extension is unavailable; install it or select sdpa"
        ) from error
    return flash_attn_func


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, backend: str = "sdpa") -> torch.Tensor:
    """Unmasked attention, BHLD layout in/out. FA3 wants BLHD -> transpose 1/2."""
    if backend == "sdpa":
        return F.scaled_dot_product_attention(query, key, value)
    if backend not in BACKENDS:
        raise ValueError(f"Unknown attention backend {backend!r}; expected one of {BACKENDS}")
    if query.device.type != "cuda" or query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"{backend} requires CUDA FP16/BF16 inputs; use sdpa for CPU/FP32")
    if backend == "flash3" and torch.cuda.get_device_capability(query.device)[0] != 9:
        raise ValueError("flash3 requires a Hopper GPU; select sdpa or flash2 on this device")
    out = flash_function(backend)(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), causal=False)
    if isinstance(out, tuple):
        out = out[0]
    return out.transpose(1, 2)
