"""Tiny NaN/Inf tracer for hunting down divergence in the UCPE branch.

Drop ``check(tensor, "name")`` calls anywhere in a forward path. The first
non-finite tensor seen in a given forward pass prints a single diagnostic
line and sets a sticky flag so subsequent layers in the same pass also log
(handy when NaN propagates: we get the chain rather than just the final
victim). The flag is reset by ``begin_forward()`` at the top of
``DiT.forward``.

Disabled by default — set ``UCPE_NAN_DEBUG=1`` to enable. Cost when
enabled is ~1 ms per check on a (T, D) tensor with T=13440, D=1792.
"""

import os
import torch

_ENABLED = bool(int(os.environ.get("UCPE_NAN_DEBUG", "0") or "0"))
_TRACE_EVERY = int(os.environ.get("UCPE_TRACE_EVERY", "10") or "10")
_TRACE_BLOCKS = set(
    int(x) for x in (os.environ.get("UCPE_TRACE_BLOCKS", "0,15,31") or "").split(",")
    if x.strip()
)
_RANK = int(os.environ.get("RANK", "0") or "0")
_TRACE_RANK = int(os.environ.get("UCPE_TRACE_RANK", "0") or "0")
_FORWARD_ID = 0
_SAW_NONFINITE = False


def enabled() -> bool:
    return _ENABLED


def trace_blocks() -> set:
    return _TRACE_BLOCKS


def begin_forward():
    global _FORWARD_ID, _SAW_NONFINITE
    _FORWARD_ID += 1
    _SAW_NONFINITE = False


def trace_stats(tensor, name: str, block_idx: int = None):
    """Unconditionally log min/max/abs_mean/abs_max stats every
    ``UCPE_TRACE_EVERY`` forwards. Optional block_idx filters via
    ``UCPE_TRACE_BLOCKS``. Use for time-series of magnitudes leading up to
    a NaN — complements ``check`` which only fires on non-finite values.
    """
    if not _ENABLED:
        return tensor
    if _RANK != _TRACE_RANK:
        return tensor
    if _FORWARD_ID % _TRACE_EVERY != 0:
        return tensor
    if block_idx is not None and block_idx not in _TRACE_BLOCKS:
        return tensor
    if not torch.is_tensor(tensor):
        return tensor
    try:
        with torch.no_grad():
            t = tensor.detach()
            abs_t = t.abs()
            nan_count = int(torch.isnan(t).sum().item())
            inf_count = int(torch.isinf(t).sum().item())
            abs_max = float(abs_t.max().item()) if t.numel() > 0 else float("nan")
            abs_mean = float(abs_t.mean().item()) if t.numel() > 0 else float("nan")
            t_min = float(t.min().item()) if t.numel() > 0 else float("nan")
            t_max = float(t.max().item()) if t.numel() > 0 else float("nan")
    except Exception as e:
        print(f"[TRACE fwd={_FORWARD_ID} {name} EXCEPTION {e}]", flush=True)
        return tensor
    print(
        f"[TRACE fwd={_FORWARD_ID} {name} shape={tuple(tensor.shape)} dtype={tensor.dtype} | "
        f"abs_mean={abs_mean:.3e} abs_max={abs_max:.3e} "
        f"min={t_min:+.3e} max={t_max:+.3e} nan={nan_count} inf={inf_count}]",
        flush=True,
    )
    return tensor


def check(tensor, name: str):
    """Log if ``tensor`` contains NaN/Inf. No-op when disabled.

    Once a forward pass has seen one non-finite, every subsequent check that
    fires also logs — so the propagation chain is visible.
    """
    global _SAW_NONFINITE
    if not _ENABLED:
        return tensor
    if not torch.is_tensor(tensor):
        return tensor
    try:
        finite_mask = torch.isfinite(tensor)
        all_finite = bool(finite_mask.all().item())
    except Exception:
        return tensor
    if all_finite and not _SAW_NONFINITE:
        return tensor

    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    finite = tensor[finite_mask]
    if finite.numel() > 0:
        f_max = float(finite.abs().max().item())
        f_mean = float(finite.abs().mean().item())
    else:
        f_max = float("nan")
        f_mean = float("nan")
    tag = "FIRST" if not _SAW_NONFINITE else "leak"
    if not all_finite:
        _SAW_NONFINITE = True
    print(
        f"[NAN_DEBUG fwd={_FORWARD_ID} {tag}] {name}: "
        f"nan={nan_count} inf={inf_count} "
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"finite_abs_max={f_max:.3e} finite_abs_mean={f_mean:.3e}",
        flush=True,
    )
    return tensor
