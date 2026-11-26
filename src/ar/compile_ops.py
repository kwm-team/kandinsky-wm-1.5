"""Compile the stand's precision-preserving DiT pointwise operations."""

from functools import lru_cache

import torch


def _scale_shift(value, scale, shift, index):
    return value * (scale.index_select(0, index) + 1.0) + shift.index_select(0, index)


def _gate(value, output, gate, index):
    return value + gate.index_select(0, index) * output


@lru_cache(maxsize=1)
def _ops():
    from .native.model.nn import apply_rotary

    options = {"emulate_precision_casts": True, "triton.cudagraphs": False}
    rotary, affine, gate = (
        torch.compile(function, fullgraph=True, dynamic=True, options=options)
        for function in (apply_rotary, _scale_shift, _gate)
    )

    def modulate(norm, value, scale, shift, index):
        with torch.autocast(device_type="cuda", enabled=False):
            return affine(norm(value), scale, shift, index)

    return rotary, modulate, gate


def configure(model):
    from .native.model.dit import TransformerDecoderBlock, TransformerEncoderBlock
    from .native.model.nn import MultiheadSelfAttention, OutLayer

    rotary, modulate, gate = _ops()
    for module in model.modules():
        if isinstance(module, MultiheadSelfAttention):
            module._speedup_rotary = rotary
        if isinstance(module, (TransformerEncoderBlock, TransformerDecoderBlock, OutLayer)):
            module._speedup_modulate = modulate
            module._speedup_gate = gate
