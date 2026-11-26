"""Precision-preserving CUDA kernels for the reference F2LF sampler."""

from contextlib import contextmanager
from functools import lru_cache
import os


def configure_environment():
    os.environ["TRITON_DEFAULT_FP_FUSION"] = "0"
    os.environ["TORCHINDUCTOR_EMULATE_PRECISION_CASTS"] = "1"
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")


@contextmanager
def reference_matmul_precision():
    import torch

    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _rotate(x, rope):
    import torch

    with torch.autocast(device_type="cuda", enabled=False):
        value = x.reshape(*x.shape[:-1], -1, 1, 2).to(torch.float32)
        result = rope[..., 0] * value[..., 0] + rope[..., 1] * value[..., 1]
        return result.reshape(*x.shape)


def _scale_shift(x, scale, shift, index):
    import torch

    with torch.autocast(device_type="cuda", enabled=False):
        return x * (scale.index_select(0, index) + 1.0) + shift.index_select(0, index)


def _gate(x, out, gate, index):
    import torch

    with torch.autocast(device_type="cuda", enabled=False):
        return x + gate.index_select(0, index) * out


def _camera_attention(q, k, v):
    from flash_attn_interface import flash_attn_func

    result = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=False)
    return (result[0] if isinstance(result, (tuple, list)) else result).transpose(1, 2)


@lru_cache(maxsize=1)
def _pointwise():
    import torch

    if any(os.environ.get(name) == "1" for name in ("TORCH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE")):
        raise ValueError("The reference F2LF path requires Torch compilation")
    configure_environment()
    rotate, scale_shift, gate = (
        torch.compile(function, fullgraph=True, dynamic=True, options={"emulate_precision_casts": True})
        for function in (_rotate, _scale_shift, _gate)
    )

    def modulate(norm, x, scale, shift, index):
        with torch.autocast(device_type="cuda", enabled=False):
            return scale_shift(norm(x.float()), scale, shift, index)

    return rotate, modulate, gate


def configure(model):
    import torch
    from ucpe.runtime.model.dit import TransformerDecoderBlock, TransformerEncoderBlock
    from ucpe.runtime.model.nn import MultiheadSelfAttention, OutLayer

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.accumulated_cache_size_limit = 4096
    torch._dynamo.config.compiled_autograd = False
    rotate, modulate, gate = _pointwise()
    for module in model.modules():
        if isinstance(module, MultiheadSelfAttention):
            module._speedup_rotary = rotate
        if isinstance(module, (TransformerDecoderBlock, TransformerEncoderBlock, OutLayer)):
            module._speedup_modulate = modulate
            module._speedup_gate = gate
    for block in model.visual_transformer_blocks:
        if hasattr(block, "ucpe_branch"):
            block.ucpe_branch._speedup_attention = _camera_attention
