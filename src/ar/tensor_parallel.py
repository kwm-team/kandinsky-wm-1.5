"""Inference tensor parallelism for the AR DiT's visual transformer blocks."""

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


def _parameter(value, dtype):
    return nn.Parameter(
        value.detach().to(dtype=dtype or value.dtype, copy=True, memory_format=torch.contiguous_format),
        requires_grad=False,
    )


class RowParallelLinear(nn.Module):
    def __init__(self, source, start, stop, group, dtype=None):
        super().__init__()
        self.weight = _parameter(source.weight[:, start:stop], dtype)
        self.bias = None if source.bias is None else _parameter(source.bias, dtype)
        self.group = group

    def forward(self, value):
        output = F.linear(value, self.weight)
        dist.all_reduce(output, group=self.group)
        if self.bias is not None:
            output.add_(self.bias.to(output.dtype))
        return output


def _column_shard(source, start, stop, dtype):
    layer = nn.Linear(source.in_features, stop - start, bias=source.bias is not None, device="meta")
    layer.weight = _parameter(source.weight[start:stop], dtype)
    if source.bias is not None:
        layer.bias = _parameter(source.bias[start:stop], dtype)
    return layer


def _bounds(width, rank, size):
    return width * rank // size, width * (rank + 1) // size


def _shard_heads(module, projections, output, rank, size, group, dtype):
    head_dim = getattr(module, projections[0]).out_features // module.num_heads
    first, last = _bounds(module.num_heads, rank, size)
    start, stop = first * head_dim, last * head_dim
    for name in projections:
        setattr(module, name, _column_shard(getattr(module, name), start, stop, dtype))
    setattr(module, output, RowParallelLinear(getattr(module, output), start, stop, group, dtype))
    module.num_heads = last - first


def _pack_self_attention(module):
    names = ("to_query", "to_key", "to_value")
    parts = [getattr(module, name) for name in names]
    packed = nn.Linear(parts[0].in_features, 3 * parts[0].out_features, device="meta")
    packed.weight = nn.Parameter(torch.cat([part.weight.detach() for part in parts]), requires_grad=False)
    packed.bias = nn.Parameter(torch.cat([part.bias.detach() for part in parts]), requires_grad=False)
    module.tp_qkv = packed
    for name in names:
        delattr(module, name)


def shard_visual_blocks(model, group, *, dtype=None):
    """Keep complete local heads and reduce the residual-width projections."""
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    if size == 1:
        return
    for block in model.visual_transformer_blocks:
        if min(block.self_attention.num_heads, block.cross_attention.num_heads,
               block.feed_forward.in_layer.out_features) < size:
            raise ValueError("TP rank count exceeds the visual attention heads or feed-forward width")

    for block in model.visual_transformer_blocks:
        block._collect_logs = False  # The AR inference path does not consume training statistics.
        for attention in (block.self_attention, block.cross_attention):
            _shard_heads(attention, ("to_query", "to_key", "to_value"), "out_layer", rank, size, group, dtype)
        _pack_self_attention(block.self_attention)

        camera = getattr(block, "ucpe_branch", None)
        if camera is not None and camera.num_heads >= size:
            _shard_heads(camera, ("q_proj", "k_proj", "v_proj"), "out_proj", rank, size, group, dtype)
            camera.attn_dim = camera.num_heads * camera.head_dim

        feed_forward = block.feed_forward
        start, stop = _bounds(feed_forward.in_layer.out_features, rank, size)
        feed_forward.in_layer = _column_shard(feed_forward.in_layer, start, stop, dtype)
        feed_forward.out_layer = RowParallelLinear(feed_forward.out_layer, start, stop, group, dtype)
