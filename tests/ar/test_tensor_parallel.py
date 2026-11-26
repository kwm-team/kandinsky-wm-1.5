from types import SimpleNamespace

import torch
from torch import nn

import ar.tensor_parallel as tp


def make_attention():
    return SimpleNamespace(
        num_heads=28,
        to_query=nn.Linear(28, 28),
        to_key=nn.Linear(28, 28),
        to_value=nn.Linear(28, 28),
        out_layer=nn.Linear(28, 28),
    )


def test_tp7_partitions_heads_and_ffn_with_independent_weight_storage(monkeypatch):
    attention, cross = make_attention(), make_attention()
    ffn = SimpleNamespace(in_layer=nn.Linear(28, 56), out_layer=nn.Linear(56, 28))
    camera = nn.Module()
    camera.num_heads, camera.head_dim, camera.attn_dim = 7, 4, 28
    camera.q_proj, camera.k_proj, camera.v_proj = (nn.Linear(28, 28) for _ in range(3))
    camera.out_proj = nn.Linear(28, 28)
    block = SimpleNamespace(self_attention=attention, cross_attention=cross,
                            feed_forward=ffn, ucpe_branch=camera)
    model = SimpleNamespace(visual_transformer_blocks=[block])
    dense_query = attention.to_query.weight
    dense_row = ffn.out_layer.weight
    activation = torch.randn(2, 28)
    expected_qkv = [layer(activation)[:, 12:16] for layer in (attention.to_query, attention.to_key, attention.to_value)]
    monkeypatch.setattr(tp.dist, "get_world_size", lambda group: 7)
    monkeypatch.setattr(tp.dist, "get_rank", lambda group: 3)
    tp.shard_visual_blocks(model, group=None)
    assert attention.num_heads == cross.num_heads == 4
    assert (camera.num_heads, camera.attn_dim) == (1, 4)
    torch.testing.assert_close(attention.tp_qkv.weight[:4], dense_query[12:16])
    for actual, expected in zip(attention.tp_qkv(activation).chunk(3, dim=-1), expected_qkv, strict=True):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(ffn.out_layer.weight, dense_row[:, 24:32])
    assert attention.tp_qkv.weight.untyped_storage().nbytes() == 3 * 4 * 28 * 4
    assert ffn.out_layer.weight.untyped_storage().nbytes() == 28 * 8 * 4


def test_row_shard_adds_bias_once_after_reduction(monkeypatch):
    layer = nn.Linear(8, 3)
    value = torch.randn(5, 8)
    other_half = torch.nn.functional.linear(value[:, 4:], layer.weight[:, 4:])
    monkeypatch.setattr(tp.dist, "all_reduce", lambda tensor, group: tensor.add_(other_half))
    shard = tp.RowParallelLinear(layer, 0, 4, group=None)
    with torch.inference_mode():
        torch.testing.assert_close(shard(value[:, :4]), layer(value))
