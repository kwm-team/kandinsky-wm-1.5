from types import SimpleNamespace

import pytest
import torch

import ar.runtime as runtime
from ar.cache import StreamCache
from ar.codec import OverlapDecoder
from ar.model import SolverSettings, StreamSettings
from ar.rope import first_tail, retain_first_tail


@pytest.mark.parametrize("token_axis", [0, 2])
def test_history_trim_preserves_startup_and_last_accepted_planes(token_axis):
    cache = StreamCache(1)
    cache.token_axis = token_axis
    for plane in range(5):
        tokens = torch.full((2, 1, 1), plane).movedim(0, token_axis)
        cache.append(0, tokens, -tokens)
    cache.place_for(object())
    cache.place(0, *cache.readable(0))

    assert first_tail([10, 11, 12, 13, 14], keep=3) == [10, 13, 14]
    retain_first_tail([cache], spatial=2, keep=3)

    keys, values = cache.readable_context(0)
    assert keys.flatten().tolist() == [0, 0, 3, 3, 4, 4]
    assert torch.equal(values, -keys)
    assert cache.placed[0] is None


def test_decoder_reuses_only_visible_latents_when_history_is_trimmed():
    decoded_inputs = []

    def decode(latents):
        decoded_inputs.append(latents.tolist())
        return torch.arange(4 * len(latents) - 3)

    overlap = OverlapDecoder(SimpleNamespace(decode=decode), torch.tensor([0]))
    assert overlap.decode(torch.tensor([1, 2, 3, 4]), 4).tolist() == list(range(1, 17))
    assert overlap.decode(torch.tensor([5, 6, 7, 8]), 2).tolist() == list(range(5, 13))
    assert overlap.decode(torch.tensor([9, 10, 11, 12]), 1).tolist() == [5, 6, 7, 8]
    assert decoded_inputs == [
        [0, 1, 2, 3, 4],
        [3, 4, 5, 6, 7, 8],
        [5, 6, 9, 10, 11, 12],
    ]


def test_ucpe_hook_keeps_a_device_float32_scalar(monkeypatch):
    def no_op(*args, **kwargs):
        pass

    def stub_stream(*args, **kwargs):
        return SimpleNamespace(begin=no_op, call=no_op, committed=no_op)

    monkeypatch.setattr(runtime, "NativeStream", stub_stream)
    monkeypatch.setattr(runtime, "install_single_caption_attention", no_op)
    branch = torch.nn.Identity()
    settings = StreamSettings(8, True, 120, 4, 1.2)
    model = SimpleNamespace(
        visual_transformer_blocks=[SimpleNamespace(ucpe_branch=branch)]
    )
    session = runtime.ARRuntime(
        model,
        torch.zeros(1, 2, 2, 16),
        torch.eye(4),
        torch.eye(3),
        (torch.zeros(4, 3584), torch.zeros(768)),
        SolverSettings(8, 8, 4, 8, 10.0, 0.5, 128, 1e-5),
        settings,
        seed=0,
    )

    assert session.ucpe_scale.shape == ()
    assert session.ucpe_scale.dtype == torch.float32
    assert session.ucpe_scale.device == session.device
    bf16 = torch.tensor([-0.125, 0.56, 2.44], dtype=torch.bfloat16)
    assert torch.equal(
        branch(bf16), bf16 * torch.tensor(settings.ucpe_scale, dtype=torch.float32)
    )
