import torch

from ar.cache import StreamCache
from ar.read_graph import copy_placements, static_cache


@torch.inference_mode()
def test_graph_placements_keep_storage_and_refresh_only_committed_history():
    source = StreamCache(2)
    source.token_axis = 2
    key = torch.arange(6, dtype=torch.float32).view(1, 1, 3, 2)
    source.place(1, key, 2 * key)
    target = static_cache(source)
    token = key[:, :, :1]
    joined = target.join(1, token, token)
    pointer = joined[0].data_ptr()
    source.place(1, key + 4, key + 8)
    copy_placements(source, target)
    assert joined[0].data_ptr() == pointer
    torch.testing.assert_close(joined[0][:, :, :3], key + 4)
    torch.testing.assert_close(joined[1][:, :, :3], key + 8)
    torch.testing.assert_close(joined[0][:, :, 3:], token)
    assert target.placed[0] is None
