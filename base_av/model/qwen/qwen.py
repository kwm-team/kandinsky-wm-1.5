import torch
from torch import nn
from .qwen_nn_utils import Qwen2RMSNorm
from .qwen_nn_vlm import Qwen2_5_VLDecoderLayer, Qwen2_5_VLRotaryEmbedding


class Qwen2_5_VLTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen2_5_VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

    def forward(self, input_ids):
        # One unpadded caption. Text positions are identical on all three RoPE axes.
        hidden = self.embed_tokens(input_ids)
        length = hidden.shape[1]
        positions = torch.arange(length, device=hidden.device).expand(3, 1, length)
        cos, sin = self.rotary_emb(hidden, positions)
        rope = (cos[:, 0], sin[:, 0])
        cu_seqlens = torch.tensor([0, length], dtype=torch.int32, device=hidden.device)
        hidden = hidden[0]
        for layer in self.layers:
            hidden = layer(hidden, cu_seqlens, length, rope)
        return self.norm(hidden)
