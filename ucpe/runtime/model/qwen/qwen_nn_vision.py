import torch
import os
from torch import nn
from flash_attn import flash_attn_varlen_func
from .qwen_nn_utils import rotate_half, Qwen2RMSNorm
import time

@torch.autocast(device_type="cuda", dtype=torch.float32)
def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
):
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class Qwen2_5_VLMLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        if config.hidden_act == "silu":
            self.act_fn = nn.SiLU()
        else:
            raise ValueError("Unimplemented activation")

    def reset_parameters(self):
        self.gate_proj.reset_parameters()
        self.up_proj.reset_parameters()
        self.down_proj.reset_parameters()

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def reset_parameters(self):
        self.proj.reset_parameters()

    def forward(self, hidden_states: torch.Tensor):
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.theta = theta
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def reset_parameters(self):
        self.inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, seqlen: int):
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen2_5_VLPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def reset_parameters(self):
        self.ln_q.reset_parameters()
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x: torch.Tensor):
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class Qwen2_5_VLVisionAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.to_query = nn.Linear(self.dim, self.dim, bias=True)
        self.to_key = nn.Linear(self.dim, self.dim, bias=True)
        self.to_value = nn.Linear(self.dim, self.dim, bias=True)

        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config

    def reset_parameters(self):
        self.to_query.reset_parameters()
        self.to_key.reset_parameters()
        self.to_value.reset_parameters()
        self.proj.reset_parameters()

    def forward( # TODO check cuseqlens for several visuals
        self,
        hidden_states,
        cu_seqlens,
        cos,
        sin
    ):
        emb_dim = hidden_states.shape[-1]

        query_states = self.to_query(hidden_states)
        key_states = self.to_key(hidden_states)
        value_states = self.to_value(hidden_states)

        query_states = query_states.reshape(query_states.shape[0], self.num_heads, -1)
        key_states = key_states.reshape(key_states.shape[0], self.num_heads, -1)
        value_states = value_states.reshape(value_states.shape[0], self.num_heads, -1)

        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        flash_kwargs = {}

        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=self.scaling,
            causal=False,
            **flash_kwargs,
        )

        attn_output = attn_output.reshape(attn_output.shape[0], -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5_VLVisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen2_5_VLVisionAttention(config=config)
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def reset_parameters(self):
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.attn.reset_parameters()
        self.mlp.reset_parameters()

    @torch.compile(dynamic=True)
    def forward(
        self,
        hidden_states,
        cu_seqlens,
        cos,
        sin
    ):
        normed = self.norm1(hidden_states)
        hidden_states = hidden_states + self.attn(
            normed,
            cu_seqlens,
            cos, sin
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

        return hidden_states
