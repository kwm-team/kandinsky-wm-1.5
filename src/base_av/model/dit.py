import torch
from torch import nn
from .nn import (TimeEmbeddings, TextEmbeddings, VisualEmbeddings, RoPE1D, RoPE3D, Modulation,
                 MultiheadSelfAttention, MultiheadCrossAttention, FeedForward, OutLayer,
                 apply_scale_shift_norm, apply_gate_sum)


class TransformerEncoderBlock(nn.Module):

    def __init__(self, model_dim, time_dim, ff_dim, head_dim):
        super().__init__()
        self.text_modulation = Modulation(time_dim, model_dim, 6)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttention(model_dim, head_dim)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def forward(self, x, time_embed, rope, cu_seqlens, time_embed_idx):
        self_attn_params, ff_params = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)

        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = apply_scale_shift_norm(self.self_attention_norm, x, scale, shift, time_embed_idx).type_as(x)
        out = self.self_attention(out, rope, cu_seqlens)
        x = apply_gate_sum(x, out, gate, time_embed_idx).type_as(x)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = apply_scale_shift_norm(self.feed_forward_norm, x, scale, shift, time_embed_idx).type_as(x)
        out = self.feed_forward(out)
        x = apply_gate_sum(x, out, gate, time_embed_idx).type_as(x)

        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim):
        super().__init__()
        self.visual_modulation = Modulation(time_dim, model_dim, 9)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttention(model_dim, head_dim)

        self.cross_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.cross_attention = MultiheadCrossAttention(model_dim, head_dim)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)
        self.model_dim = model_dim

    def forward(
        self, visual_embed, text_embed, time_embed, rope, visual_cu_seqlens,
        text_cu_seqlens, time_embed_idx, text_routing=None,
    ):
        self_attn_params, cross_attn_params, ff_params = torch.chunk(self.visual_modulation(time_embed), 3, dim=-1)

        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(self.self_attention_norm, visual_embed, scale, shift, time_embed_idx).type_as(visual_embed)
        visual_out = self.self_attention(visual_out, rope, visual_cu_seqlens)
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate, time_embed_idx).type_as(visual_embed)

        shift, scale, gate = torch.chunk(cross_attn_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(self.cross_attention_norm, visual_embed, scale, shift, time_embed_idx).type_as(visual_embed)
        visual_out = self.cross_attention(
            visual_out, text_embed, visual_cu_seqlens, text_cu_seqlens, routing=text_routing
        )
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate, time_embed_idx).type_as(visual_embed)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(self.feed_forward_norm, visual_embed, scale, shift, time_embed_idx).type_as(visual_embed)
        visual_out = self.feed_forward(visual_out)
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate, time_embed_idx).type_as(visual_embed)

        return visual_embed


class DiffusionTransformer3D(nn.Module):
    """Video-only backbone shared across Base AV generation modes."""

    def __init__(self, in_visual_dim, in_text_dim, in_text_dim2, model_dim, time_dim,
                 ff_dim, axes_dims, patch_size, num_text_blocks, num_visual_blocks):
        super().__init__()
        self.patch_size = patch_size
        head_dim = sum(axes_dims)
        self.visual_embeddings = VisualEmbeddings(2 * in_visual_dim + 1, model_dim, patch_size)
        self.visual_token_type_embeddings = nn.Embedding(2, model_dim)
        self.visual_rope_embeddings = RoPE3D(axes_dims)
        self.out_layer = OutLayer(model_dim, time_dim, in_visual_dim, patch_size)
        self.time_embeddings = TimeEmbeddings(model_dim, time_dim)
        self.text_embeddings = TextEmbeddings(in_text_dim, model_dim)
        self.pooled_text_embeddings = TextEmbeddings(in_text_dim2, time_dim)
        self.text_rope_embeddings = RoPE1D(head_dim)
        self.text_transformer_blocks = nn.ModuleList(
            TransformerEncoderBlock(model_dim, time_dim, ff_dim, head_dim)
            for _ in range(num_text_blocks)
        )
        self.visual_blocks = nn.ModuleList(
            TransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim)
            for _ in range(num_visual_blocks)
        )

    def forward(self, x, text, pooled, time, time_positions, token_types, text_routing=None):
        device = x.device
        text_seqlens = torch.tensor([0, len(text)], dtype=torch.int32, device=device)
        if text_routing is not None:
            text_routing.validate_video(x, text, time_positions, token_types, self.patch_size)
            text_seqlens = text_routing.text_cu
        frame_seqlens = torch.tensor([0, len(x)], dtype=torch.int32, device=device)
        time, time_idx = self.time_embeddings(time)
        time = time + self.pooled_text_embeddings(pooled)
        text = self.text_embeddings(text)
        text_positions = torch.arange(len(text), device=device) if text_routing is None else text_routing.text_rope
        text_rope = self.text_rope_embeddings(text_positions)
        for block in self.text_transformer_blocks:
            text = block(text, time, text_rope, text_seqlens, time_idx.expand(len(text)))

        video, frame_seqlens = self.visual_embeddings(x, frame_seqlens)
        video = video + self.visual_token_type_embeddings(token_types)[:, None, None, :]
        frames, height, width, channels = video.shape
        positions = [time_positions, torch.arange(height, device=device), torch.arange(width, device=device)]
        rope = self.visual_rope_embeddings(video.shape[:-1], positions, (1., 2., 2.)).flatten(0, 2)
        video = video.flatten(0, 2)
        seqlens = frame_seqlens * (height * width)
        for block in self.visual_blocks:
            video = block(
                video, text, time, rope, seqlens, text_seqlens, time_idx.expand(len(video)),
                text_routing=text_routing,
            )
        video = video.reshape(frames, height, width, channels)
        return self.out_layer(video, text, time, frame_seqlens, time_idx.expand(frames))
