"""OmniNWM-style Plucker tokens mixed with visual tokens in a side attention."""

import torch
import torch.nn.functional as F
from torch import nn

from .plucker import build_plucker_raymaps_from_viewmats
from .ucpe import validate_ucpe_attention_config


def build_raymap_input(camera_inputs, *, device, dtype):
    """Return (sum_T, H, W, 6) in OmniNWM's (moment, direction) order."""
    maps = []
    for viewmats, intrinsics, shape, frame_wh in zip(
        camera_inputs["viewmats"], camera_inputs["Ks"],
        camera_inputs["latent_shapes"], camera_inputs["frame_wh"],
    ):
        if viewmats.shape[-2:] != (4, 4):
            raise ValueError("RayMap requires SE(3) 4x4 viewmats; velocity pose modes are unsupported")
        latent_t, latent_h, latent_w = shape
        frame_w, frame_h = frame_wh
        rays = build_plucker_raymaps_from_viewmats(
            viewmats.to(device), intrinsics.to(device), frame_w, frame_h,
            latent_h, latent_w, raw_frames_per_latent=1, dtype=dtype,
        )
        if rays.shape[0] != latent_t:
            raise ValueError(f"RayMap has {rays.shape[0]} frames, expected {latent_t}")
        maps.append(torch.cat((rays[..., 3:], rays[..., :3]), dim=-1))
    return torch.cat(maps, dim=0)


class RayMapSideBranch(nn.Module):
    def __init__(self, model_dim, head_dim, attn_mask=None, chunk_len=3, attn_compress=8):
        super().__init__()
        validate_ucpe_attention_config(attn_mask, chunk_len)
        attn_dim = model_dim // attn_compress
        head = next(
            (candidate for candidate in (head_dim, head_dim // 2, 56, 32, 16, 8)
             if candidate > 0 and attn_dim % candidate == 0), None
        )
        if head is None:
            raise ValueError(f"Cannot divide RayMap attention dimension {attn_dim} into heads")
        self.attn_dim = attn_dim
        self.num_heads = attn_dim // head
        self.head_dim = head
        self.attn_mask = attn_mask
        self.chunk_len = chunk_len
        self.visual_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.ray_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.visual_qkv = nn.Linear(model_dim, 3 * attn_dim, bias=False)
        self.ray_qkv = nn.Linear(model_dim, 3 * attn_dim, bias=False)
        self.q_norm = nn.RMSNorm(head)
        self.k_norm = nn.RMSNorm(head)
        self.out_proj = nn.Linear(attn_dim, model_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.visual_qkv.reset_parameters()
        self.ray_qkv.reset_parameters()
        self.q_norm.reset_parameters()
        self.k_norm.reset_parameters()
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, visual, raymap, cu_seqlens, latent_shapes):
        if visual.shape != raymap.shape:
            raise ValueError(f"RayMap tokens {raymap.shape} must match visual tokens {visual.shape}")
        v = self.visual_qkv(self.visual_norm(visual).to(visual.dtype))
        r = self.ray_qkv(self.ray_norm(raymap).to(raymap.dtype))
        outputs = []
        for i, (latent_t, height, width) in enumerate(latent_shapes):
            start, end = int(cu_seqlens[i].item()), int(cu_seqlens[i + 1].item())
            per_frame = height * width
            if end - start != latent_t * per_frame:
                raise ValueError("RayMap token count does not match latent shape")
            # Frame-major ordering makes one temporal chunk contain both modalities.
            paired = torch.stack((r[start:end].reshape(latent_t, per_frame, -1),
                                  v[start:end].reshape(latent_t, per_frame, -1)), dim=1)
            paired = paired.reshape(-1, 3, self.num_heads, self.head_dim)
            q, k, val = paired.unbind(dim=1)
            q = self.q_norm(q).transpose(0, 1).unsqueeze(0)
            k = self.k_norm(k).transpose(0, 1).unsqueeze(0)
            val = val.transpose(0, 1).unsqueeze(0)
            if self.attn_mask is None:
                attended = F.scaled_dot_product_attention(q, k, val)
            else:
                chunk_tokens = self.chunk_len * 2 * per_frame
                chunks = []
                for query_start in range(0, q.shape[-2], chunk_tokens):
                    query_end = min(query_start + chunk_tokens, q.shape[-2])
                    key_start = (query_start if self.attn_mask == "A" else
                                 max(0, query_start - chunk_tokens) if self.attn_mask == "B" else 0)
                    chunks.append(F.scaled_dot_product_attention(
                        q[..., query_start:query_end, :],
                        k[..., key_start:query_end, :],
                        val[..., key_start:query_end, :],
                    ))
                attended = torch.cat(chunks, dim=-2)
            attended = attended.squeeze(0).transpose(0, 1).reshape(latent_t, 2, per_frame, self.attn_dim)
            outputs.append(attended[:, 1].reshape(-1, self.attn_dim))
        return self.out_proj(torch.cat(outputs, dim=0))
