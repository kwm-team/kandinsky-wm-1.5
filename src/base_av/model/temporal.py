"""Per-segment text cross-attention for five-second image-to-video generation.

Each Qwen token sequence has independent rotary positions. Pooled CLIP encodes
the scene and conditions the entire clip. Smooth routing applies group priors
inside the attention softmax using the per-group FlashAttention normalizers.
"""

import math
from dataclasses import dataclass

import torch


def frame_priors(policy="hard"):
    """Give the first generated plane and clean reference plane to the scene."""
    if policy not in ("hard", "smooth"):
        raise ValueError("Temporal policy must be hard or smooth")
    weights = torch.zeros(32, 6, dtype=torch.float32)
    weights[0, 0] = weights[31, 0] = 1
    centers = [3.5 + 6 * i for i in range(5)]
    for position in range(1, 31):
        if policy == "hard":
            weights[position, 1 + (position - 1) // 6] = 1
        elif position <= centers[0]:
            weights[position, 1] = 1
        elif position >= centers[-1]:
            weights[position, 5] = 1
        else:
            left = next(i for i in range(4) if centers[i] <= position < centers[i + 1])
            u = (position - centers[left]) / 6
            alpha = 3 * u * u - 2 * u * u * u
            weights[position, left + 1] = 1 - alpha
            weights[position, left + 2] = alpha
    return weights


@dataclass(frozen=True)
class TemporalCondition:
    text: torch.Tensor
    pooled: torch.Tensor
    group_lengths: tuple

    def __post_init__(self):
        if (
            len(self.group_lengths) != 6
            or any(n <= 0 for n in self.group_lengths)
            or self.text.ndim != 2
            or self.text.shape[0] != sum(self.group_lengths)
            or self.pooled.ndim != 2
            or self.pooled.shape[0] != 1
        ):
            raise ValueError(
                "Expected six nonempty Qwen groups and one scene-only pooled CLIP"
            )


def encode_temporal_text(encoder, scene=None, captions=None, *, negative_caption=None):
    if negative_caption is not None:
        if not isinstance(negative_caption, str):
            raise ValueError("Negative caption must be a string")
        groups = [negative_caption] * 6
    else:
        if (
            not isinstance(scene, str)
            or not scene.strip()
            or not isinstance(captions, (list, tuple))
            or len(captions) != 5
            or any(not isinstance(c, str) or not c.strip() for c in captions)
        ):
            raise ValueError("Expected scene and five whole-scene captions")
        groups = [scene] + [scene + " " + caption for caption in captions]
    # The native encoder accepts one caption. Separate calls preserve the
    # independent Qwen sequences. The first group supplies pooled CLIP.
    encoded = [encoder.encode(group) for group in groups]
    lengths = tuple(len(text) for text, _ in encoded)
    return TemporalCondition(
        torch.cat([text for text, _ in encoded]), encoded[0][1], lengths
    )


class PackedRouting:
    """Pack active frame-group pairs for FlashAttention cross-attention."""

    def __init__(self, text_lengths, patches, device, policy="hard"):
        if (
            len(text_lengths) != 6
            or any(
                not isinstance(n, int) or isinstance(n, bool) or n <= 0
                for n in text_lengths
            )
            or not isinstance(patches, int)
            or isinstance(patches, bool)
            or patches <= 0
        ):
            raise ValueError(
                "Expected six positive group lengths and a positive spatial patch count"
            )
        self.weights = frame_priors(policy)
        self.text_lengths = tuple(text_lengths)
        self.patches = patches
        offsets = [0]
        for n in text_lengths:
            offsets.append(offsets[-1] + n)
        self.text_cu = torch.tensor(offsets, dtype=torch.int32, device=device)
        self.text_rope = torch.cat(
            [torch.arange(n, device=device) for n in text_lengths]
        )
        routes, qindices, kindices, klengths = [], [], [], []
        primary, secondary, logprimary, logsecondary = [], [], [], []
        for frame in range(32):
            active = self.weights[frame].nonzero().flatten().tolist()
            start = len(routes)
            for group in active:
                routes.append((frame, group))
                qindices.extend(range(frame * patches, (frame + 1) * patches))
                a, b = offsets[group : group + 2]
                kindices.extend(range(a, b))
                klengths.append(b - a)
            primary.append(start)
            secondary.append(start + 1 if len(active) == 2 else start)
            logprimary.append(math.log(float(self.weights[frame, active[0]])))
            logsecondary.append(
                math.log(float(self.weights[frame, active[1]]))
                if len(active) == 2
                else -math.inf
            )
        self.routes = routes
        self.qindex = torch.tensor(qindices, dtype=torch.long, device=device)
        self.kindex = torch.tensor(kindices, dtype=torch.long, device=device)
        self.cuq = (
            torch.arange(len(routes) + 1, dtype=torch.int32, device=device) * patches
        )
        cumulative = [0]
        for n in klengths:
            cumulative.append(cumulative[-1] + n)
        self.cuk = torch.tensor(cumulative, dtype=torch.int32, device=device)
        self.maxk = max(klengths)
        self.primary = torch.tensor(primary, dtype=torch.long, device=device)
        self.secondary = torch.tensor(secondary, dtype=torch.long, device=device)
        self.logprimary = torch.tensor(logprimary, device=device)[:, None, None]
        self.logsecondary = torch.tensor(logsecondary, device=device)[:, None, None]
        self.identity_queries = len(routes) == 32

    def validate_video(self, x, text, time_positions, token_types, patch_size):
        if (
            list(patch_size) != [1, 2, 2]
            or x.ndim != 4
            or len(x) != 32
            or x.shape[1] % 2
            or x.shape[2] % 2
            or (x.shape[1] // 2) * (x.shape[2] // 2) != self.patches
        ):
            raise ValueError(
                "Temporal text requires 31 generated latent planes, a clean reference plane, and patch size [1, 2, 2]"
            )
        expected_positions = torch.tensor(list(range(31)) + [0], device=x.device)
        expected_types = torch.tensor([0] * 31 + [1], device=x.device)
        if not torch.equal(time_positions, expected_positions) or not torch.equal(
            token_types, expected_types
        ):
            raise ValueError(
                "Expected chronological generated positions and a clean I2V reference at position 0"
            )
        if (
            len(text) != sum(self.text_lengths)
            or text.device != self.text_cu.device
            or x.device != self.text_cu.device
        ):
            raise ValueError(
                "Text length or device differs from the registered temporal layout"
            )

    def attention(self, query, key, value):
        from flash_attn import flash_attn_varlen_func

        if (
            query.ndim != 3
            or key.ndim != 3
            or key.shape != value.shape
            or query.shape[0] != 32 * self.patches
            or key.shape[0] != sum(self.text_lengths)
            or query.shape[1:] != key.shape[1:]
            or query.device != self.text_cu.device
            or key.device != query.device
            or value.device != query.device
        ):
            raise ValueError(
                "Runtime query/key layout differs from the validated temporal layout"
            )
        packed_q = (
            query if self.identity_queries else query.index_select(0, self.qindex)
        )
        packed_k = key.index_select(0, self.kindex)
        packed_v = value.index_select(0, self.kindex)
        out, lse, _ = flash_attn_varlen_func(
            packed_q,
            packed_k,
            packed_v,
            self.cuq,
            self.cuk,
            self.patches,
            self.maxk,
            dropout_p=0.0,
            causal=False,
            return_attn_probs=True,
        )
        if self.identity_queries:
            return out
        heads, dim = query.shape[1:]
        if lse.shape != (heads, len(self.routes) * self.patches):
            raise ValueError(
                f"Unexpected FlashAttention log-sum-exp layout: {lse.shape}"
            )
        chunks = out.reshape(len(self.routes), self.patches, heads, dim)
        logz = lse.T.reshape(len(self.routes), self.patches, heads)
        loga = logz.index_select(0, self.primary) + self.logprimary
        logb = logz.index_select(0, self.secondary) + self.logsecondary
        # Weight each route by both its prior and its softmax partition function.
        fraction_b = torch.sigmoid(logb - loga)[..., None]
        result = chunks.index_select(0, self.primary).float() * (1 - fraction_b)
        result.add_(chunks.index_select(0, self.secondary).float() * fraction_b)
        return result.to(query.dtype).reshape(query.shape)
