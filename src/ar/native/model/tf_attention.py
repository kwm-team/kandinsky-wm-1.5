"""Exact doubled visibility ported from the successful TF4200 line."""
import torch
from .attention import attention as _span_attention

def latent_chunk_sizes(planes, chunk):
    """Split video latent planes into autoregressive chunks; first chunk absorbs the
    remainder (latent 0 = one RGB frame, later latents = four). Ported from carla-swm."""
    if chunk <= 0 or chunk > planes:
        raise ValueError(f"Chunk of {chunk} latents does not fit {planes} planes")
    remainder = planes % chunk
    return [chunk + remainder] + [chunk] * (planes // chunk - 1)


def block_causal_attention(query, key, value, prefix, chunk, backend="sdpa"):
    """query/key/value: [1, heads, tokens, head_dim]. prefix = leading clean tokens.
    chunk: tokens per generated chunk (int) or explicit list of per-chunk token sizes."""
    total = query.shape[2]
    if not 0 <= prefix <= total:
        raise ValueError("Prefix token count must lie inside the packed sequence")
    if prefix == total:
        return _span_attention(query, key, value, backend)
    current = total - prefix
    sizes = list(chunk) if not isinstance(chunk, int) else (
        [chunk] * (current // chunk) + ([current % chunk] if current % chunk else []))
    if any(s <= 0 for s in sizes) or sum(sizes) != current:
        raise ValueError(f"Chunk sizes {sizes} do not sum to the {current} current tokens")
    parts = []
    if prefix:
        parts.append(_span_attention(query[:, :, :prefix], key[:, :, :prefix], value[:, :, :prefix], backend))
    start = prefix
    for size in sizes:
        stop = start + size
        parts.append(_span_attention(query[:, :, start:stop], key[:, :, :stop], value[:, :, :stop], backend))
        start = stop
    return torch.cat(parts, dim=2)


def doubled_causal_attention(query, key, value, prefix, sizes, backend="sdpa", memory=None):
    """Visibility for RF's doubled sequence [prefix | noisy_0, restored_0, noisy_1, restored_1, ...].

    A noisy chunk attends to the prefix, to the *restored* (odd) chunks before it, and to
    itself -- never to another noisy chunk (inference never provides a noisy copy of an
    earlier chunk). Same rule as s2v `tf_structural_doc` / carla-swm doubled_causal_attention:
    self within each duplicated chunk, ORed with a causal mask over the odd (restored) keys.
    Keys are gathered because the visible set is non-contiguous. `sizes` = per-span token
    counts (even count: noisy/restored interleaved), summing to total-prefix.

    `memory` = сколько ПРЕДЫДУЩИХ чанков видно (None/0 — все, как было). Окно режет только
    видимость: rope остаётся абсолютным, префикс (anchor) виден всегда. Нужен, чтобы обучение
    совпадало с инференсом, который идёт скользящим окном с выселением."""
    if len(sizes) % 2:
        raise ValueError(f"A doubled sequence needs an even span count, got {len(sizes)}")
    total = query.shape[2]
    starts, at = [], prefix
    for size in sizes:
        starts.append(at)
        at += size
    if at != total:
        raise ValueError(f"Spans cover {at - prefix} planes but the sequence holds {total - prefix}")
    device = query.device
    head = torch.arange(prefix, device=device)
    parts = []
    if prefix:
        parts.append(_span_attention(query[:, :, :prefix], key[:, :, :prefix], value[:, :, :prefix], backend))
    for index, size in enumerate(sizes):
        own = torch.arange(starts[index], starts[index] + size, device=device)
        # Both halves see the SAME previous restored (odd) spans. Using index
        # directly makes lo even for a restored query and leaks noisy history.
        lo = 1 if not memory else max(1, 2 * (index // 2 - int(memory)) + 1)
        context = [torch.arange(starts[e], starts[e] + sizes[e], device=device) for e in range(lo, index, 2)]
        # anchor (prefix) is also evicted once `memory` chunks precede this one (carla evicts the
        # start image too): chunk index//2 keeps the anchor only while it is inside the window.
        inc_head = bool(prefix) and (not memory or (index // 2) < int(memory))
        head_part = [head] if inc_head else []
        visible = torch.cat([*head_part, *context, own]) if (context or inc_head) else own
        parts.append(_span_attention(query[:, :, starts[index]:starts[index] + size],
                                     key[:, :, visible], value[:, :, visible], backend))
    return torch.cat(parts, dim=2)



def attend_doubled(q, k, v, layout, tokens_per_frame):
    prefix = int(layout.get("prefix_frames", 0))
    total = q.shape[-2] // tokens_per_frame
    content = total - prefix
    if content % 2:
        raise ValueError("Doubled TF content must have even length")
    sizes = [n * tokens_per_frame for n in latent_chunk_sizes(content // 2, int(layout["chunk_frames"])) for _ in range(2)]
    return doubled_causal_attention(q, k, v, prefix * tokens_per_frame, sizes,
                                    backend=layout.get("backend", "sdpa"),
                                    memory=layout.get("memory_chunks"))


def attend_span(q, k, v, layout, tokens_per_frame, owner):
    """Dense clean prefix + current chunk; cache the prefix across Euler steps.

    Caches belong to an attention module (DiT or transformed UCPE separately),
    and are discarded between chunks and before the gradient forward.
    """
    prefix = int(layout.get("prefix_frames", 0)) * tokens_per_frame
    backend = layout.get("backend", "sdpa")
    request = layout.get("kv_cache")
    if request is not None:
        if torch.is_grad_enabled():
            raise RuntimeError("RF rollout cache must never enter a gradient forward")
        slot = int(request.get("slot", 0))
        if request["mode"] == "read":
            if prefix:
                raise ValueError("Cache read must pack only the current chunk")
            cached = getattr(owner, "_rf_kv", {}).get(slot)
            if cached is None:
                raise RuntimeError("RF prefix cache read without a matching write")
            pk, pv = cached
            return _span_attention(q, torch.cat((pk, k), dim=2),
                                   torch.cat((pv, v), dim=2), backend)
        if request["mode"] != "write":
            raise ValueError("Unknown RF cache mode")
        if not hasattr(owner, "_rf_kv"):
            owner._rf_kv = {}
        owner._rf_kv[slot] = (k[:, :, :prefix].detach().clone(),
                              v[:, :, :prefix].detach().clone())
    if not 0 <= prefix < q.shape[2]:
        raise ValueError("Span requires a nonempty current chunk")
    current = _span_attention(q[:, :, prefix:], k, v, backend)
    if not prefix:
        return current
    clean = _span_attention(q[:, :, :prefix], k[:, :, :prefix], v[:, :, :prefix], backend)
    return torch.cat((clean, current), dim=2)


def clear_rf_cache(model):
    for module in model.modules():
        if hasattr(module, "_rf_kv"):
            module._rf_kv.clear()
