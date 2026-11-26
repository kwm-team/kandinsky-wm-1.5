"""Activation collector for the UCPE / self-attention branch study.

Sibling of ``nan_debug.py``: a module-global, API-gated collector that taps the
two injected residuals inside ``TransformerDecoderBlock.forward`` (self-attention
and UCPE) and dumps a subsample of them to disk for offline PCA / distribution
analysis.

Design mirrors ``nan_debug``: the ``record(...)`` call sites live in ``dit.py``
behind an ``enabled()`` guard so a normal training/eval run pays nothing (the
flag defaults to False and is only flipped by ``enable(...)`` in the collection
script).

Per block, per branch, per sample we store:
  * ``*_tok``  — a deterministic random subsample of ``tokens_per_sample`` token
                 rows, ``(K, D)`` fp16. The SAME token indices are used for both
                 branches of a given (sample, block) so rows are paired.
  * ``*_mean`` — the mean over ALL tokens of that sample, ``(D,)`` fp16.

``flush()`` writes one ``.npz`` per (sample, timestep) under ``<out_dir>/raw/``.
The separate ``full_ucpe_only`` mode writes atomic ``.pt`` payloads and is used
only by the standalone inference collector.
"""

import json
import os
import tempfile

import numpy as np
import torch


# ---- configuration / state (set by enable()) --------------------------------
_ENABLED = False
_OUT_DIR = None
_BLOCKS = None            # set[int] of block indices to capture, or None => all
_TOKENS_PER_SAMPLE = 1024
_CAPTURE = "injected_residual"
_RANK = 0
_SEED = 0
_WRITER_RANK = 0
_SAMPLER_STEPS = None      # set[int] for endpoint capture, or None => all

# ---- per-forward context (set by set_seqlens / set_context) ------------------
_SEQLENS = None           # list[int] cumulative token boundaries (len = n_samples + 1)
_SAMPLE_GIDS = None       # list[str]  length n_samples
_CAPTIONS = None          # list[str]  length n_samples
_TIMESTEP = None          # float noise fraction
_METADATA = None          # arbitrary JSON-compatible per-job metadata
_LAYOUT_THW = None        # full-capture output layout
_SAMPLER_STEP = None
_FORWARD_KIND = None
_CAPTURED_STEP = None
_CAPTURED_TIMESTEP = None

# ---- accumulation ------------------------------------------------------------
_BUFFER = None            # {gid: {block_idx: {"sa_tok","sa_mean","ucpe_tok","ucpe_mean"}}}
_TOK_IDX_CACHE = None     # {(gid, block_idx): np.ndarray of row indices}

_PREFIX = {"self_attn": "sa", "ucpe": "ucpe"}


def enable(out_dir, blocks="all", tokens_per_sample=1024,
           capture="injected_residual", rank=0, seed=0, writer_rank=0,
           sampler_steps="all"):
    """Turn the collector on and set run-level config. Clears any buffer."""
    global _ENABLED, _OUT_DIR, _BLOCKS, _TOKENS_PER_SAMPLE, _CAPTURE, _RANK, _SEED
    global _WRITER_RANK, _SAMPLER_STEPS
    global _BUFFER, _TOK_IDX_CACHE
    _OUT_DIR = out_dir
    if blocks is None or blocks == "all":
        _BLOCKS = None
    else:
        _BLOCKS = set(int(b) for b in blocks)
    _TOKENS_PER_SAMPLE = int(tokens_per_sample)
    _CAPTURE = str(capture)
    if _CAPTURE not in ("injected_residual", "full_ucpe_only", "endpoint_ucpe"):
        raise ValueError(f"unsupported activation capture mode: {_CAPTURE}")
    _RANK = int(rank)
    _SEED = int(seed)
    _WRITER_RANK = int(writer_rank)
    if sampler_steps is None or sampler_steps == "all":
        _SAMPLER_STEPS = None
    else:
        _SAMPLER_STEPS = set(int(step) for step in sampler_steps)
    _BUFFER = {}
    _TOK_IDX_CACHE = {}
    os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
    _ENABLED = True


def disable():
    global _ENABLED, _BUFFER, _TOK_IDX_CACHE, _SEQLENS, _SAMPLE_GIDS, _CAPTIONS, _TIMESTEP
    global _METADATA, _LAYOUT_THW, _SAMPLER_STEP, _FORWARD_KIND
    global _CAPTURED_STEP, _CAPTURED_TIMESTEP, _SAMPLER_STEPS
    _ENABLED = False
    _BUFFER = None
    _TOK_IDX_CACHE = None
    _SEQLENS = _SAMPLE_GIDS = _CAPTIONS = _TIMESTEP = None
    _METADATA = _LAYOUT_THW = _SAMPLER_STEP = _FORWARD_KIND = None
    _CAPTURED_STEP = _CAPTURED_TIMESTEP = None
    _SAMPLER_STEPS = None


def enabled() -> bool:
    return _ENABLED


def capture_mode() -> str:
    return _CAPTURE


def set_seqlens(cu_seqlens):
    """Stash the per-sample token boundaries the block loop actually sees.

    Called from ``DiT.forward`` right before the visual-block loop with the
    POST-``fractal_flatten`` ``visual_cu_seqlens`` so the split matches the
    packed ``visual_embed`` the blocks operate on.
    """
    global _SEQLENS
    if not _ENABLED:
        return
    _SEQLENS = [int(x) for x in cu_seqlens.detach().cpu().tolist()]


def set_context(sample_gids, captions, timestep, metadata=None, layout_thw=None):
    """Set the per-forward sample identities (called by the collection script)."""
    global _SAMPLE_GIDS, _CAPTIONS, _TIMESTEP, _METADATA, _LAYOUT_THW
    global _CAPTURED_STEP, _CAPTURED_TIMESTEP
    _SAMPLE_GIDS = list(sample_gids)
    _CAPTIONS = list(captions)
    _TIMESTEP = float(timestep)
    _METADATA = dict(metadata or {})
    _LAYOUT_THW = tuple(int(x) for x in layout_thw) if layout_thw is not None else None
    _CAPTURED_STEP = _CAPTURED_TIMESTEP = None


def set_sampler_context(step, timestep, forward_kind):
    """Identify the sampler/model call for opt-in full inference capture."""
    global _SAMPLER_STEP, _TIMESTEP, _FORWARD_KIND
    if not _ENABLED:
        return
    _SAMPLER_STEP = int(step)
    _TIMESTEP = float(timestep)
    _FORWARD_KIND = str(forward_kind)


def _tok_indices(gid, block_idx, n_tokens):
    """Deterministic token-row subsample, shared across branches of a (sample, block)."""
    key = (gid, block_idx)
    idx = _TOK_IDX_CACHE.get(key)
    if idx is None:
        k = min(_TOKENS_PER_SAMPLE, n_tokens)
        # seed depends on run seed + sample + block (NOT branch) so sa/ucpe pair up.
        rng = np.random.default_rng(abs(hash((_SEED, gid, int(block_idx)))) % (2**32))
        idx = np.sort(rng.choice(n_tokens, size=k, replace=False))
        _TOK_IDX_CACHE[key] = idx
    return idx


def record(branch, block_idx, tensor):
    """Tap point. ``branch`` in {"self_attn","ucpe"}; ``tensor`` is (sum_T, D)."""
    global _CAPTURED_STEP, _CAPTURED_TIMESTEP
    if not _ENABLED:
        return
    if _BLOCKS is not None and int(block_idx) not in _BLOCKS:
        return
    if _SEQLENS is None or _SAMPLE_GIDS is None:
        return
    prefix = _PREFIX.get(branch)
    if prefix is None:
        return

    if _CAPTURE == "full_ucpe_only":
        if (
            branch != "ucpe"
            or _RANK != _WRITER_RANK
            or _SAMPLER_STEP != 0
            or _FORWARD_KIND != "conditional"
        ):
            return
        if _LAYOUT_THW is None:
            raise RuntimeError("full_ucpe_only capture requires layout_thw")
        if _CAPTURED_STEP is None:
            _CAPTURED_STEP = _SAMPLER_STEP
            _CAPTURED_TIMESTEP = _TIMESTEP
        expected_tokens = int(np.prod(_LAYOUT_THW))
        t = tensor.detach()
        if hasattr(t, "to_local"):
            t = t.to_local()
        for j, gid in enumerate(_SAMPLE_GIDS):
            start, end = _SEQLENS[j], _SEQLENS[j + 1]
            seg = t[start:end]
            if seg.shape[0] != expected_tokens:
                raise ValueError(
                    f"{gid}: token count {seg.shape[0]} does not match layout "
                    f"{_LAYOUT_THW} ({expected_tokens})"
                )
            _BUFFER.setdefault(gid, {})[int(block_idx)] = (
                seg.reshape(*_LAYOUT_THW, seg.shape[-1]).to(torch.bfloat16).cpu()
            )
        return

    if _CAPTURE == "endpoint_ucpe":
        if (
            branch != "ucpe"
            or _RANK != _WRITER_RANK
            or _FORWARD_KIND != "conditional"
            or (_SAMPLER_STEPS is not None and _SAMPLER_STEP not in _SAMPLER_STEPS)
        ):
            return
        if _LAYOUT_THW is None:
            raise RuntimeError("endpoint_ucpe capture requires layout_thw")
        expected_tokens = int(np.prod(_LAYOUT_THW))
        t = tensor.detach()
        if hasattr(t, "to_local"):
            t = t.to_local()
        for j, gid in enumerate(_SAMPLE_GIDS):
            start, end = _SEQLENS[j], _SEQLENS[j + 1]
            seg = t[start:end]
            if seg.shape[0] != expected_tokens:
                raise ValueError(
                    f"{gid}: token count {seg.shape[0]} does not match layout "
                    f"{_LAYOUT_THW} ({expected_tokens})"
                )
            shaped = seg.reshape(*_LAYOUT_THW, seg.shape[-1]).float()
            vector = shaped[-1].mean(dim=(0, 1)) - shaped[0].mean(dim=(0, 1))
            step_entry = _BUFFER.setdefault(gid, {}).setdefault(
                int(_SAMPLER_STEP), {"actual_timestep": float(_TIMESTEP), "blocks": {}}
            )
            step_entry["blocks"][int(block_idx)] = vector.cpu()
        return

    t = tensor.detach()
    if hasattr(t, "to_local"):          # defensive: unwrap a DTensor if ever sharded
        t = t.to_local()

    n_samples = len(_SEQLENS) - 1
    for j in range(n_samples):
        if j >= len(_SAMPLE_GIDS):
            break
        start, end = _SEQLENS[j], _SEQLENS[j + 1]
        n_tokens = end - start
        if n_tokens <= 0:
            continue
        gid = _SAMPLE_GIDS[j]
        seg = t[start:end]                       # (T_j, D) on device
        mean = seg.float().mean(dim=0)           # (D,)
        idx = _tok_indices(gid, block_idx, n_tokens)
        sel = torch.as_tensor(idx, device=seg.device, dtype=torch.long)
        tok = seg.index_select(0, sel).float()   # (K, D)

        entry = _BUFFER.setdefault(gid, {}).setdefault(int(block_idx), {})
        entry[f"{prefix}_tok"] = tok.to(torch.float16).cpu().numpy()
        entry[f"{prefix}_mean"] = mean.to(torch.float16).cpu().numpy()
        entry["_n_tokens"] = int(n_tokens)


def flush():
    """Write one npz per (sample, timestep) and clear the buffer."""
    if not _ENABLED or not _BUFFER:
        return
    if _CAPTURE == "full_ucpe_only":
        _flush_full_ucpe()
        return
    if _CAPTURE == "endpoint_ucpe":
        _flush_endpoint_ucpe()
        return

    ts_tag = f"{_TIMESTEP:.2f}" if _TIMESTEP is not None else "na"
    raw_dir = os.path.join(_OUT_DIR, "raw")
    index_path = os.path.join(raw_dir, f"index_rank{_RANK}.jsonl")

    for j, gid in enumerate(_SAMPLE_GIDS):
        blocks = _BUFFER.get(gid)
        if not blocks:
            continue
        arrays = {}
        block_ids = sorted(blocks.keys())
        n_tokens = 0
        for b in block_ids:
            for key, val in blocks[b].items():
                if key == "_n_tokens":
                    n_tokens = val
                    continue
                arrays[f"{key.split('_')[0]}_b{b}_{key.split('_')[1]}"] = val
        caption = _CAPTIONS[j] if _CAPTIONS and j < len(_CAPTIONS) else ""
        arrays["block_ids"] = np.asarray(block_ids, dtype=np.int32)
        arrays["timestep"] = np.asarray(_TIMESTEP if _TIMESTEP is not None else -1.0, dtype=np.float32)
        arrays["caption"] = np.asarray(caption)
        arrays["gid"] = np.asarray(gid)
        arrays["n_tokens"] = np.asarray(n_tokens, dtype=np.int64)
        arrays["capture"] = np.asarray(_CAPTURE)

        out_path = os.path.join(raw_dir, f"{gid}_t{ts_tag}.npz")
        np.savez_compressed(out_path, **arrays)
        with open(index_path, "a") as fh:
            fh.write(json.dumps({
                "gid": gid, "timestep": _TIMESTEP, "caption": caption,
                "n_tokens": n_tokens, "blocks": block_ids, "file": os.path.basename(out_path),
            }) + "\n")

    _BUFFER.clear()
    _TOK_IDX_CACHE.clear()


def _flush_full_ucpe():
    """Atomically write one torch payload per sample for sampler step zero."""
    raw_dir = os.path.join(_OUT_DIR, "raw")
    index_path = os.path.join(raw_dir, f"index_rank{_RANK}.jsonl")
    for j, gid in enumerate(_SAMPLE_GIDS or []):
        blocks = _BUFFER.get(gid)
        if not blocks:
            continue
        block_ids = sorted(blocks)
        payload = dict(_METADATA or {})
        payload.update({
            "format_version": 1,
            "gid": gid,
            "caption": _CAPTIONS[j] if _CAPTIONS and j < len(_CAPTIONS) else "",
            "capture": "full_ucpe_only",
            "sampler_step": int(_CAPTURED_STEP),
            "actual_timestep": float(_CAPTURED_TIMESTEP),
            "layout_thw": tuple(_LAYOUT_THW),
            "model_dim": int(blocks[block_ids[0]].shape[-1]),
            "block_ids": block_ids,
        })
        payload.update({f"ucpe_b{block}": blocks[block] for block in block_ids})
        out_path = os.path.join(raw_dir, f"{gid}.pt")
        fd, tmp_path = tempfile.mkstemp(prefix=f".{gid}.", suffix=".tmp", dir=raw_dir)
        os.close(fd)
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, out_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        with open(index_path, "a") as fh:
            fh.write(json.dumps({
                "gid": gid,
                "timestep": _CAPTURED_TIMESTEP,
                "sampler_step": _CAPTURED_STEP,
                "layout_thw": list(_LAYOUT_THW),
                "blocks": block_ids,
                "file": os.path.basename(out_path),
            }) + "\n")
    _BUFFER.clear()
    _TOK_IDX_CACHE.clear()


def _flush_endpoint_ucpe():
    """Atomically write endpoint vectors for all selected sampler steps."""
    raw_dir = os.path.join(_OUT_DIR, "raw")
    index_path = os.path.join(raw_dir, f"index_rank{_RANK}.jsonl")
    for j, gid in enumerate(_SAMPLE_GIDS or []):
        steps = _BUFFER.get(gid)
        if not steps:
            continue
        sampler_steps = sorted(steps)
        block_ids = sorted(set.intersection(*(
            set(steps[step]["blocks"]) for step in sampler_steps
        )))
        if not block_ids:
            raise RuntimeError(f"{gid}: endpoint capture contains no common blocks")
        payload = dict(_METADATA or {})
        payload.update({
            "format_version": 1,
            "gid": gid,
            "caption": _CAPTIONS[j] if _CAPTIONS and j < len(_CAPTIONS) else "",
            "capture": "endpoint_ucpe",
            "sampler_steps": sampler_steps,
            "actual_timesteps": [float(steps[step]["actual_timestep"]) for step in sampler_steps],
            "model_dim": int(steps[sampler_steps[0]]["blocks"][block_ids[0]].shape[-1]),
            "block_ids": block_ids,
        })
        for step in sampler_steps:
            payload.update({
                f"ucpe_b{block}_s{step}": steps[step]["blocks"][block]
                for block in block_ids
            })
        out_path = os.path.join(raw_dir, f"{gid}.pt")
        fd, tmp_path = tempfile.mkstemp(prefix=f".{gid}.", suffix=".tmp", dir=raw_dir)
        os.close(fd)
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, out_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        with open(index_path, "a") as fh:
            fh.write(json.dumps({
                "gid": gid,
                "sampler_steps": sampler_steps,
                "actual_timesteps": payload["actual_timesteps"],
                "blocks": block_ids,
                "file": os.path.basename(out_path),
            }) + "\n")
    _BUFFER.clear()
    _TOK_IDX_CACHE.clear()
