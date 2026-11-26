"""Pinned startup latent and rolling temporal context."""

import torch


def first_tail(values, keep):
    if keep < 1:
        raise ValueError("Pinned cache must retain its T1")
    return (
        values
        if len(values) <= keep
        else values[:1] + (values[-(keep - 1) :] if keep > 1 else [])
    )


def retain_first_tail(caches, spatial, keep):
    for cache in caches:
        rows = []
        for layer in range(cache.layers):
            pair = cache.readable_context(layer)
            if pair is None:
                rows.append([])
                continue
            axis = cache.token_axis
            count = pair[0].shape[axis]
            if count > keep * spatial:
                pair = tuple(
                    torch.cat(
                        (
                            x.narrow(axis, 0, spatial),
                            x.narrow(
                                axis, count - (keep - 1) * spatial, (keep - 1) * spatial
                            ),
                        ),
                        dim=axis,
                    )
                    for x in pair
                )
            rows.append([pair])
        cache.context = rows
        cache.clear_transient()
        cache._changed()
