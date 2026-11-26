"""Per-layer causal key and value storage."""

import torch


class StreamCache:
    """Per-layer keys and values of a stream's committed latents, for one attention.

    Keys are stored after normalisation but *before* RoPE (visual attention) or PRoPE
    (the UCPE branch). Each read places them at the positions and in the camera gauge
    chosen for the chunk being generated, which is what keeps an arbitrarily long stream
    inside the trained range. The model clamps camera centres at 150 m from the gauge.

    Two spans:

      context    committed latents, oldest first: the start image or memory, then chunks.
                 Evicted oldest first, never recomputed.
      transient  the current chunk's references and sink, stored the same way. They are
                 visible to this chunk alone and replaced before the next.
    """

    def __init__(self, layers: int) -> None:
        if layers < 1:
            raise ValueError("A cache needs at least one layer")
        self.layers = layers
        self.context: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(layers)
        ]
        self.transient: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * layers
        self.token_axis = 0
        # Context + transient already placed (rotated, or PRoPE-transformed) for one chunk.
        # Every Euler step and the commit of a chunk read the same placement, so it is
        # computed on the chunk's first read and dropped whenever the cache changes.
        self.placed: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * layers
        self.placed_for = None
        # The placement followed by room for the chunk's own keys and values (join).
        self.joined: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * layers

    def _changed(self) -> None:
        self.placed, self.placed_for = [None] * self.layers, None
        self.joined = [None] * self.layers

    def place_for(self, owner) -> None:
        """Keep the placement only if it was made for `owner`, the chunk's context tuple."""
        if owner is None or owner is not self.placed_for:
            self._changed()
            self.placed_for = owner

    def place(
        self, layer: int, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.placed[layer] = (key, value)
        return key, value

    def join(
        self, layer: int, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The placed context followed by the current chunk's keys and values.

        Concatenating copies the whole context again in every layer on every step. Without
        autograd the first call of a chunk concatenates once and keeps the result; the
        placement becomes a view of its head, and each later call of the chunk writes only
        its own tokens into the tail. The tensors are the same either way.
        """
        placed_key, placed_value = self.placed[layer]
        axis, context, own = (
            self.token_axis,
            placed_key.shape[self.token_axis],
            key.shape[self.token_axis],
        )
        joined = self.joined[layer]
        if (
            joined is not None
            and not torch.is_grad_enabled()
            and joined[0].shape[axis] == context + own
            and joined[0].dtype == key.dtype
            and joined[1].dtype == value.dtype
        ):
            joined[0].narrow(axis, context, own).copy_(key)
            joined[1].narrow(axis, context, own).copy_(value)
            return joined
        keys, values = torch.cat((placed_key, key), dim=axis), torch.cat(
            (placed_value, value), dim=axis
        )
        if not torch.is_grad_enabled():
            self.joined[layer] = (keys, values)
            self.placed[layer] = (
                keys.narrow(axis, 0, context),
                values.narrow(axis, 0, context),
            )
        return keys, values

    def _changed_layer(self, layer: int) -> None:
        # Layers are independent: a commit or reference pass writing layer by layer must
        # not throw away the placement the layers after it are about to read.
        self.placed[layer], self.joined[layer] = None, None

    def append(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
        self.context[layer].append((key, value))
        self._changed_layer(layer)

    def set_transient(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
        self.transient[layer] = (key, value)
        self._changed_layer(layer)

    def clear_transient(self) -> None:
        self.transient = [None] * self.layers
        self._changed()

    def readable(self, layer: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Context then transient, concatenated along the token axis, or None if empty."""
        parts = list(self.context[layer]) + (
            [self.transient[layer]] if self.transient[layer] is not None else []
        )
        if not parts:
            return None
        return (
            torch.cat([key for key, _ in parts], dim=self.token_axis),
            torch.cat([value for _, value in parts], dim=self.token_axis),
        )

    def context_tokens(self, layer: int = 0) -> int:
        return sum(key.shape[self.token_axis] for key, _ in self.context[layer])

    def evict(self, keep_tokens: int) -> None:
        """Keep the newest `keep_tokens` context tokens in every layer."""
        for layer in range(self.layers):
            joined = self.readable_context(layer)
            if joined is None:
                continue
            key, value = joined
            total = key.shape[self.token_axis]
            if total > keep_tokens:
                key = key.narrow(
                    self.token_axis, total - keep_tokens, keep_tokens
                ).clone()
                value = value.narrow(
                    self.token_axis, total - keep_tokens, keep_tokens
                ).clone()
            self.context[layer] = [(key, value)]
        self._changed()

    def readable_context(self, layer: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        parts = self.context[layer]
        if not parts:
            return None
        return (
            torch.cat([key for key, _ in parts], dim=self.token_axis),
            torch.cat([value for _, value in parts], dim=self.token_axis),
        )
