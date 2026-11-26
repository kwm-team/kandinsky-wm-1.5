"""Replay AR denoising reads with static buffers and live causal history."""

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .cache import StreamCache


def static_cache(source):
    cache = StreamCache(source.layers)
    cache.token_axis = source.token_axis
    for layer, pair in enumerate(source.placed):
        if pair is not None:
            cache.place(layer, pair[0].clone(), pair[1].clone())
    return cache


def copy_placements(source, target):
    for src, dst in zip(source.placed, target.placed, strict=True):
        if (src is None) != (dst is None):
            raise RuntimeError("The AR graph's causal cache layout changed")
        if src is not None:
            for to, value in zip(dst, src, strict=True):
                to.copy_(value)


@dataclass
class ReadBuffers:
    value: torch.Tensor
    level: torch.Tensor
    result: torch.Tensor
    positions: torch.Tensor
    geometry: tuple[torch.Tensor, torch.Tensor]
    visual: StreamCache
    camera: StreamCache
    owner: object
    graph: torch.cuda.CUDAGraph


class GraphReader:
    def __init__(self, stream, device):
        self.stream = stream
        self.eager = stream.call
        self.bundles = {}
        self.text = None
        self.capture_stream = torch.cuda.Stream(device=device)

    def close(self):
        # NCCL retains each captured collective until its graph is destroyed.
        for buffers in self.bundles.values():
            buffers.graph.reset()
        self.bundles.clear()

    @contextmanager
    def _bind(self, buffers):
        values = dict(visual={0: buffers.visual}, ucpe={0: buffers.camera},
                      current_geometry=buffers.geometry, context_geometry=None,
                      context_rope=None, positions=buffers.positions, functions={},
                      calls={"read": 0, "commit": 0})
        previous = {name: getattr(self.stream, name) for name in values}
        for name, value in values.items():
            setattr(self.stream, name, value)
        try:
            yield
        finally:
            for name, value in previous.items():
                setattr(self.stream, name, value)

    def _capture(self, value, level, text, use_ucpe):
        native = self.stream
        native.place_context(0, use_ucpe)
        camera = static_cache(native.ucpe[0]) if use_ucpe else StreamCache(len(native.blocks))
        buffers = ReadBuffers(value.clone(), level.clone(), None, native.positions.clone(),
                              tuple(part.clone() for part in native.current_geometry),
                              static_cache(native.visual[0]), camera, native.context_owner,
                              torch.cuda.CUDAGraph())

        current = torch.cuda.current_stream(value.device)
        self.capture_stream.wait_stream(current)
        with torch.cuda.stream(self.capture_stream), self._bind(buffers):
            warm = self.eager(buffers.value, buffers.level, text, use_ucpe=use_ucpe)
            buffers.result = torch.empty_like(warm)
            self.eager(buffers.value, buffers.level, text, use_ucpe=use_ucpe)
        current.wait_stream(self.capture_stream)
        # Resets and the camera gate can replay different layouts in a different
        # order from capture. Each graph therefore owns its scratch pool.
        with torch.cuda.graph(buffers.graph, stream=self.capture_stream), self._bind(buffers):
            buffers.result.copy_(self.eager(buffers.value, buffers.level, text, use_ucpe=use_ucpe))
        return buffers

    def __call__(self, value, level, text, slot=0, mode="read", anchor=False, *, differentiable=False, use_ucpe=True):
        if slot or mode != "read" or anchor or differentiable:
            return self.eager(value, level, text, slot, mode, anchor, differentiable=differentiable, use_ucpe=use_ucpe)
        if self.text is not text:
            self.close()
            self.text = text
        native = self.stream
        key = (len(native.ids), tuple(value.shape), use_ucpe)
        if key not in self.bundles:
            self.bundles[key] = self._capture(value, level, text, use_ucpe)
        buffers = self.bundles[key]
        if buffers.owner is not native.context_owner:
            native.place_context(0, use_ucpe)
            copy_placements(native.visual[0], buffers.visual)
            if use_ucpe:
                copy_placements(native.ucpe[0], buffers.camera)
            for to, data in zip(buffers.geometry, native.current_geometry, strict=True):
                to.copy_(data)
            buffers.positions.copy_(native.positions)
            buffers.owner = native.context_owner
        buffers.value.copy_(value)
        buffers.level.copy_(level)
        buffers.graph.replay()
        native.calls["read"] += 1
        return buffers.result
