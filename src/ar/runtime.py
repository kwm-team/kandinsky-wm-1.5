"""Video autoregression with a fixed startup image and causal rolling history."""

import types

import torch
import torch.distributed as dist

from .model import SolverSettings, StreamSettings
from .rope import first_tail, retain_first_tail
from .solver import (
    PiFlowPolicy,
    apply_rollout_coefficients,
    build_solver_segments,
    precompute_rollout_coefficients,
    rollout_policy,
)
from .stream import NativeStream


def install_single_caption_attention(model, backend: str) -> None:
    """The model's one-caption inference path uses the same FA3 dispatch as causal attention."""
    from .native.model.attention import attention
    from .native.model.nn import MultiheadCrossAttention, MultiheadSelfAttention

    def attend(_module, q, k, v, _cu, *_args, **_kwargs):
        output = attention(
            q.transpose(0, 1)[None],
            k.transpose(0, 1)[None],
            v.transpose(0, 1)[None],
            backend,
        )
        return output[0].transpose(0, 1).flatten(1)

    for module in model.modules():
        if isinstance(module, (MultiheadSelfAttention, MultiheadCrossAttention)):
            module.scaled_dot_product_attention = types.MethodType(attend, module)


def active_camera_levels(sigmas, gate):
    """Preserve levels that activate the camera gate in either inference dtype."""
    if gate is None:
        return (True,) * len(sigmas)
    low, high = gate
    time = sigmas.cpu() * 1000
    active = torch.zeros_like(time, dtype=torch.bool)
    for dtype in (torch.float32, torch.bfloat16):
        clean = 1 - time.to(dtype) / 1000
        active |= (clean >= low) & (clean <= high)
    return tuple(active.tolist())


class ARRuntime:
    """Own one transformer and its mutable state for a single video stream."""

    def __init__(
        self,
        model,
        startup: torch.Tensor,
        startup_pose: torch.Tensor,
        startup_intrinsics: torch.Tensor,
        text: tuple[torch.Tensor, torch.Tensor],
        solver: SolverSettings,
        settings: StreamSettings,
        *,
        seed: int,
        backend: str = "flash3",
        final_model=None,
        process_group=None,
        linear_rollout=False,
    ):
        if startup.ndim != 4 or startup.shape[0] != 1 or startup.shape[-1] != 16:
            raise ValueError(
                "Expected an independently encoded [1,H,W,16] startup latent"
            )
        if not torch.isfinite(startup).all():
            raise ValueError("Startup latent must be finite")
        self.startup = startup.float()
        self.device = startup.device
        self.solver = solver
        self.settings = settings
        self.ucpe_scale = torch.tensor(
            settings.ucpe_scale, device=self.device, dtype=torch.float32
        )
        self.seed = seed
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self.group = process_group if process_group is not None else dist.group.WORLD
        self.distributed = dist.is_initialized() and dist.get_world_size(self.group) > 1
        self.text = self.prepare_text(text)
        self.schedule = build_solver_segments(
            solver.nfe,
            solver.final_step_size_scale,
            solver.shift,
            solver.eps,
            self.device,
        )
        self.rollout_weights = (
            precompute_rollout_coefficients(self.schedule, solver) if linear_rollout else None
        )
        gated_camera = linear_rollout and not any(
            getattr(model, key, False) for key in ("use_raymap", "use_crepe", "use_rayrope")
        )
        self.read_camera_levels = active_camera_levels(
            self.schedule[2], getattr(model, "ucpe_t", None) if gated_camera else None
        )

        capacity = 128
        poses = torch.eye(4, device=self.device, dtype=torch.float64).repeat(
            capacity, 1, 1
        )
        intrinsics = startup_intrinsics.to(device=self.device, dtype=torch.float32)[
            None
        ].repeat(capacity, 1, 1)
        poses[0] = startup_pose.to(device=self.device, dtype=torch.float32).double()
        stream_class = NativeStream
        stream_kwargs = {}
        if self.distributed:
            from .parallel import ParallelNativeStream

            stream_class = ParallelNativeStream
            stream_kwargs["group"] = self.group

        def make_stream(transformer):
            install_single_caption_attention(transformer, backend)
            for block in transformer.visual_transformer_blocks:
                if hasattr(block, "ucpe_branch"):
                    block.ucpe_branch.register_forward_hook(self.scale_ucpe)
            stream = stream_class(
                transformer,
                history=settings.history + 1,
                backend=backend,
                poses=poses,
                intrinsics=intrinsics,
                **stream_kwargs,
            )
            stream.infr_rope = True
            stream.single_timestep = linear_rollout
            return stream

        self.stream = make_stream(model)
        self.final_stream = (
            make_stream(final_model) if final_model is not None else None
        )
        self.cursor = 1
        with (
            torch.inference_mode(),
            torch.autocast(self.device.type, dtype=torch.bfloat16),
        ):
            self.stream.begin(
                [0], *self.startup.shape[1:3], self.device, advance_gauge=False
            )
            self.stream.call(self.startup, 0.0, self.text, mode="commit", anchor=True)
            self.stream.committed(anchor=True)
        self._ready = True

    def prepare_text(self, text):
        tokens, pooled = [
            tensor.to(self.device, dtype=torch.float32) for tensor in text
        ]
        if (
            tokens.ndim != 2
            or tokens.shape[1] != 3584
            or len(tokens) == 0
            or len(tokens) > 256
            or pooled.shape not in ((768,), (1, 768))
            or not torch.isfinite(tokens).all()
            or not torch.isfinite(pooled).all()
        ):
            raise ValueError(
                "Expected up to 256 finite Qwen [L,3584] tokens and a CLIP [768] vector"
            )
        return {
            "text": tokens,
            "pooled": pooled.reshape(1, 768),
            "cu": torch.tensor([0, len(tokens)], device=self.device, dtype=torch.int32),
            "rope": torch.arange(len(tokens), device=self.device),
        }

    def scale_ucpe(self, _module, _arguments, output):
        return output * self.ucpe_scale

    def compact_poses(self):
        """Reuse camera storage slots without changing cached KV ordering or RoPE positions."""
        stream = self.stream
        ids = stream.ids
        poses = stream.poses[ids].clone()
        intrinsics = stream.intrinsics[ids].clone()
        stream.poses[0] = stream.poses[stream.gauge_id].clone()
        stream.intrinsics[0] = stream.intrinsics[stream.gauge_id].clone()
        stream.poses[1 : len(ids) + 1] = poses
        stream.intrinsics[1 : len(ids) + 1] = intrinsics
        stream.ids = list(range(1, len(ids) + 1))
        stream.gauge_id = 0
        self.cursor = len(ids) + 1

    def retain_history(self):
        stream = self.stream
        keep = min(self.settings.retain_after_reset, len(stream.ids) - 1) + 1
        stream.ids = first_tail(stream.ids, keep)
        retain_first_tail(
            [*stream.visual.values(), *stream.ucpe.values()], stream.spatial, keep
        )

    def initial_noise(self):
        shape = (self.solver.chunk_planes, *self.startup.shape[1:])
        if not self.distributed or dist.get_rank(self.group) == 0:
            value = torch.randn(
                shape,
                device=self.device,
                generator=self.generator,
                dtype=self.startup.dtype,
            )
        else:
            value = torch.empty(shape, device=self.device, dtype=self.startup.dtype)
        if self.distributed:
            dist.broadcast(value, src=dist.get_global_rank(self.group, 0), group=self.group)
        return value

    def borrow_final_context(self):
        for name in (
            "ids",
            "current_ids",
            "visual",
            "ucpe",
            "positions",
            "context_positions",
            "context_rope",
            "current_geometry",
            "context_geometry",
            "context_owner",
            "spatial",
            "h",
            "w",
            "device",
        ):
            setattr(self.final_stream, name, getattr(self.stream, name))
        self.final_stream.functions = {}

    @torch.inference_mode()
    def generate(
        self, poses, intrinsics, accepted_planes: int, *, reset_after: bool = False
    ):
        """Generate a full chunk and commit the visible prefix exactly once."""
        planes = self.solver.chunk_planes
        if poses.shape != (planes, 4, 4) or intrinsics.shape != (planes, 3, 3):
            raise ValueError(
                "AR requires four candidate pose and intrinsics samples per chunk"
            )
        if not 1 <= accepted_planes <= planes:
            raise ValueError("Accepted length must be within the current chunk")
        if not self._ready:
            raise RuntimeError(
                "The AR stream has an active or failed generation call"
            )
        # A failed call may have advanced the RNG or partially committed the cache.
        self._ready = False
        stream = self.stream
        if self.cursor + planes >= len(stream.poses):
            self.compact_poses()
        ids = list(range(self.cursor, self.cursor + planes))
        stream.poses[ids] = poses.to(stream.poses)
        stream.intrinsics[ids] = intrinsics.to(stream.intrinsics)
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            stream.begin(ids, *self.startup.shape[1:3], self.device)
            value = self.initial_noise()
            for index, (src, dst, sigma, _, span) in enumerate(zip(*self.schedule), 1):
                reader = stream
                if self.final_stream is not None and index == self.solver.nfe:
                    self.borrow_final_context()
                    reader = self.final_stream
                grid = reader.call(
                    value, sigma, self.text, use_ucpe=self.read_camera_levels[index - 1]
                ).float().reshape(planes, self.solver.grid_points, -1)
                value = value.reshape(planes, -1)
                if self.rollout_weights is not None:
                    value = apply_rollout_coefficients(value, grid, self.rollout_weights[index - 1])
                else:
                    policy = PiFlowPolicy(
                        grid, value, sigma.expand(planes), span.expand(planes),
                        self.solver.shift, self.solver.eps,
                    )
                    value, _ = rollout_policy(
                        value, sigma.expand(planes, 1), src.expand(planes), dst.expand(planes),
                        self.solver.total_substeps, policy,
                    )
                value = value.reshape(planes, *self.startup.shape[1:])
            if not torch.isfinite(value).all():
                raise FloatingPointError("AR produced non-finite latents")
            if accepted_planes < planes:
                stream.begin(
                    ids[:accepted_planes],
                    *self.startup.shape[1:3],
                    self.device,
                    advance_gauge=False,
                )
            stream.call(value[:accepted_planes], 0.0, self.text, mode="commit")
            stream.committed()
        self.cursor += accepted_planes
        if reset_after:
            self.retain_history()
        self._ready = True
        return value
