"""Drive the AR checkpoint from a local keyboard-controlled browser stream."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from .demo_control import Drive, Layout, local_device_index, read_rgb, resize_intrinsics, write_rgb


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True, help="Trajectory JSON supplying frame zero and camera intrinsics")
    parser.add_argument("--size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), help="Resize the startup image and calibration for faster generation")
    parser.add_argument("--reset-frames", type=int, help="Override the AR history reset interval in output frames")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    prompt.add_argument("--text-features", type=Path)
    parser.add_argument("--text-weights", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("flash3", "flash2", "sdpa"), default="flash3")
    parser.add_argument("--final-gan", action="store_true", help="Use the optional transformer on the last solver evaluation")
    parser.add_argument("--dit-ranks", type=int, help="Count of contiguous DiT ranks, beginning at rank zero")
    parser.add_argument("--dit-parallel", choices=("sp", "tp"), default="sp", help="Sequence or tensor parallel DiT")
    parser.add_argument("--dit-graphs", action="store_true", help="Replay CUDA graphs for TP denoising reads")
    parser.add_argument("--compile-dit", action="store_true", help="Compile the DiT's rotary, affine and residual kernels")
    parser.add_argument("--vae-rank", type=int, help="Global VAE and browser rank, the final rank by default")
    parser.add_argument("--vae-device", type=int, help="Local CUDA device for the VAE/browser process")
    parser.add_argument("--sr-rank", type=int, help="Global rank for optional SR, shared with the VAE by default")
    parser.add_argument("--sr-python", type=Path, help="Python in the separate Kandinsky SR environment")
    parser.add_argument("--sr-config", type=Path, help="Kandinsky video SR YAML with local or Hub weights")
    parser.add_argument("--sr-scale", type=int, choices=(2, 4), default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--frames", type=int, help="Stop a live session after this many generated frames")
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--benchmark-chunks", type=int, help="Measure this many chunks without display pacing")
    parser.add_argument("--warmup-chunks", type=int, default=3)
    args = parser.parse_args(argv)
    if (args.sr_config is None) != (args.sr_python is None):
        parser.error("--sr-config and --sr-python must be supplied together")
    if args.sr_rank is not None and args.sr_config is None:
        parser.error("--sr-rank requires --sr-config")
    if args.vae_device is not None and args.vae_device < 0:
        parser.error("--vae-device must be nonnegative")
    if args.dit_graphs and (args.dit_parallel != "tp" or args.backend != "flash3"):
        parser.error("--dit-graphs requires --dit-parallel tp and FlashAttention 3")
    if args.compile_dit and any(os.environ.get(name) == "1" for name in ("TORCH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE")):
        parser.error("Unset TORCH_COMPILE_DISABLE and TORCHDYNAMO_DISABLE to use --compile-dit")
    if not 1 <= args.jpeg_quality <= 95:
        parser.error("--jpeg-quality must be between 1 and 95")
    if args.warmup_chunks < 0 or (args.benchmark_chunks is not None and args.benchmark_chunks < 2):
        parser.error("A benchmark needs at least two measured chunks and nonnegative warmup")
    if args.frames is not None and (args.frames < 4 or args.frames % 4 or args.benchmark_chunks is not None):
        parser.error("--frames must be a positive multiple of four in live mode")
    if args.size is not None and (min(args.size) < 16 or any(size % 16 for size in args.size)):
        parser.error("--size dimensions must be positive multiples of 16")
    if args.reset_frames is not None and (args.reset_frames < 4 or args.reset_frames % 4):
        parser.error("--reset-frames must be a positive multiple of four")
    return args


class SuperResolution:
    def __init__(self, args, device):
        env = os.environ.copy()
        # SR needs its own compilation policy even when AR runs eagerly.
        env.pop("TORCH_COMPILE_DISABLE", None)
        self.process = subprocess.Popen(
            [
                str(args.sr_python), str(Path(__file__).with_name("demo_sr_worker.py")),
                str(args.sr_config), str(device), str(args.sr_scale),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
        )

    def apply(self, frames):
        try:
            write_rgb(self.process.stdin, frames)
            output = read_rgb(self.process.stdout)
        except (EOFError, BrokenPipeError) as err:
            raise RuntimeError("The Kandinsky SR process stopped") from err
        if output is None or len(output) != len(frames):
            raise RuntimeError("The Kandinsky SR process returned an invalid frame count")
        return output

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        finally:
            self.process.stdout.close()


def transfer_rgb(frames, target, group):
    import torch
    import torch.distributed as dist

    shape = frames.shape if frames is not None else (0, 0, 0, 3)
    dist.send(torch.tensor(shape[:3], dtype=torch.int64), dst=target, group=group)
    if frames is not None:
        dist.send(torch.from_numpy(frames.copy()), dst=target, group=group)


def receive_rgb(source, group):
    import torch
    import torch.distributed as dist

    size = torch.empty(3, dtype=torch.int64)
    dist.recv(size, src=source, group=group)
    frames, height, width = size.tolist()
    if frames == 0:
        return None
    rgb = torch.empty((frames, height, width, 3), dtype=torch.uint8)
    dist.recv(rgb, src=source, group=group)
    return rgb.numpy()


def prepare_inputs(args, image, layout, rank, device, distributed):
    import torch
    import torch.distributed as dist

    from .cli import load_text_features
    from .codec import VideoCodec

    startup = torch.empty((1, image.height // 8, image.width // 8, 16))
    tokens = torch.zeros((256, 3584))
    pooled = torch.empty(768)
    count = torch.zeros(1, dtype=torch.int64)
    codec = None
    if rank == layout.vae:
        features, caption = load_text_features(args, device)
        count.fill_(len(features))
        tokens[: len(features)].copy_(features)
        pooled.copy_(caption.reshape(768))
        del features, caption
        codec = VideoCodec(
            args.weights / "vae", device, turbo_weights=args.weights / "turbo_decoder.pt"
        )
        startup.copy_(codec.encode_startup(image, opt_tiling=args.size is None))
    if distributed:
        for value in (startup, count, tokens, pooled):
            dist.broadcast(value, src=layout.vae)
    return startup.to(device), (tokens[: int(count.item())].to(device), pooled.to(device)), codec


def load_dit(args, config, startup, trajectory, text, device, dit_group):
    import torch

    from .model import load_model
    from .runtime import ARRuntime

    torch.set_float32_matmul_precision("highest")
    dit = load_model(
        config.architecture,
        args.weights / "transformer.safetensors",
        config.solver.grid_points,
        device,
    )
    final = None
    if args.final_gan:
        final = load_model(
            config.architecture,
            args.weights / "final_transformer.safetensors",
            config.solver.grid_points,
            device,
        )
    if args.compile_dit:
        from .compile_ops import configure

        configure(dit)
        if final is not None:
            configure(final)
    return ARRuntime(
        dit, startup, torch.from_numpy(trajectory.camera_to_world[0]),
        torch.from_numpy(trajectory.intrinsics[0]), text, config.solver, config.stream,
        seed=args.seed, backend=args.backend, process_group=dit_group, final_model=final,
        linear_rollout=True, dit_parallel=args.dit_parallel, dit_graphs=args.dit_graphs,
    )


def dit_worker(runtime, config, intrinsics, layout, rank, device, dit_group):
    import torch
    import torch.distributed as dist

    planes = config.solver.chunk_planes
    host_message = torch.empty(planes * 16 + 2, dtype=torch.float64)
    message = host_message.to(device)
    leader = layout.dit[0]
    while True:
        if rank == leader:
            dist.recv(host_message, src=layout.vae)
            message.copy_(host_message)
        if len(layout.dit) > 1:
            dist.broadcast(message, src=leader, group=dit_group)
        accepted, reset = map(int, message[-2:].tolist())
        if not accepted:
            return
        generated = runtime.generate(message[:-2].view(planes, 4, 4), intrinsics, accepted, reset_after=bool(reset))
        if rank == leader:
            dist.send(generated.cpu().contiguous(), dst=layout.vae)


def publish_chunk(decoder, generated, window, super_resolve, quality, browser, device):
    import torch

    from .demo_web import encode_jpeg

    # The VAE worker has its own stream so receiving latents can overlap decoding.
    with torch.inference_mode(), torch.cuda.device(device), torch.cuda.stream(torch.cuda.Stream(device)):
        started = time.monotonic()
        frames = decoder.decode(generated, window.accepted_planes).numpy()
        decoded = time.monotonic()
        if super_resolve is not None:
            frames = super_resolve(frames)
        resolved = time.monotonic()
        jpegs = [encode_jpeg(rgb, quality) for rgb in frames]
        encoded = time.monotonic()
        if browser is not None:
            browser.enqueue(jpegs)
    return {
        "frames": len(jpegs),
        "vae_s": decoded - started,
        "sr_s": resolved - decoded,
        "jpeg_s": encoded - resolved,
        "finished_at": time.monotonic(),
    }


def controller(args, config, trajectory, image, startup, codec, layout, device, runtime, intrinsics, sr_group):
    import numpy as np
    import torch
    import torch.distributed as dist

    from .codec import OverlapDecoder
    from .demo_web import Browser, encode_jpeg

    benchmark = args.benchmark_chunks is not None
    chunks = args.benchmark_chunks + args.warmup_chunks if benchmark else None
    decoder = OverlapDecoder(codec, startup if runtime is not None else startup.cpu())
    planes = config.solver.chunk_planes
    remote_sr = layout.sr != layout.vae
    local_sr = SuperResolution(args, device) if args.sr_config and not remote_sr else None

    def enhance(frames):
        if remote_sr:
            transfer_rgb(frames, layout.sr, sr_group)
            return receive_rgb(layout.sr, sr_group)
        return local_sr.apply(frames)

    def generate(window):
        poses = torch.from_numpy(window.poses)
        if runtime is not None:
            generated = runtime.generate(poses, intrinsics, window.accepted_planes, reset_after=window.reset_after)
            torch.cuda.current_stream(device).synchronize()
            return generated
        signal = poses.new_tensor((window.accepted_planes, window.reset_after))
        dist.send(torch.cat((poses.flatten(), signal)), dst=layout.dit[0])
        result = torch.empty((planes, *startup.shape[1:]), dtype=startup.dtype)
        dist.recv(result, src=layout.dit[0])
        return result

    def finish(record):
        index, start, inference_time, result = record
        row = result.result() if hasattr(result, "result") else result
        row.update(chunk=index, dit_route_s=inference_time)
        print(json.dumps({key: value for key, value in row.items() if key != "finished_at"}), flush=True)
        if benchmark and index >= args.warmup_chunks:
            samples.append((start, row))

    drive = Drive(
        trajectory.camera_to_world[0],
        reset_frames=config.stream.reset_frames,
        chunk_planes=planes,
    )
    samples, pending = [], None
    try:
        browser_context = nullcontext(None) if benchmark else Browser(args.host, args.port)
        with browser_context as browser, ThreadPoolExecutor(max_workers=1) as worker:
            if browser is not None:
                browser.publish(encode_jpeg(np.asarray(image), args.jpeg_quality))
                print(f"Open http://{args.host}:{browser.server.server_port} on the VAE node", flush=True)
            index = 0
            try:
                while (chunks is None or index < chunks) and (args.frames is None or drive.frame < args.frames):
                    keys = {"forward"} if benchmark else browser.next_keys()
                    window = drive.next_window(keys, frame_limit=args.frames)
                    start = time.monotonic()
                    generated = generate(window)
                    inference_time = time.monotonic() - start
                    if pending is not None:
                        finish(pending)
                    inputs = (
                        decoder, generated, window, enhance if args.sr_config else None,
                        args.jpeg_quality, browser, device,
                    )
                    result = publish_chunk(*inputs) if runtime is not None else worker.submit(publish_chunk, *inputs)
                    pending = index, start, inference_time, result
                    index += 1
                if pending is not None:
                    finish(pending)
                    pending = None
                if browser is not None:
                    browser.drain()
            except KeyboardInterrupt:
                if pending is not None:
                    finish(pending)
    finally:
        if runtime is None:
            stop = torch.zeros(planes * 16 + 2, dtype=torch.float64)
            dist.send(stop, dst=layout.dit[0])
        if remote_sr:
            transfer_rgb(None, layout.sr, sr_group)
        if local_sr is not None:
            local_sr.close()
    if samples:
        n = sum(row["frames"] for _, row in samples)
        elapsed = samples[-1][1]["finished_at"] - samples[0][0]
        print(json.dumps({
            "benchmark": {
                "chunks": len(samples), "frames": n, "seconds": elapsed,
                "video_ready_fps": n / elapsed, "target_fps": 24,
                "dit_median_ms": 1000 * statistics.median(row["dit_route_s"] for _, row in samples),
                "native_width": image.width, "native_height": image.height,
                "reset_frames": config.stream.reset_frames,
                "dit_parallel": args.dit_parallel if len(layout.dit) > 1 else "single",
                "dit_ranks": len(layout.dit),
                "dit_graphs": args.dit_graphs,
                "compile_dit": args.compile_dit,
                "sr_scale": args.sr_scale if args.sr_config else None,
                "final_gan": args.final_gan,
            }
        }), flush=True)


def main(argv=None):
    args = parse_args(argv)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    layout = Layout.resolve(world, dit_ranks=args.dit_ranks, vae_rank=args.vae_rank, sr_rank=args.sr_rank)
    if args.compile_dit:
        if rank in layout.dit:
            os.environ["TRITON_DEFAULT_FP_FUSION"] = "0"
            os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        else:
            os.environ["TORCH_COMPILE_DISABLE"] = "1"
    import torch
    import torch.distributed as dist
    from PIL import Image

    from .model import load_configuration
    from .trajectory import load_trajectory

    if args.dit_graphs and len(layout.dit) < 2:
        raise ValueError("--dit-graphs requires at least two DiT ranks")
    if not torch.cuda.is_available():
        raise RuntimeError("The AR demo requires CUDA")
    index = local_device_index(layout, rank, int(os.environ.get("LOCAL_RANK", "0")), args.vae_device)
    if index >= torch.cuda.device_count():
        raise ValueError(f"Rank {rank} maps to CUDA device {index}, outside the visible devices")
    device = torch.device("cuda", index)
    config = load_configuration(args.weights / "config.json")
    if args.reset_frames is not None:
        config = replace(config, stream=replace(config.stream, reset_frames=args.reset_frames))
    trajectory = load_trajectory(args.calibration)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    if trajectory.image_size is not None and image.size != trajectory.image_size:
        raise ValueError("Startup image size must match the calibration")
    if args.size is not None:
        size = tuple(args.size)
        trajectory = replace(
            trajectory,
            intrinsics=resize_intrinsics(trajectory.intrinsics, image.size, size),
            image_size=size,
        )
        image = image.resize(size, Image.Resampling.LANCZOS)
    if min(image.size) < 16 or any(size % 16 for size in image.size):
        raise ValueError("Image dimensions must be multiples of 16")

    torch.cuda.set_device(device)
    if world > 1:
        # The small role-to-role transfers use TCP. NCCL only serves model
        # collectives within the actively sharded DiT ranks.
        dist.init_process_group("gloo")
    try:
        dit_backend = "nccl" if len(layout.dit) > 1 else "gloo"
        dit_group = dist.new_group(list(layout.dit), backend=dit_backend) if world > 1 else None
        sr_group = dist.new_group([layout.vae, layout.sr], backend="gloo") if layout.sr != layout.vae else None
        with torch.inference_mode():
            startup, text, codec = prepare_inputs(args, image, layout, rank, device, world > 1)
            intrinsics = torch.from_numpy(trajectory.intrinsics[0]).to(device).expand(config.solver.chunk_planes, -1, -1)
            runtime = load_dit(args, config, startup, trajectory, text, device, dit_group) if rank in layout.dit else None
            print(f"Rank {rank} on {torch.cuda.get_device_name(device)} ready", flush=True)
            if world > 1:
                dist.barrier()
            if rank == layout.vae:
                controller(args, config, trajectory, image, startup, codec, layout, device, runtime, intrinsics, sr_group)
            elif rank in layout.dit:
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        dit_worker(runtime, config, intrinsics, layout, rank, device, dit_group)
                finally:
                    runtime.close()
            elif rank == layout.sr:
                sr = SuperResolution(args, device)
                try:
                    while (frames := receive_rgb(layout.vae, sr_group)) is not None:
                        transfer_rgb(sr.apply(frames), layout.vae, sr_group)
                finally:
                    sr.close()
            if world > 1:
                dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
