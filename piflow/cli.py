"""Generate video from a startup picture and a calibrated trajectory file."""

import argparse
import os
from pathlib import Path
import tempfile
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=Path, required=True, help="Local PiFlow bundle"
    )
    parser.add_argument(
        "--image", type=Path, required=True, help="Calibrated RGB startup image"
    )
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="Caption for the generated video")
    prompt.add_argument("--prompt-file", type=Path, help="UTF-8 caption file")
    prompt.add_argument(
        "--text-features", type=Path, help="Safetensors with tokens and pooled"
    )
    parser.add_argument(
        "--text-weights",
        type=Path,
        help="Override the bundled Qwen/CLIP assets with a local text directory or UCPE bundle",
    )
    parser.add_argument(
        "--trajectory", type=Path, required=True, help="JSON camera trajectory"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output MP4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--frames", type=int, help="Output length as a positive multiple of four"
    )
    parser.add_argument(
        "--backend", choices=("flash3", "flash2", "sdpa"), default="flash3"
    )
    parser.add_argument("--decoder", choices=("turbo", "hunyuan"), default="turbo")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic cuDNN kernels for the VAE",
    )
    parser.add_argument(
        "--final-gan",
        action="store_true",
        help="Use the optional final-step transformer",
    )
    return parser.parse_args()


def load_text_features(args, device):
    if args.text_features is not None:
        from safetensors.torch import load_file

        tensors = load_file(args.text_features)
        return tensors["tokens"].to(device), tensors["pooled"].to(device)

    import json
    import torch

    from ucpe.pipeline import TextEncoder

    caption = (
        args.prompt_file.read_text(encoding="utf-8").strip()
        if args.prompt_file
        else args.prompt.strip()
    )
    if not caption:
        raise ValueError("Provide a non-empty caption")
    weights = args.text_weights or args.weights / "text"
    config = json.loads((weights / "config.json").read_text(encoding="utf-8"))
    if config.get("format") not in ("kwm-av-piflow-text", "kwm-av-ucpe"):
        raise ValueError("Expected PiFlow or UCPE text assets and config")
    encoder = TextEncoder({**config["qwen"], "max_length": 256}, weights, device)
    # The standalone stand encoder evaluates CLIP in full FP32, then caches both
    # features as FP16 and restores them in FP32 for the transformer.
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        return tuple(feature.to(torch.float16).float() for feature in encoder(caption))
    finally:
        torch.set_float32_matmul_precision(precision)


def main():
    args = parse_args()

    import av
    from PIL import Image
    import torch
    import torch.distributed as dist

    from .codec import OverlapDecoder, VideoCodec
    from .model import load_configuration, load_model
    from .runtime import PiFlowRuntime
    from .trajectory import load_trajectory, rollout_windows

    trajectory = load_trajectory(args.trajectory)
    frames = trajectory.frames if args.frames is None else args.frames
    if frames > trajectory.frames:
        raise ValueError("The trajectory does not cover the requested output")
    config = load_configuration(args.weights / "config.json")
    windows = list(
        rollout_windows(
            frames,
            chunk_planes=config.solver.chunk_planes,
            reset_frames=config.stream.reset_frames,
        )
    )
    # Keep the final model window's preview when more of the input path is available.
    if frames < trajectory.frames:
        from dataclasses import replace

        terminal = windows[-1]
        anchors = tuple(
            min(trajectory.frames, terminal.start + 4 * index)
            for index in range(1, config.solver.chunk_planes + 1)
        )
        windows[-1] = replace(terminal, anchors=anchors)

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    if trajectory.image_size is not None and image.size != trajectory.image_size:
        raise ValueError("Startup image size must match the trajectory calibration")
    if min(image.size) < 16 or any(value % 16 for value in image.size):
        raise ValueError("Image dimensions must be multiples of 16")

    if not torch.cuda.is_available():
        raise RuntimeError("PiFlow requires a CUDA device")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    try:
        with torch.inference_mode():
            text = load_text_features(args, device)
            codec = None
            startup = torch.empty(
                (1, image.height // 8, image.width // 8, 16), device=device
            )
            if rank == 0:
                codec = VideoCodec(
                    args.weights / "vae",
                    device,
                    turbo_weights=(
                        args.weights / "turbo_decoder.pt"
                        if args.decoder == "turbo"
                        else None
                    ),
                )
                if args.deterministic:
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                startup.copy_(codec.encode_startup(image))
            if world > 1:
                dist.broadcast(startup, src=0)
            # The standalone DiT worker uses full FP32 for non-autocast math.
            torch.set_float32_matmul_precision("highest")
            model = load_model(
                config.architecture,
                args.weights / "transformer.safetensors",
                config.solver.grid_points,
                device,
            )
            final = (
                load_model(
                    config.architecture,
                    args.weights / "final_transformer.safetensors",
                    config.solver.grid_points,
                    device,
                )
                if args.final_gan
                else None
            )
            runtime = PiFlowRuntime(
                model,
                startup,
                torch.from_numpy(trajectory.camera_to_world[0]),
                torch.from_numpy(trajectory.intrinsics[0]),
                text,
                config.solver,
                config.stream,
                seed=args.seed,
                backend=args.backend,
                final_model=final,
            )
            if rank == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                decoder = OverlapDecoder(codec, startup)
                temp = tempfile.NamedTemporaryFile(
                    suffix=".mp4",
                    dir=args.output.parent,
                    delete=False,
                )
                temporary_output = Path(temp.name)
                temp.close()
            else:
                temporary_output = None
            try:
                container = None
                if rank == 0:
                    container = av.open(str(temporary_output), mode="w")
                    stream = container.add_stream("libx264", rate=24)
                    stream.width, stream.height = image.size
                    stream.pix_fmt = "yuv420p"
                    stream.options = {"crf": "10"}
                try:
                    for window in windows:
                        started = time.perf_counter()
                        indices = list(window.anchors)
                        generated = runtime.generate(
                            torch.from_numpy(trajectory.camera_to_world[indices]),
                            torch.from_numpy(trajectory.intrinsics[indices]),
                            window.frames // 4,
                            reset_after=window.reset_after,
                        )
                        if rank == 0:
                            torch.cuda.synchronize(device)
                            elapsed = time.perf_counter() - started
                            print(
                                f"Generated frames {window.start + 1}–{window.start + window.frames} (DiT {elapsed:.2f}s)",
                                flush=True,
                            )
                            for rgb in decoder.decode(
                                generated, window.frames // 4
                            ).numpy():
                                for packet in stream.encode(
                                    av.VideoFrame.from_ndarray(rgb, format="rgb24")
                                ):
                                    container.mux(packet)
                    if rank == 0:
                        for packet in stream.encode():
                            container.mux(packet)
                finally:
                    if container is not None:
                        container.close()
                if temporary_output is not None:
                    temporary_output.replace(args.output)
            finally:
                if temporary_output is not None:
                    temporary_output.unlink(missing_ok=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
