"""Command-line entry point for native Base AV inference."""
import argparse
import math
import os
from pathlib import Path

from .inputs import Job, find_source, load_jobs
from .temporal import TEMPORAL_OUTPUT_FRAMES, read_temporal_prompt

FPS = 24
NEGATIVE_PROMPT = (
    "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low "
    "quality, ugly, deformed, walking backwards"
)
COMPONENTS = (
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "text_encoder_2/config.json",
    "text_encoder_2/model.safetensors",
    "tokenizer/tokenizer_config.json",
    "tokenizer_2/tokenizer_config.json",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("i2v", "v2v"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True, help="Local Base AV weight bundle")
    media = parser.add_mutually_exclusive_group(required=True)
    media.add_argument("--image", type=Path)
    media.add_argument("--video", type=Path)
    media.add_argument("--input", type=Path, help="CSV or JSON batch manifest")
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt", help="Text prompt for a single input")
    prompt.add_argument("--prompt-segments", type=Path, help="JSON scene and five one-second captions")
    parser.add_argument("--temporal-text", nargs="?", const="hard", choices=("hard", "smooth"),
                        help="Route the temporal captions with hard or smooth transitions")
    parser.add_argument("--output", type=Path, help="Output MP4 for a single input")
    parser.add_argument("--image-dir", type=Path, help="Image root for a batch manifest")
    parser.add_argument("--video-dir", type=Path, help="Video root for a batch manifest")
    parser.add_argument("--output-dir", type=Path, help="Output root for a batch manifest")
    parser.add_argument("--seed", type=int, help="Override the default or manifest seed")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-frames", type=int, default=120, help="Total frames including any prefix")
    parser.add_argument("--prefix-frames", type=int, help="Observed frames for V2V, default 5")
    parser.add_argument("--visual-size", type=int, choices=(256, 512), default=512)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance-weight", type=float, default=5.0)
    parser.add_argument("--scheduler-scale", type=float, default=5.0)
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Resolve manifest entries and media paths without a GPU")
    args = parser.parse_args(argv)

    if (args.image and args.mode != "i2v") or (args.video and args.mode != "v2v"):
        parser.error("Use --image for I2V and --video for V2V")
    if args.prompt_segments:
        args.temporal_text = args.temporal_text or "hard"
    if args.temporal_text is not None:
        if args.mode != "i2v" or args.output_frames not in TEMPORAL_OUTPUT_FRAMES:
            parser.error("Temporal text requires I2V with 120 or 121 output frames")
        if args.input is not None and args.input.suffix.lower() != ".json":
            parser.error("Temporal batch inference requires a JSON manifest")
        if args.input is None and args.prompt_segments is None:
            parser.error("Temporal inference requires --prompt-segments with --image")
    if args.input is None:
        if args.prompt_segments is None and (args.prompt is None or not args.prompt.strip()):
            parser.error("A single image or video requires --prompt or --prompt-segments")
        if args.output is None or args.output.suffix.lower() != ".mp4":
            parser.error("A single image or video requires an MP4 path in --output")
        if args.output_dir or args.image_dir or args.video_dir or args.start or args.limit is not None:
            parser.error("Batch options require a manifest in --input")
    else:
        if args.prompt is not None or args.prompt_segments is not None or args.output is not None:
            parser.error("Prompt options and --output apply to a single image or video")
        if args.output_dir is None:
            parser.error("--input requires --output-dir")
    if args.start < 0 or (args.limit is not None and args.limit < 0):
        parser.error("--start and --limit must be nonnegative")
    if args.num_steps < 2 or not 1 <= args.output_frames <= 509:
        parser.error("--num-steps must be at least 2 and --output-frames must be in [1, 509]")
    if not math.isfinite(args.scheduler_scale) or args.scheduler_scale <= 0:
        parser.error("--scheduler-scale must be positive and finite")
    if not math.isfinite(args.guidance_weight):
        parser.error("--guidance-weight must be finite")
    if args.mode == "i2v" and args.prefix_frames is not None:
        parser.error("--prefix-frames applies to V2V")
    if args.mode == "v2v":
        if args.prefix_frames is None:
            args.prefix_frames = 5
        if not 5 <= args.prefix_frames <= 73 or (args.prefix_frames - 1) % 4:
            parser.error("--prefix-frames must be 5, 9, ..., 73")
        if args.prefix_frames >= args.output_frames:
            parser.error("V2V requires at least one output frame after the prefix")
    return args


def build_jobs(args):
    if args.input:
        return load_jobs(args)

    source = find_source([args.image or args.video])
    if args.output.resolve() == source.resolve():
        raise ValueError(f"Output would overwrite its conditioning media: {source}")
    seed = args.seed if args.seed is not None else 1137
    if args.prompt_segments is not None and args.output.resolve() == args.prompt_segments.resolve():
        raise ValueError("Output must not overwrite the prompt segments file")
    prompt = read_temporal_prompt(args.prompt_segments) if args.prompt_segments else args.prompt
    return [Job(source, prompt, seed, args.output)]


def run_inference(args, jobs):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from .media import complete_video, save_video
    from .pipeline import BaseAVPipeline

    for name in COMPONENTS:
        if not (args.model_dir / name).is_file():
            raise FileNotFoundError(f"Missing Base AV weight bundle component: {args.model_dir / name}")
    if args.temporal_text and not args.overwrite:
        for job in jobs:
            if job.output.exists():
                raise ValueError(f"Temporal output already exists. Choose a new path or use --overwrite: {job.output}")
    rank = int(os.getenv("RANK", "0"))
    world = int(os.getenv("WORLD_SIZE", "1"))
    device = torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    mesh = init_device_mesh("cuda", (world,)) if world > 1 else None
    try:
        if args.overwrite:
            pending = jobs
        else:
            payload = [
                [job for job in jobs if not complete_video(job.output, args.output_frames, fps=FPS)]
                if rank == 0 else None
            ]
            if mesh is not None:
                dist.broadcast_object_list(payload, src=0)
            pending = payload[0]
        if not pending:
            if rank == 0:
                print("All requested outputs are complete.")
            return

        pipeline = BaseAVPipeline(args.model_dir, device, mesh)
        negative = pipeline.encode_negative(args.negative_prompt, temporal=args.temporal_text is not None)
        # FSDP requires equal forward counts on every rank in the last batch.
        for offset in range(0, len(pending), world):
            index = offset + rank
            job = pending[index if index < len(pending) else offset]
            video = pipeline.generate(
                job.source, job.prompt, negative, mode=args.mode, prefix_frames=args.prefix_frames,
                visual_size=args.visual_size, output_frames=args.output_frames, seed=job.seed,
                num_steps=args.num_steps, scheduler_scale=args.scheduler_scale,
                guidance_weight=args.guidance_weight, temporal_policy=args.temporal_text,
            )
            if index < len(pending):
                save_video(video, job.output, args.output_frames, FPS)
                print(f"Saved {job.output}", flush=True)
    finally:
        if mesh is not None:
            dist.destroy_process_group()


def main(argv=None):
    args = parse_args(argv)
    try:
        jobs = build_jobs(args)
        if args.dry_run:
            print(f"{len(jobs)} {args.mode} jobs")
            for job in jobs[:3]:
                print(f"{job.source} -> {job.output}")
        elif jobs:
            run_inference(args, jobs)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
    return 0
