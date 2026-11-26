"""CLI for AV UCPE trajectory-conditioned generation."""

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--components-dir", type=Path)
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--image", type=Path, help="Still image for image-to-video")
    reference.add_argument("--video", type=Path, help="24 fps video prefix for video-to-video")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--guidance-weight", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use eager execution and deterministic cuDNN for reference comparisons",
    )
    args = parser.parse_args()

    if args.deterministic:
        os.environ["TORCH_COMPILE_DISABLE"] = "1"

    from PIL import Image

    from .media import read_video_prefix, write_mp4
    from .pipeline import UCPEPipeline, VIDEO_PREFIX_FRAMES

    if args.deterministic:
        import torch

        # Override the CUDA defaults initialized by the native VAE.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("medium")
    from .trajectory import Trajectory

    trajectory = Trajectory.from_file(args.trajectory)
    prompt_text = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8").strip()
    if args.output.suffix.lower() != ".mp4" or args.output.exists():
        parser.error("--output must be a new .mp4 path")
    image = None
    video = None
    if args.image is not None:
        with Image.open(args.image) as source:
            image = source.convert("RGB")
    elif args.video is not None:
        video = read_video_prefix(args.video, VIDEO_PREFIX_FRAMES, int(trajectory.fps))
    pipeline = UCPEPipeline(args.weights_dir, args.components_dir, args.device)
    frames = pipeline(
        image, prompt_text, trajectory, seed=args.seed, width=args.width, height=args.height,
        num_steps=args.num_steps, guidance_weight=args.guidance_weight, video=video,
    )
    write_mp4(args.output, frames, int(trajectory.fps))
    print(f"Wrote {len(frames)} frames to {args.output}")


if __name__ == "__main__":
    main()
