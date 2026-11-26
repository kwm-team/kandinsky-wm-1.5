"""Generate a driving clip between two calibrated reference images."""

import argparse
import json
from os import link
from pathlib import Path
from tempfile import NamedTemporaryFile


def _save_new(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        write(temporary)
        link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Local F2LF bundle")
    parser.add_argument("--a", type=Path, required=True, help="First reference image")
    parser.add_argument("--b", type=Path, required=True, help="Last reference image")
    parser.add_argument("--camera", type=Path, required=True, help="JSON with metric anchor poses and intrinsics")
    parser.add_argument("--prompt-file", type=Path, required=True)
    scene = parser.add_mutually_exclusive_group(required=True)
    scene.add_argument("--scene", help="Short global scene caption")
    scene.add_argument("--scene-file", type=Path, help="UTF-8 global scene caption")
    parser.add_argument("--output", type=Path, required=True, help="New MP4 output path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offset-m", type=float, default=0, help="Signed lateral midpoint offset in metres")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic cuDNN for VAE decoding")
    parser.add_argument("--latents-output", type=Path, help="Save generated body latents as Safetensors")
    parser.add_argument("--validate-only", action="store_true", help="Check inputs without loading CUDA")
    args = parser.parse_args()

    from PIL import Image
    from .camera import HEIGHT, WIDTH, camera_path

    try:
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()
        caption = (args.scene_file.read_text(encoding="utf-8") if args.scene_file else args.scene).strip()
        if not 1 <= len(prompt.split()) <= 140 or not caption:
            raise ValueError("Provide a 1–140 word video prompt and a short scene caption")
        if not (args.weights / "config.json").is_file():
            raise ValueError("--weights must point to an F2LF bundle with config.json")
        path = camera_path(json.loads(args.camera.read_text(encoding="utf-8")), args.offset_m)
        for reference in (args.a, args.b):
            with Image.open(reference) as image:
                if image.size not in ((WIDTH, HEIGHT), (2 * WIDTH, 2 * HEIGHT)):
                    raise ValueError("Reference images must be 768×512 or aligned 1536×1024")
        if args.output.suffix.lower() != ".mp4" or args.output.exists():
            raise ValueError("--output must be a new .mp4 path")
        if args.latents_output and (args.latents_output.suffix != ".safetensors" or args.latents_output.exists()):
            raise ValueError("--latents-output must be a new .safetensors path")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if args.validate_only:
        print(f"Validated two anchors, {len(path.poses)} latent poses and a {path.translation_scale:.2f} m path")
        return

    from .kernels import configure_environment

    configure_environment()
    import imageio.v2 as imageio
    from .pipeline import F2LFPipeline

    video = F2LFPipeline(args.weights, args.device)(
        args.a, args.b, path, prompt, caption, args.seed, deterministic=args.deterministic,
    )
    _save_new(args.output, lambda filename: imageio.mimwrite(
        filename, video.frames.numpy(), fps=24, codec="libx264", quality=None, macro_block_size=1,
        ffmpeg_params=["-crf", "18", "-preset", "fast", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    ))
    if args.latents_output:
        from safetensors.torch import save_file

        _save_new(args.latents_output, lambda filename: save_file({"latents": video.latents.contiguous()}, filename))
    print(f"Wrote {len(video.frames)} frames to {args.output}")
