# Reproduce the video gallery

Run the commands from the repository root. Install the environment for each model and download its weight bundle. The native clips are 768×512 at 24 fps. Super-resolution is applied in a separate pass, preserving each clip's frame count and frame rate. [Weight bundles](weights.md)

## Base AV and accidents

The seven clips use their PAI-Bench-G conditioning images and English prompts, 120 frames, seed 42, 50 Euler schedule points, shift 5 and guidance 5. The gallery batch uses eight H100 ranks. [Base AV setup](../src/base_av/README.md) · [Benchmark protocol](benchmark.md)

```bash
mkdir -p outputs/gallery
python - <<'PY'
import json
from pathlib import Path

scenes = (
    'intersection-collision', 'intersection-rollover', 'rainy-night',
    'highway-trailer-collision', 'motorcycle-overtake', 'pickup-and-suv', 'opening-car-door',
)
rows = []
for scene in scenes:
    folder = Path('examples/base-av', scene)
    if not folder.is_dir():
        folder = Path('examples/accidents', scene)
    rows.append({
        'video_id': scene,
        'img_id': str((folder / 'reference.jpg').resolve()),
        'prompt': (folder / 'prompt.txt').read_text(encoding='utf-8').strip(),
    })
Path('outputs/gallery/base-av.json').write_text(json.dumps(rows), encoding='utf-8')
PY

torchrun --standalone --nproc-per-node=8 scripts/infer_base_av.py --mode i2v \
  --model-dir weights/base-av --input outputs/gallery/base-av.json \
  --output-dir outputs/gallery/base-av --seed 42 --output-frames 120 \
  --visual-size 512 --num-steps 50 --scheduler-scale 5 --guidance-weight 5
```

The four motorcycle timing clips reuse the same image and seed with four hard temporal-text schedules. The labels are the requested times in those schedules. [Timing inputs and command](../examples/base-av/temporal-motorcycle/README.md#reproduce-the-clips)

On H100s, all seven native MP4s matched independent runs of the source implementation byte for byte with the same eight-rank settings. Against older benchmark exports, their decoded PSNR ranged from 25.07 to 31.32 dB while retaining the same scene progression. The four motorcycle timing MP4s matched their originals byte for byte.

## UCPE camera control

The four 121-frame renders share an image, prompt and seed. Each uses a separate trajectory file. [UCPE setup](../src/ucpe/README.md)

```bash
for path in left right straight stationary; do
  python scripts/infer_ucpe.py --weights-dir weights/ucpe \
    --image examples/reference.jpg --prompt-file examples/prompt.txt \
    --trajectory "examples/$path.json" --seed 42 --deterministic \
    --output "outputs/gallery/ucpe/$path.mp4"
done
```

All four native MP4s matched the original camera-control comparison byte for byte.

## F2LF endpoint control

The straight, left and right renders share the endpoint frames, captions, camera calibration and seed. The arc offsets are −3.5 m and +3.5 m. [F2LF inputs and command](../examples/f2lf/directions/README.md)

All three H100 renders matched the original sampled latents and the MP4s from the source decoder's deterministic replay. The recipe includes the two encoded reference images to keep their conditioning stable across VAE kernel selections.

## Video super-resolution

The gallery uses a pinned public Kandinsky 6 VSR 2.0 checkpoint on the newly generated native MP4s. It combines a 1.125× pre-resize with a learned 2× latent upscaler and two denoiser evaluations per tile. A 120-frame clip repeats the final frame for the VAE's 121-frame input grid and trims that frame after upscaling. The output is 1728×1152, with the original frame count and timestamps. [VSR 2.0 weights](https://huggingface.co/kandinskylab/Kandinsky-6.0-VSR-2.0-distilled2steps-Diffusers/tree/36929c88a2cecc6d436444def27a89c8af0a37e6)

For the SR pass, use seed 42 with the pinned VSR 2.0 weights and the [Base AV / F2LF environment](../src/base_av/README.md#environment). The MP4s linked from the main gallery contain the upscaled outputs.

The gallery WebPs use the FFmpeg 7.0.2 build bundled with `imageio-ffmpeg` 0.6.0 on Linux. Use `PREVIEW_FPS=24` for motorcycle timing and F2LF, or `PREVIEW_FPS=12` for Base AV, accidents and UCPE.

```bash
FFMPEG=$(python -c 'from imageio_ffmpeg import get_ffmpeg_exe; print(get_ffmpeg_exe())')
"$FFMPEG" -threads 2 -i "$SR_MP4" -filter_threads 1 -an \
  -vf "fps=$PREVIEW_FPS,scale=480:320:flags=lanczos" \
  -c:v libwebp_anim -preset picture -quality 95 -compression_level 6 -loop 0 "$PREVIEW_WEBP"
```

The published checksums cover all 18 native MP4s, upscaled MP4s and WebP previews. [Gallery checksums](gallery-checksums.csv)
