# Base AV

Base AV generates driving video from an initial frame and a text prompt. It also continues a short observed video prefix. Describe the action and the scene in the prompt. The model infers motion from the visual input and the text.

## Environment

The reference environment uses Python 3.12, PyTorch 2.7, CUDA 12.8 and FlashAttention 2. Install from the repository root using `uv`. Building FlashAttention requires the matching CUDA toolkit and a C++ compiler.

```sh
uv sync --locked --python 3.12 --extra hub --extra base-av-f2lf --group flash-build
MAX_JOBS=4 uv pip install --no-build-isolation flash-attn==2.7.4.post1
source .venv/bin/activate
```

Run from the repository root. An H100 is used for the reference checks. Long clips require more GPU memory than short clips.

## Weight bundle

Pass the bundle's local root using `--model-dir`. Use the transformer, VAE, Qwen, and CLIP components distributed for **Base AV**.

```text
base-av/
├── transformer/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── text_encoder/       # Qwen configuration, index and weight shards
├── text_encoder_2/     # CLIP configuration and weights
├── tokenizer/         # Qwen tokenizer files
└── tokenizer_2/       # CLIP tokenizer files
```

## Start from an image

```sh
python scripts/infer_base_av.py --mode i2v --model-dir weights/base-av \
  --image first-frame.png \
  --prompt 'A dashcam approaches a green light and follows the road through the intersection.' \
  --seed 42 --output drive.mp4
```

## Change the prompt over time

For a five-second I2V clip at 24 FPS, supply a standalone JSON file with a `scene` string and exactly five `captions`. Describe the scene across the full clip in `scene`, and describe the action for each successive second in `captions`. The model conditions each segment on the scene and its caption.

`hard` assigns one caption to each segment and is the default for `--prompt-segments`. `smooth` blends adjacent captions across segment boundaries. Both accept 120 or 121 output frames. In both modes, visual attention spans the whole clip to carry motion and scene content across segments. [Example prompt segments](../../examples/prompt_segments.json)

```sh
python scripts/infer_base_av.py --mode i2v --model-dir weights/base-av \
  --image examples/reference.jpg --prompt-segments examples/prompt_segments.json \
  --temporal-text smooth --seed 42 --output drive-temporal.mp4
```

For a temporal batch, pass `--temporal-text hard` or `--temporal-text smooth` with `--input` and a JSON manifest. Each row includes the same `scene` and `captions` fields alongside `video_id` and `img_id`. Use `--overwrite` to rerun a temporal job at an existing output path.

The motorcycle example uses one image and four schedules to show how the prompt changes when the rider enters the frame. The example includes the exact captions and batch command. [Motorcycle timing gallery and reproduction](../../examples/base-av/temporal-motorcycle/README.md)

## Continue a video

```sh
python scripts/infer_base_av.py --mode v2v --model-dir weights/base-av \
  --video observed.mp4 --prefix-frames 9 \
  --prompt 'The vehicle leaves the intersection and continues along the same lane.' \
  --seed 42 --output continuation.mp4
```

`--prefix-frames` accepts 5, 9, 13, and so on up to 73. `--output-frames` is the total length, including the VAE-reconstructed prefix. Output defaults to 120 frames at 24 FPS. The default visual size chooses among 768×512, 512×512, and 512×768 based on the aspect ratio. Use `--visual-size 256` for a smaller test.

## Batch generation

Pass a JSON array or compatible CSV manifest as `--input` and choose an `--output-dir`. An I2V JSON row can provide `video_id`, `prompt`, and `img_id`. A V2V row can provide `video_id`, `prompt`, and `video_path`. Media paths are relative to the manifest by default. With `--image-dir` or `--video-dir`, provide an explicit media root.

```json
[
  {
    "video_id": "intersection",
    "prompt": "The dashcam proceeds through the intersection.",
    "img_id": "inputs/intersection.jpg"
  }
]
```

A Physical AI Bench CSV with `filename` and `prompt` columns is also supported. If an image path is omitted, images are looked up under `condition_image/` by video ID. A suffix of the form `__42` on the filename supplies a per-row seed. Otherwise the default is 1137. `--seed` overrides the seed for all rows.

Use `--dry-run` to list resolved inputs and outputs, and `--start` and `--limit` to select a batch slice. A completed output at the requested path is reused unless `--overwrite` is set. For a multi-GPU batch, launch the script with `torchrun --standalone --nproc-per-node=2 scripts/infer_base_av.py` and pass the same arguments.

## Code layout

- `cli.py` validates requests and runs single- or multi-GPU inference.
- `pipeline.py` runs reference encoding, text conditioning and native sampling.
- `reference.py` preprocesses conditioning media, and `media.py` encodes MP4s.
- `model/` contains checkpoint loading, the native video, text and VAE modules, and temporal cross-attention.
- `sampler.py` implements flow matching for image and video-prefix conditioning.

## Verification

Run argument, manifest and media tests from the repository root.

```sh
python -m unittest discover -s tests/base_av -v
```

For a GPU comparison, generate from the same weight bundle and compare decoded outputs with the same inputs, seed, resolution and sampler settings. Check image conditioning and video continuation separately. [Reproducing outputs](../../docs/reproducibility.md)
