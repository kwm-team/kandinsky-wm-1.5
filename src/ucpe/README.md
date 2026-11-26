# AV UCPE

Generate an autonomous-driving scene from a text prompt and a sequence of
calibrated camera poses. Start from a still image, continue an observed video
prefix, or generate from the prompt and trajectory alone.

## Environment

The reference environment uses Python 3.12, PyTorch 2.7.0 with CUDA 12.8,
FlashAttention 2.7.4.post1 and the FlashAttention 3 Hopper extension.
Install from the repository root with `uv` on a machine with a CUDA toolkit and a
compiler compatible with your PyTorch build.

```sh
uv sync --locked --python 3.12 --extra hub --extra ucpe-ar --group flash-build
MAX_JOBS=4 uv pip install --no-build-isolation flash-attn==2.7.4.post1
MAX_JOBS=4 FLASH_ATTENTION_DISABLE_BACKWARD=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE \
  uv pip install --no-build-isolation \
  'git+https://github.com/Dao-AILab/flash-attention.git@7b0bfcc3d1f69786f0c4277c582ad58acdfb297d#subdirectory=hopper'
source .venv/bin/activate
```

The denoiser uses FlashAttention 3 when the extension is available. The `ucpe-ar` extra selects the locked Hugging Face and PyAV versions used for UCPE and AR.

## Run

```sh
python scripts/infer_ucpe.py --weights-dir weights/ucpe \
  --image examples/reference.jpg --prompt-file examples/prompt.txt \
  --trajectory examples/left.json --deterministic --output outputs/ucpe-left.mp4
```

The still image occupies a separate clean reference slot with the pose at time
zero. Pass `--video observed.mp4` in place of `--image` to continue a video.
The runner encodes the first five RGB frames at 24 fps and retains their
reconstructed prefix in the output. The trajectory starts at the first input
frame and must span at least nine output frames. For text-to-video, omit both
reference flags and provide calibration in output-image pixel coordinates.

The default render size is 768×512 at 24 fps. Set `--seed` to reproduce a run
with the same parameters and runtime. Results may vary across GPU architectures
and library versions. The bundle contains `model.safetensors` and
`config.json`, together with the matching VAE, Qwen text encoder, processor,
and CLIP assets. The denoiser file contains the complete camera-conditioned
model.

## Trajectory file

Pass a standalone JSON or NPZ file. Both formats use the following field names.

| Field | Shape and meaning |
| --- | --- |
| `camera_to_world` | `[N, 4, 4]` rigid transforms. The upper-left block is rotation, and the final column is translation in metres. Camera axes follow OpenCV with positive x to the right, positive y down and positive z forward. Matrices transform camera-space column vectors to a common world frame. |
| `intrinsics` | `[3, 3]` or `[N, 3, 3]` pinhole calibration matrices in input image or video pixel coordinates, or in output pixels for text-to-video. A constant or per-frame `[fx, fy, cx, cy]` is also accepted. |
| `frame_indices` | `[N]` increasing integer offsets beginning at zero. Omit for one pose per video frame. The final index plus one is the output length and must have the form 4k+1 with 1 ≤ k ≤ 30. |
| `fps` | Scalar frame rate of 24. Defaults to 24. |
| `image_size` | Optional `[width, height]` of the calibrated image plane. If present, it must match the input image or video dimensions, or the render dimensions for text-to-video. |

The runner interpolates translation and calibration linearly and rotation with
spherical interpolation on the VAE's temporal grid. It expresses the trajectory
relative to the first camera and applies the checkpoint's translation transform.
For a portrait or differently sized reference, express the provided focal lengths
and principal point against that reference's dimensions.

Use the same image, prompt and seed with `examples/straight.json` and
`examples/left.json` to compare camera paths.

## Code layout

- `pipeline.py` loads the weights and encodes text and visual references.
- `trajectory.py` validates and interpolates the standalone camera track.
- `sampling.py` runs Euler sampling and packs clean references by input mode.
- `runtime/` contains the camera-conditioned denoiser, text encoder and VAE.
- `export.py` assembles dense inference weights from sharded base and EMA weights.

## Verification

Run the CPU tests with `python -m unittest discover -s tests/ucpe -v`.
Add `--deterministic` to the CLI to use eager execution and deterministic
cuDNN for reference comparisons. With these settings on an H100, the
packaged pipeline reproduced the native sampler's VAE input latents and
decoded RGB exactly for image-to-video at 768×512 over 121 frames, and
for text-to-video and video-to-video at 384×256 over nine frames. The
comparisons used seed 42 and the default 50-point schedule.
