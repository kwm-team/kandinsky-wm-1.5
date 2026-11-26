# AR · Autoregressive video

Generate an autoregressive driving video from a starting image, a caption, and a trajectory. AR uses eight transformer evaluations per chunk. It commits generated latents to a causal key/value cache and rolls that context forward as new poses arrive.

## Install

The reference environment uses Python 3.12.9, PyTorch 2.7.0, CUDA 12.8, and an NVIDIA H100. Install from the repository root using `uv`. Build FlashAttention with the matching CUDA toolkit and a C++ compiler.

```bash
uv sync --locked --python 3.12 --extra hub --extra ucpe-ar --group flash-build
MAX_JOBS=4 uv pip install --no-build-isolation flash-attn==2.7.4.post1
MAX_JOBS=4 FLASH_ATTENTION_DISABLE_BACKWARD=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE \
  uv pip install --no-build-isolation \
  'git+https://github.com/Dao-AILab/flash-attention.git@7b0bfcc3d1f69786f0c4277c582ad58acdfb297d#subdirectory=hopper'
source .venv/bin/activate
```

Qwen caption encoding uses FlashAttention 2. Install the Hopper extension to use the default `--backend flash3` for the video transformer. [FlashAttention build instructions](https://github.com/Dao-AILab/flash-attention#flashattention-3-beta-release).

## Drive with the keyboard

The browser demo uses this AR bundle with keyboard-generated camera poses. The one-node TP7 profile streams a 768×432 crop at 24 fps after warm-up on eight H100s, including a dedicated TurboVAE GPU. The guide includes two- and three-node launches with video super-resolution. [Keyboard demo and timings](../../docs/ar-demo.md)

## Run a trajectory

From the repository root, point the runner at the AR weight directory. The example uses the bundled text encoder and TurboVAE for video decoding.

```bash
TORCH_COMPILE_DISABLE=1 python scripts/infer_ar.py \
  --weights weights/ar \
  --image examples/reference.jpg \
  --prompt-file examples/prompt.txt \
  --trajectory examples/rollout.json \
  --seed 42 \
  --deterministic \
  --output outputs/ar.mp4
```

The default attention backend is FlashAttention 3 on Hopper. Use `--backend flash2` or `--backend sdpa` on supported hardware if that extension is unavailable. Use `--decoder hunyuan` to decode with the base VAE. The starting image is encoded independently at one frame with Hunyuan. The output contains generated frames and starts at trajectory frame 1.

The reference comparison used eager execution (`TORCH_COMPILE_DISABLE=1`) and deterministic cuDNN kernels for the VAE (`--deterministic`). For distributed token sharding across two local GPUs, launch the same script with `torchrun --standalone --nproc-per-node=2 scripts/infer_ar.py` and pass the arguments above.

The default command loads `transformer.safetensors` for all eight denoiser evaluations in each chunk. With `--final-gan`, the same transformer handles evaluations one through seven and `final_transformer.safetensors` handles evaluation eight. The primary transformer commits the causal history in either mode. [Pinned weights and component files](../../docs/weights.md#ar-transformers)

### Camera coordinates

The standalone JSON trajectory contains `frame_indices` beginning at zero and advancing by one per RGB frame at `fps: 24`. `camera_to_world` contains a 4×4 rigid transform per frame. Camera axes follow OpenCV conventions with +x right, +y down and +z forward. Translation is in metres, expressed in one fixed world frame. Supply the **effective camera poses** for the model, including any motion scaling applied to a captured trajectory.

Set `intrinsics` to one 3×3 calibration matrix or one matrix per frame. Focal lengths and principal point use pixels at the startup image resolution. Set `image_size` to `[width, height]` to enable a calibration size check. Frame zero describes the starting image. The model uses a new pose every four RGB frames. A full chunk generates sixteen frames from four latent planes. The terminal chunk pads its pose window with the final pose when the trajectory ends before all four planes have a sample.

The runtime retains the startup image in the cache while its temporal history rolls forward. At each 120-frame boundary it commits only the visible planes and trims its history. The decoder retains accepted latent overlap across that boundary.

### Weight directory

```text
ar/
  config.json
  transformer.safetensors
  vae/
    config.json
    diffusion_pytorch_model.safetensors
  text/
    config.json
    text_encoder/
    text_encoder_2/
    tokenizer/
  turbo_decoder.pt                     # used by the default TurboVAE decoder
  final_transformer.safetensors        # optional final-step GAN
```

Use `--text-weights` to point to a different copy of these assets or the compatible UCPE bundle. Pass `--text-features` to reuse precomputed features. The Safetensors file contains `tokens` with shape `[L,3584]` for up to 256 tokens and `pooled` with shape `[768]` or `[1,768]`.

## Incremental inference

`ARRuntime` retains causal state across calls to `generate`. Initialize it once with a DiT, a startup latent, a pose and calibration, caption features, and the settings from `load_configuration`. Call

```python
latents = runtime.generate(
    poses, intrinsics, accepted_planes=4, reset_after=False,
)
rgb_frames = decoder.decode(latents, accepted_planes=4)
```

Each call accepts four future camera matrices `[4,4,4]` and four calibration matrices `[4,3,3]`. Provide the number of visible latent planes and whether a scheduled history trim follows this chunk. Keep one runtime and one `OverlapDecoder` for the whole rollout. A separate streaming application can call this API once its next poses are available.
