# PiFlow AV

Generate an autoregressive driving video from a starting image, a caption, and a trajectory. PiFlow uses eight transformer evaluations per chunk. It commits generated latents to a causal key/value cache and rolls that context forward as new poses arrive.

## Install

The reference environment uses Python 3.12.9, PyTorch 2.7.0, CUDA 12.8, and an NVIDIA H100. Build FlashAttention with the matching CUDA toolkit and a C++ compiler. Use a dedicated environment for PiFlow's dependency versions.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e . -r piflow/requirements.txt packaging psutil ninja wheel
MAX_JOBS=4 python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
MAX_JOBS=4 python -m pip install --no-build-isolation \
  'git+https://github.com/Dao-AILab/flash-attention.git@v2.7.4.post1#subdirectory=hopper'
```

Qwen caption encoding uses FlashAttention 2. Install the Hopper extension to use the default `--backend flash3` for the video transformer. [FlashAttention build instructions](https://github.com/Dao-AILab/flash-attention#flashattention-3-beta-release).

## Run a trajectory

From the repository root, point the runner at the PiFlow weight directory. The example uses the bundled text encoder and TurboVAE for video decoding.

```bash
TORCH_COMPILE_DISABLE=1 python -m piflow \
  --weights weights/piflow \
  --image examples/reference.jpg \
  --prompt-file examples/prompt.txt \
  --trajectory examples/rollout.json \
  --seed 42 \
  --deterministic \
  --output outputs/piflow.mp4
```

The default attention backend is FlashAttention 3 on Hopper. Use `--backend flash2` or `--backend sdpa` on supported hardware if that extension is unavailable. Use `--decoder hunyuan` to decode with the base VAE. The starting image is encoded independently at one frame with Hunyuan. The output contains generated frames and starts at trajectory frame 1.

The reference comparison used eager execution (`TORCH_COMPILE_DISABLE=1`) and deterministic cuDNN kernels for the VAE (`--deterministic`). The launcher also supports distributed token sharding. Replace `python -m piflow` with `torchrun --standalone --nproc-per-node=2 -m piflow` to run across two local GPUs.

The optional `--final-gan` selects the companion transformer on the last solver evaluation. This requires `final_transformer.safetensors`. The base transformer still provides the previous solver evaluations and commits all causal history.

### Camera coordinates

The standalone JSON trajectory contains `frame_indices` beginning at zero and advancing by one per RGB frame at `fps: 24`. `camera_to_world` contains a 4×4 rigid transform per frame. Camera axes follow OpenCV conventions with +x right, +y down and +z forward. Translation is in metres, expressed in one fixed world frame. Supply the **effective camera poses** for the model, including any motion scaling applied to a captured trajectory.

Set `intrinsics` to one 3×3 calibration matrix or one matrix per frame. Focal lengths and principal point use pixels at the startup image resolution. Set `image_size` to `[width, height]` to enable a calibration size check. Frame zero describes the starting image. The model uses a new pose every four RGB frames. A full chunk generates sixteen frames from four latent planes. The terminal chunk pads its pose window with the final pose when the trajectory ends before all four planes have a sample.

The runtime retains the startup image in the cache while its temporal history rolls forward. At each 120-frame boundary it commits only the visible planes and trims its history. The decoder retains accepted latent overlap across that boundary.

### Weight directory

```text
piflow/
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

`PiFlowRuntime` retains causal state across calls to `generate`. Initialize it once with a DiT, a startup latent, a pose and calibration, caption features, and the settings from `load_configuration`. Call

```python
latents = runtime.generate(
    poses, intrinsics, accepted_planes=4, reset_after=False,
)
rgb_frames = decoder.decode(latents, accepted_planes=4)
```

Each call accepts four future camera matrices `[4,4,4]` and four calibration matrices `[4,3,3]`. Provide the number of visible latent planes and whether a scheduled history trim follows this chunk. Keep one runtime and one `OverlapDecoder` for the whole rollout. A separate streaming application can call this API once its next poses are available.
