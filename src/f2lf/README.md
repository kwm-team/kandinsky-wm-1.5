# F2LF · First-to-last-frame generation

F2LF generates a five-second driving transition between two calibrated reference images. It encodes both references independently and uses them as conditioning slots at the first and last latent positions. All 121 video frames are generated between those anchors. The denoiser uses bidirectional attention with UCPE camera conditioning and an Euler flow-matching schedule with 50 updates.

## Install

The reference environment uses Python 3.12, PyTorch 2.7, CUDA 12.8 and an NVIDIA H100 with the FlashAttention 3 Hopper extension. Use a dedicated environment with the matching CUDA toolkit and C++ compiler. The denoiser compiles its pointwise kernels on the first run.

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e . -r src/f2lf/requirements.txt packaging psutil ninja wheel
MAX_JOBS=4 python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
MAX_JOBS=4 python -m pip install --no-build-isolation \
  'git+https://github.com/Dao-AILab/flash-attention.git@v2.7.4.post1#subdirectory=hopper'
```

## Generate a transition

```bash
python scripts/infer_f2lf.py --weights weights/f2lf \
  --a examples/f2lf/first.png --b examples/f2lf/last.png \
  --camera examples/f2lf/camera.json \
  --prompt-file examples/f2lf/prompt.txt --scene-file examples/f2lf/scene.txt \
  --deterministic --seed 2026102991 --output outputs/f2lf-straight.mp4
```

Use `--validate-only` to check input files and camera geometry before loading the model. Use `--latents-output outputs/f2lf.safetensors` to save the generated body latents alongside the MP4. For exact RGB comparisons, use `--deterministic` to turn off cuDNN autotuning during VAE decoding. The bundle includes the transformer, canonical buffers, negative text embeddings, HunyuanVideo VAE, and Qwen and CLIP conditioning assets.

The detailed prompt describes traffic and scene changes between the anchors. The shorter scene caption supplies the pooled CLIP conditioning. The two reference files may be native 768×512 images or aligned 1536×1024 images. The runner downsamples the latter to 768×512 before VAE encoding.

## Camera JSON

| Field | Meaning |
|---|---|
| `coordinate_system` | `metric_ENU`, with positions measured in metres |
| `resolution` | `[768, 512]`, the calibrated image plane |
| `c2w_A`, `c2w_B` | Rigid 4×4 camera-to-world matrices using right/down/forward camera axes and the same endpoint orientation |
| `K` | Shared 3×3 pinhole calibration matrix in native pixels |

The default path goes straight between the anchors. Change `--offset-m` for an arc with the same endpoint poses. A positive offset moves right and a negative offset moves left. The maximum allowed magnitude is 5 metres. To compare geometry, keep both images, captions and seed fixed. [Example camera file](../../examples/f2lf/camera.json)
