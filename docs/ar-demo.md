# Keyboard driving demo

Use the released AR checkpoint to drive from a calibrated reference image. `Space` starts or pauses generation, `W` accelerates, `S` brakes and `A` / `D` steer. Arrow keys are also supported. Choose one tab to control the drive. Other tabs can watch the stream. Unfocusing the control tab releases the keys and pauses future generation.

Each command affects the next pose window. A full window is 16 frames, and the browser can play already generated frames after pausing. The released example has a level camera. With a custom calibration, movement follows the horizontal plane of its starting camera.

## One node with eight H100s

Install the AR environment, download the AR bundle and run from the repository root. TP7 shards the DiT across CUDA devices 0–6. Device 7 is dedicated to TurboVAE and JPEG delivery. Use `--crop 768 432` for a 16:9 stream. The demo removes 40 rows from the top and bottom of the bundled image and shifts the camera's principal point. The retained pixels keep their original scale. [AR environment](../src/ar/README.md#install) · [AR weight bundle](weights.md)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
  torchrun --standalone --nproc-per-node=8 scripts/demo_ar.py \
  --weights weights/ar --image examples/reference.jpg \
  --calibration examples/rollout.json --prompt-file examples/prompt.txt \
  --dit-parallel tp --dit-ranks 7 --vae-rank 7 \
  --crop 768 432 --dit-graphs --compile-dit --reset-frames 128
```

Open `http://127.0.0.1:7860` on the VAE node. For a remote node, forward that port to your browser with `ssh -L 7860:127.0.0.1:7860 USER@VAE_NODE`. Keep the default loopback bind for private access. The server accepts keyboard input without login.

The 128-frame history reset aligns with eight full generation windows. The calibration file supplies frame zero and intrinsics. Keyboard input supplies the remaining poses.

The demo defaults to `ar/transformer.safetensors` for the eight solver evaluations. Add `--final-gan` to use `ar/final_transformer.safetensors` for evaluation eight while the base transformer maintains causal history.

The CUDA graphs use FlashAttention 3 and capture on first use of each cache layout. The initial chunks can pause while kernels compile and the graphs are captured. Leave `TORCH_COMPILE_DISABLE` and `TORCHDYNAMO_DISABLE` unset when using `--compile-dit`. To run eager TP, omit `--dit-graphs` and `--compile-dit`.

TP changes the rounding order of projected activations through BF16 reductions across shards. Small differences from the dense model can accumulate into different rollouts when generated latents feed back into later windows.

## Optional video super-resolution

Install the public Kandinsky 6 video SR package in a separate Python environment. The validated setup used the distilled VSR 1.0 bundle, the Torch VAE backend and two denoising steps. [SR installation guide](https://github.com/kandinskylab/kandinsky-6-sr/tree/2c9f25d32c78ce90795fb6fc59e4aaa6e6d41f09#manual-setup-this-repository) · [Tested SR weights revision](https://huggingface.co/kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers/tree/3d6672923a4dfcd387961864654e123dd53b8ee4)

Download the pinned SR model on the machine that runs the SR process.

```bash
hf download kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers \
  --revision 3d6672923a4dfcd387961864654e123dd53b8ee4 --local-dir weights/vsr
```

```yaml
# sr.yaml
sr:
  checkpoint_path: weights/vsr
  num_steps: 2
  resolution_scale: 2
  nabla: flex
  vae_backend: torch
  lu_load_scales: [2x]
```

### Run SR on a separate node

The public VSR pipeline uses one GPU per stream and produces 1536×864 from the 768×432 crop. In a two-node launch, the first node hosts TP7 and TurboVAE. In an 8+8+1 allocation, the first node runs SP8, the middle node runs SR on one of its GPUs and the last node runs TurboVAE. The browser endpoint is on the TurboVAE node in either layout.

Use the same AR environment on each node. Every node needs the checkout, the input files and `weights/ar/config.json`. The DiT and TurboVAE nodes need their AR weights. The SR node needs the separate SR environment, `sr.yaml` and the SR weights. Set `SR_PYTHON` to the SR interpreter path on the SR node.

| Layout | `NODES` | `NODE_RANK` | `LOCAL_PROCESSES` | `CUDA_VISIBLE_DEVICES` | Role |
| --- | ---: | ---: | ---: | --- | --- |
| Two nodes | 2 | 0 | 8 | `0,1,2,3,4,5,6,7` | TP7, TurboVAE and browser |
| Two nodes | 2 | 1 | 1 | `0` | SR |
| Three nodes (8+8+1) | 3 | 0 | 8 | `0,1,2,3,4,5,6,7` | SP8 |
| Three nodes (8+8+1) | 3 | 1 | 1 | `0` | SR |
| Three nodes (8+8+1) | 3 | 2 | 1 | `0` | TurboVAE and browser |

Run the following from the repository root on each node after setting the variables from the table. Use the same reachable address of node 0 as `MASTER_ADDR`, choose a common `MASTER_PORT` and set `SR_PYTHON` consistently. The locked PyTorch version assigns global ranks across the unequal process counts with `TORCH_ELASTIC_WORKER_IDENTICAL=0`.

```bash
export OMP_NUM_THREADS=1 TORCH_ELASTIC_WORKER_IDENTICAL=0 CUDA_VISIBLE_DEVICES

if [ "$NODES" = 2 ]; then
  parallel=(--dit-parallel tp --dit-ranks 7 --vae-rank 7 --sr-rank 8 --dit-graphs)
else
  parallel=(--dit-parallel sp --dit-ranks 8 --vae-rank 9 --sr-rank 8)
fi

torchrun --nnodes "$NODES" --nproc-per-node "$LOCAL_PROCESSES" \
  --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  scripts/demo_ar.py --weights weights/ar --image examples/reference.jpg \
  --calibration examples/rollout.json --prompt-file examples/prompt.txt \
  --crop 768 432 --reset-frames 128 --compile-dit "${parallel[@]}" \
  --sr-python "$SR_PYTHON" --sr-config sr.yaml --sr-scale 2
```

Set `GLOO_SOCKET_IFNAME` and `NCCL_SOCKET_IFNAME` to the nodes' routable interfaces when the default selection is unsuitable. The 24 fps profile measures AR with SR off. SR adds model processing and a round trip of uncompressed RGB frames.

## Measure the pipeline

Add `--benchmark-chunks 24 --warmup-chunks 12` to the eight-H100 command. The benchmark drives straight and measures 384 generated frames after warm-up. Throughput includes latent transfer, history resets, TurboVAE decoding and JPEG encoding. Startup and delivery to the browser are outside the measured interval.

At 768×432 using the primary AR transformer with SR off, TP7 produced **24.78 JPEG-ready fps**, with a median DiT and latent transfer time of **639 ms per 16 frames**. To render the full 768×512 frame, omit `--crop`. The full frame measured 22.92 JPEG-ready fps with the same layout.

A live steering run with the bundled caption delivered 192 generated frames plus the reference. A loopback MJPEG client received the final 128 frames at **24.00 fps** after warm-up.

The demo precomputes the linear integration coefficients of each fixed PiFlow segment. This changes the floating-point operation order. The offline AR CLI retains the reference integration.
