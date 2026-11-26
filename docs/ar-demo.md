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

## Measure the pipeline

Add `--benchmark-chunks 24 --warmup-chunks 12` to the eight-H100 command. The benchmark drives straight and measures 384 generated frames after warm-up. Throughput includes latent transfer, history resets, TurboVAE decoding and JPEG encoding. Startup and delivery to the browser are outside the measured interval.

At 768×432 using the primary AR transformer with SR off, TP7 produced **24.78 JPEG-ready fps**, with a median DiT and latent transfer time of **639 ms per 16 frames**. To render the full 768×512 frame, omit `--crop`. The full frame measured 22.92 JPEG-ready fps with the same layout.

A live steering run with the bundled caption delivered 192 generated frames plus the reference. A loopback MJPEG client received the final 128 frames at **24.00 fps** after warm-up.

The demo precomputes the linear integration coefficients of each fixed PiFlow segment. This changes the floating-point operation order. The offline AR CLI retains the reference integration.

## Video super-resolution

Multi-node video super-resolution (VSR) release is TBD.
