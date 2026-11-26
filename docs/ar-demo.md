# Keyboard driving demo

Use the released AR checkpoint to drive from a calibrated reference image. `Space` starts or pauses generation, `W` accelerates, `S` brakes and `A` / `D` steer. Arrow keys are also supported. Choose one tab to control the drive. Other tabs can watch the stream. Unfocusing the control tab releases the keys and pauses future generation.

Each command affects the next pose window. A full window is 16 frames, and the browser can play already generated frames after pausing. The released example has a level camera. With a custom calibration, movement follows the horizontal plane of its starting camera.

## One node with eight H100s

Install the AR environment, download the AR bundle and run from the repository root. TP7 shards the DiT across CUDA devices 0–6. Device 7 is dedicated to TurboVAE and JPEG delivery. The bundled image runs at its native 768×512 resolution. [AR environment](../src/ar/README.md#install) · [AR weight bundle](weights.md)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
  torchrun --standalone --nproc-per-node=8 scripts/demo_ar.py \
  --weights weights/ar --image examples/reference.jpg \
  --calibration examples/rollout.json --prompt-file examples/prompt.txt \
  --dit-parallel tp --dit-ranks 7 --vae-rank 7 \
  --dit-graphs --compile-dit --reset-frames 128
```

Open `http://127.0.0.1:7860` on the VAE node. For a remote node, forward that port to your browser with `ssh -L 7860:127.0.0.1:7860 USER@VAE_NODE`. Keep the default loopback bind for private access. The server accepts keyboard input without login.

The 128-frame history reset aligns with eight full generation windows. The calibration file supplies frame zero and intrinsics. Keyboard input supplies the remaining poses.

The demo defaults to `ar/transformer.safetensors` for the eight solver evaluations. Add `--final-gan` to use `ar/final_transformer.safetensors` for evaluation eight while the base transformer maintains causal history.

The CUDA graphs use FlashAttention 3 and capture on first use of each cache layout. The initial chunks can pause while kernels compile and the graphs are captured. Leave `TORCH_COMPILE_DISABLE` and `TORCHDYNAMO_DISABLE` unset when using `--compile-dit`. To run eager TP, omit `--dit-graphs` and `--compile-dit`.

TP changes the rounding order of projected activations through BF16 reductions across shards. Small differences from the dense model can accumulate into different rollouts when generated latents feed back into later windows.

## Two or three nodes

Run one command per node in the same AR environment with matching model, startup and calibration files. Set `MASTER_ADDR` to a routable address of node 0. Set `NODE_RANK` to 0, 1 or 2 on the corresponding machine. The browser runs on the node containing the VAE rank.

```bash
# Two nodes, one DiT GPU and one VAE GPU
TORCH_COMPILE_DISABLE=1 torchrun --nnodes=2 --nproc-per-node=1 \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port=29500 \
  scripts/demo_ar.py --weights weights/ar --image examples/reference.jpg \
  --calibration examples/rollout.json --prompt-file examples/prompt.txt
```

For three nodes without SR, change `--nnodes` to `3`. The first two ranks shard the transformer. The last rank handles TurboVAE and the browser. For one GPU per node on a host with several GPUs, set `CUDA_VISIBLE_DEVICES` to the selected GPU on each node.

Increase `--nproc-per-node` uniformly to use more GPUs. `--dit-ranks N` selects the first N global ranks, and `--vae-rank R` selects the VAE rank, outside that prefix. Unassigned ranks wait at the end of the session. For example, two eight-GPU nodes with `--dit-ranks 8 --vae-rank 8` keep the transformer on the first node and use the first GPU of the second node for the VAE. Use `GLOO_SOCKET_IFNAME` and `NCCL_SOCKET_IFNAME` if the default interface cannot reach the other nodes.

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

Add `--sr-python /path/to/sr-env/bin/python --sr-config sr.yaml --sr-scale 2` to the launcher. With no dedicated SR rank, the SR process uses the VAE GPU. With three one-GPU nodes, pass `--dit-ranks 1 --vae-rank 1 --sr-rank 2` to place the transformer, VAE and SR on separate nodes. SR increases the time before a chunk reaches the browser.

On three H100 nodes with one GPU per role, this SR setup delivered two 16-frame chunks at 768×512 from 384×256 AR frames. The first SR stage took 132.9 s including initialization and compilation. The next 16-frame SR stage and transport took 0.93 s.

## Measure the pipeline

Add `--benchmark-chunks 24 --warmup-chunks 12` to the eight-H100 command. The benchmark drives straight and measures 384 generated frames after warm-up. Throughput includes latent transfer, history resets, TurboVAE decoding and JPEG encoding. Startup and delivery to the browser are outside the measured interval.

At 768×512 using the primary AR transformer with SR off, TP7 produced **22.92 JPEG-ready fps**. The median DiT and latent transfer time was **686 ms per 16 frames**. The target is 24 fps.

The demo precomputes the linear integration coefficients of each fixed PiFlow segment. This changes the floating-point operation order. The offline AR CLI retains the reference integration.
