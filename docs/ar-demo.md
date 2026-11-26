# Keyboard driving demo

Use the released AR checkpoint to drive from a calibrated reference image. `Space` starts or pauses generation, `W` accelerates, `S` brakes and `A` / `D` steer. Arrow keys are also supported. Choose one tab to control the drive. Other tabs can watch the stream. Unfocusing the control tab releases the keys and pauses future generation.

Each command affects the next pose window. A full window is 16 frames, and the browser can play already generated frames after pausing. The released example has a level camera. With a custom calibration, movement follows the horizontal plane of its starting camera.

## One node

Install the [AR environment](../src/ar/README.md#install) and [download the AR bundle](weights.md). Run from the repository root. The measured 24 fps profile uses eight H100s. Seven run the transformer and the last one runs TurboVAE and hosts the browser.

```bash
OMP_NUM_THREADS=1 TORCH_COMPILE_DISABLE=1 torchrun --standalone --nproc-per-node=8 scripts/demo_ar.py \
  --weights weights/ar --image examples/reference.jpg \
  --calibration examples/rollout.json --prompt-file examples/prompt.txt \
  --size 576 384 --reset-frames 128
```

Open `http://127.0.0.1:7860` on the VAE node. For a remote node, forward that port to your browser with `ssh -L 7860:127.0.0.1:7860 USER@VAE_NODE`. Keep the default loopback bind for private access. The server accepts keyboard input without login.

For one GPU, start the same command with `python scripts/demo_ar.py` in place of `torchrun --standalone --nproc-per-node=8 scripts/demo_ar.py`. The demo rescales the image and intrinsics together. By default it keeps the input resolution and the bundle's 120-frame history reset. The 128-frame option aligns resets to eight full generation windows. The calibration file supplies frame zero and intrinsics, while keyboard input supplies the remaining poses.

The demo defaults to `ar/transformer.safetensors` for the eight solver evaluations. Add `--final-gan` to use `ar/final_transformer.safetensors` for evaluation eight while the base transformer maintains causal history.

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

Add `--benchmark-chunks 32 --warmup-chunks 4` to run the built-in straight-drive benchmark. It reports generated frames per second after TurboVAE and JPEG encoding, including cross-rank transfer and causal history resets. It excludes startup, browser playback pacing and network transit to the browser.

The eight-H100 profile above produced 512 JPEG-ready frames in 19.96 s, or 25.66 fps, across 32 measured chunks. Median DiT and latent transfer time was 618 ms per chunk. It uses the base AR transformer and leaves SR off.

In a separate live run, the same profile streamed 320 generated frames plus the reference. A loopback MJPEG client received 256 steady-state frames at 24.00 fps. This measures HTTP frame delivery. Browser rendering and remote-network latency depend on the client and connection.

The demo precomputes the linear integration coefficients of each fixed PiFlow segment. This changes the floating-point operation order. The offline AR CLI retains the reference integration.
