# Reproducing outputs

Freeze the input media, prompt, camera trajectory, seed, weight bundle, sampler settings and decoder before comparing two implementations. Use the same precision policy and attention backend. Record the hardware and versions of PyTorch, CUDA and the enabled attention implementation.

For Base AV, check image conditioning and video-prefix continuation separately. For UCPE, include a camera path with both translation and rotation. For PiFlow, compare more than one chunk and include a run that crosses the cache reset boundary. Check startup, latent history, camera indexing and decoding separately when diagnosing a mismatch.

The comparison utility decodes both videos to RGB. It validates dimensions, FPS and decoded frame count before reporting the number of different frames, maximum and mean RGB error and PSNR. A null PSNR represents exact equality. Use `--exact` to give a failing exit code when decoded pixel values differ.

```bash
python tools/compare_videos.py source.mp4 candidate.mp4 --exact
```

Both videos should be encoded with the same codec and settings when using exact mode. Compare the raw latents as a separate check when isolating the denoiser or solver. Exact reproducibility applies to the recorded inputs and runtime configuration.

## GPU reference comparisons

The table compares independently executed native and release code on NVIDIA H100 GPUs. All clips use 24 fps. UCPE and PiFlow use deterministic cuDNN for exact decoded-RGB comparisons.

| Path | Test input | Result |
|---|---|---|
| Base AV I2V | 120 frames at 768×512 | Identical MP4 and decoded RGB |
| Base AV V2V | 25 frames at 384×256 with a 9-frame prefix | Identical MP4 and decoded RGB |
| Base AV temporal text | 120 frames at 768×512 in hard mode and 384×256 in both modes | Identical MP4 and decoded RGB |
| UCPE I2V | 121 frames at 768×512, left-turn trajectory | Identical latents and decoded RGB |
| UCPE T2V and V2V | 9 frames at 384×256 for each mode | Identical latents and decoded RGB |
| PiFlow | 144 frames at 768×512 over 10 chunks, including a history reset | Identical latents and decoded RGB |
| PiFlow sequence parallel and final-step GAN | 48 frames at 768×512 for each variant | Identical latents and decoded RGB |

Base AV comparisons use native-resolution outputs before super-resolution. The PiFlow comparison exercises the eager transformer and TurboVAE with matching input features.
