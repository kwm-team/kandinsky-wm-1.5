# Reproducing outputs

Freeze the input media, prompt, camera trajectory, seed, weight bundle, sampler settings and decoder before comparing two implementations. Use the same precision policy and attention backend. Record the hardware and versions of PyTorch, CUDA and the enabled attention implementation.

For Base AV, check image conditioning and video-prefix continuation separately. For UCPE, include a camera path with both translation and rotation. For AR, compare more than one chunk and include a run that crosses the cache reset boundary. For F2LF, vary the path between the same calibrated endpoints. Check startup, camera indexing and decoding separately when diagnosing a mismatch. For AR, also check the latent history.

Compare the dimensions, frame rate and decoded frame count of both outputs. To check decoded RGB for exact equality with FFmpeg, compare the frame hashes of both streams.

```bash
ffmpeg -v error -i source.mp4 -map 0:v:0 -pix_fmt rgb24 -f framehash source.rgb.sha256
ffmpeg -v error -i candidate.mp4 -map 0:v:0 -pix_fmt rgb24 -f framehash candidate.rgb.sha256
diff -u source.rgb.sha256 candidate.rgb.sha256
```

Both videos should be encoded with the same codec and settings when using exact mode. Compare the raw latents as a separate check when isolating the denoiser or solver. Exact reproducibility applies to the recorded inputs and runtime configuration.

## GPU reference comparisons

The table compares independently executed native and release code on NVIDIA H100 GPUs. All clips use 24 fps. UCPE and AR use deterministic cuDNN for exact decoded-RGB comparisons.

| Path | Test input | Result |
|---|---|---|
| Base AV I2V | 120 frames at 768×512 | Identical MP4 and decoded RGB |
| Base AV V2V | 25 frames at 384×256 with a 9-frame prefix | Identical MP4 and decoded RGB |
| Base AV temporal text | 120 frames at 768×512 in hard mode and 384×256 in both modes | Identical MP4 and decoded RGB |
| Base AV motorcycle entry | Four temporal prompt schedules at 768×512 | All four release MP4s match their source renders byte for byte |
| UCPE I2V | 121 frames at 768×512, left-turn trajectory | Identical latents and decoded RGB |
| UCPE T2V and V2V | 9 frames at 384×256 for each mode | Identical latents and decoded RGB |
| AR | 144 frames at 768×512 over 10 chunks, including a history reset | Identical latents and decoded RGB |
| AR sequence parallel and final-step GAN | 48 frames at 768×512 for each variant | Identical latents and decoded RGB |
| F2LF | 121 frames at 768×512 per path, straight and 3.5 m right arc | Identical latents, MP4 and decoded RGB with deterministic VAE decoding |

Base AV comparisons use native-resolution outputs before super-resolution. The AR comparison exercises the eager transformer and TurboVAE with matching input features.

The default F2LF path uses the upstream cuDNN autotuning policy. Independent default runs produced identical conditioning features and latents. The resulting straight and right-arc MP4s scored 40.38 dB and 40.15 dB PSNR against the respective upstream runs. Both VAEs gave identical raw RGB when replayed with deterministic cuDNN. The F2LF CLI with `--deterministic` reproduced those native replays byte for byte through MP4 encoding.
