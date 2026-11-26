# Weight bundles

The release has one self-contained inference bundle per model, including its text conditioning and video components. The model code runs from this repository. Hugging Face Hub is the distribution point for versioned weight files.

| Model | Local directory | Download size | Weight contents |
|---|---|---:|---|
| Base AV | `weights/base-av/` | 27.3 GB | Video DiT, HunyuanVideo VAE, Qwen and CLIP encoders and tokenizers |
| UCPE | `weights/ucpe/` | 25.7 GB | Merged camera-conditioned DiT, architecture config and matching conditioning components |
| PiFlow | `weights/piflow/` | 34.0 GB | Autoregressive DiT, architecture and solver settings, HunyuanVideo VAE, TurboVAE decoder, matching Qwen and CLIP encoders and optional final-step DiT |

PiFlow includes its compatible text conditioning assets under `text/`. It can also accept precomputed caption features. [PiFlow input formats](../piflow/README.md)

The bundle's `bundle.json` records its model identifier, relative file paths, byte counts and SHA-256 digests. The downloader uses the manifest to fetch exactly the release payload. Pass a full commit hash from the model card so the manifest and weights come from the same Hub revision.

```bash
kwm-av fetch --repo-id "$HF_REPO_ID" --revision "$HF_COMMIT" --output weights/base-av
kwm-av inspect weights/base-av --model base-av --verify
```

For already staged weights, use `kwm-av inspect` directly. With `--verify`, it reads and hashes all weight files. A regular inspection checks layout and file sizes.

Keep the inference bundle's own component revisions with it. Shapes alone cannot establish that one model's text or VAE assets reproduce another model's inference. List any required download separately in the model's quickstart.
