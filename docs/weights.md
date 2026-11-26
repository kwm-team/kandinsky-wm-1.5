# Weight bundles

The release has one self-contained inference bundle per model, including its text conditioning and video components. The four bundles are subfolders of a single Hugging Face model repository. [Kandinsky WM 1.5 AV weights](https://huggingface.co/kandinskylab/Kandinsky-WM-1.5-AV)

| Model | Local directory | Download size | Weight contents |
|---|---|---:|---|
| Base AV | `weights/base-av/` | 27.3 GB | Video DiT, HunyuanVideo VAE, Qwen and CLIP encoders and tokenizers |
| UCPE | `weights/ucpe/` | 25.7 GB | Merged camera-conditioned DiT, architecture config and matching conditioning components |
| AR | `weights/ar/` | 34.0 GB | Autoregressive DiT, architecture and solver settings, HunyuanVideo VAE, TurboVAE decoder, matching Qwen and CLIP encoders and optional final-step DiT |
| F2LF | `weights/f2lf/` | 25.7 GB | Two-reference DiT, canonical inference buffers, negative text embeddings, HunyuanVideo VAE and matching Qwen and CLIP encoders |

AR includes its compatible text conditioning assets under `text/`. It can also accept precomputed caption features. [AR input formats](../src/ar/README.md)

Each subfolder has a `bundle.json` with its model identifier, relative file paths, byte counts and SHA-256 digests. Pass a full commit hash from the model card so the manifest and weights come from the same Hub revision. With `--model base-av --output weights`, the downloader writes `weights/base-av/` and verifies the payload.

```bash
kwm-av fetch --repo-id kandinskylab/Kandinsky-WM-1.5-AV \
  --revision "$HF_COMMIT" --model base-av --output weights
kwm-av inspect weights/base-av --model base-av --verify
```

For already staged weights, use `kwm-av inspect` directly. With `--verify`, it reads and hashes all weight files. A regular inspection checks layout and file sizes.

Keep the inference bundle's own component revisions with it. Shapes alone cannot establish that one model's text or VAE assets reproduce another model's inference. List any required download separately in the model's quickstart.
