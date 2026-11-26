# Motorcycle entry timing with Base AV

Each clip uses the same conditioning image, seed and model. The four prompt schedules move the rider's entry from the lower left through successive one-second intervals. The timing in the labels comes from the prompts. The first visible parts of the motorcycle appear in the same order, with some variation in their exact timing. [Conditioning image](../motorcycle-overtake/reference.jpg) · [Prompt schedules](prompts.json)

| Prompted entry at 0.5 s | Prompted entry at 1.5 s |
| --- | --- |
| [![Motorcycle entry prompted at 0.5 seconds](entry-0.5s.webp)](https://huggingface.co/datasets/sashakunitsyn/static_videos/resolve/8060bc2656ea72acbc40d64838e05d5b4cd923f5/kandinsky-wm-1.5/av/base-av/temporal-motorcycle/entry-0.5s.mp4) | [![Motorcycle entry prompted at 1.5 seconds](entry-1.5s.webp)](https://huggingface.co/datasets/sashakunitsyn/static_videos/resolve/8060bc2656ea72acbc40d64838e05d5b4cd923f5/kandinsky-wm-1.5/av/base-av/temporal-motorcycle/entry-1.5s.mp4) |

| Prompted entry at 2.5 s | Prompted entry at 3.5 s |
| --- | --- |
| [![Motorcycle entry prompted at 2.5 seconds](entry-2.5s.webp)](https://huggingface.co/datasets/sashakunitsyn/static_videos/resolve/8060bc2656ea72acbc40d64838e05d5b4cd923f5/kandinsky-wm-1.5/av/base-av/temporal-motorcycle/entry-2.5s.mp4) | [![Motorcycle entry prompted at 3.5 seconds](entry-3.5s.webp)](https://huggingface.co/datasets/sashakunitsyn/static_videos/resolve/8060bc2656ea72acbc40d64838e05d5b4cd923f5/kandinsky-wm-1.5/av/base-av/temporal-motorcycle/entry-3.5s.mp4) |

## Reproduce the clips

Install the Base AV environment and download the Base AV bundle. Run from the repository root. [Environment setup](../../../src/base_av/README.md#environment) · [Weight bundle](../../../docs/weights.md)

```bash
python scripts/infer_base_av.py --mode i2v --model-dir weights/base-av \
  --input examples/base-av/temporal-motorcycle/prompts.json \
  --temporal-text hard --output-frames 120 --visual-size 512 \
  --num-steps 50 --guidance-weight 5 --scheduler-scale 5 --seed 42 \
  --output-dir outputs/temporal-motorcycle
```

The recipe generates native 768×512 MP4s at 24 fps. The Euler schedule uses 50 points and 49 updates. Add `--start 2 --limit 1` to reproduce only the 2.5 s schedule. Add `--dry-run` to check input and output paths without a GPU.

Separate single-GPU H100 runs with this command reproduced all four source MP4s byte for byte.

The gallery uses the regenerated renders. Each 480×320 WebP loop displays all 120 frames in about five seconds and opens its super-resolved MP4. [Super-resolution and WebP encoding](../../../docs/gallery.md#video-super-resolution)
