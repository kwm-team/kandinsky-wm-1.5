# Base AV accident examples

These looping WebPs use super-resolution renders from the Base AV checkpoint. The gallery shows selected cases from the benchmark's eight-scene accident subset. The input frame and original English prompt are included for each render. [PAI-Bench-G dataset](https://huggingface.co/datasets/shi-labs/physical-ai-bench-generation) · [Benchmark recipe](../../docs/benchmark.md)

| Scene | Benchmark ID | Inputs | Video |
| --- | --- | --- | --- |
| Intersection rollover | `av_accident_2_2` | [Frame](intersection-rollover/reference.jpg) · [Prompt](intersection-rollover/prompt.txt) | [WebP](intersection-rollover/video.webp) · [MP4](https://huggingface.co/datasets/sashakunitsyn/static_videos/resolve/8060bc2656ea72acbc40d64838e05d5b4cd923f5/kandinsky-wm-1.5/av/accidents/intersection-rollover.mp4) |
| Intersection collision | `av_accident_2_4` | [Frame](../base-av/intersection-collision/reference.jpg) · [Prompt](../base-av/intersection-collision/prompt.txt) | [WebP](../base-av/intersection-collision/preview.webp) · [MP4](https://huggingface.co/datasets/sashakunitsyn/static_videos/resolve/8060bc2656ea72acbc40d64838e05d5b4cd923f5/kandinsky-wm-1.5/av/base-av/intersection-collision.mp4) |
| Highway trailer collision | `av_accident_2_5` | [Frame](highway-trailer-collision/reference.jpg) · [Prompt](highway-trailer-collision/prompt.txt) | [WebP](highway-trailer-collision/video.webp) · [MP4](https://huggingface.co/datasets/sashakunitsyn/static_videos/resolve/8060bc2656ea72acbc40d64838e05d5b4cd923f5/kandinsky-wm-1.5/av/accidents/highway-trailer-collision.mp4) |
| Pickup and SUV collision | `av_accident_2_6` | [Frame](pickup-and-suv/reference.jpg) · [Prompt](pickup-and-suv/prompt.txt) | [WebP](pickup-and-suv/video.webp) · [MP4](https://huggingface.co/datasets/sashakunitsyn/static_videos/resolve/8060bc2656ea72acbc40d64838e05d5b4cd923f5/kandinsky-wm-1.5/av/accidents/pickup-and-suv.mp4) |

Run from the repository root with the Base AV environment and weights installed. This command writes a native-resolution render. [Super-resolution and WebP encoding](../../docs/gallery.md#video-super-resolution)

```bash
SCENE=examples/accidents/intersection-rollover
python scripts/infer_base_av.py --mode i2v --model-dir weights/base-av \
  --image "$SCENE/reference.jpg" --prompt "$(cat "$SCENE/prompt.txt")" \
  --seed 42 --output-frames 120 --visual-size 512 \
  --num-steps 50 --scheduler-scale 5 --guidance-weight 5 \
  --output outputs/intersection-rollover.mp4
```
