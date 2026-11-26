# Three F2LF camera paths

The three five-second clips share a start image, an end image, camera calibration, captions and seed. The straight path is the reference. The two arcs move the midpoint 3.5 m left or right while keeping both endpoints fixed. [First frame](first.png) · [Last frame](last.png) · [Reference latents](reference-latents.safetensors) · [Metric camera geometry](camera.json) · [Prompt](prompt.txt) · [Scene caption](scene.txt)

Install the F2LF environment and download the F2LF bundle. Run from the repository root. [F2LF setup](../../../src/f2lf/README.md) · [Weight bundles](../../../docs/weights.md)

```bash
INPUT=examples/f2lf/directions
for variant in straight left-3.5m right-3.5m; do
  case "$variant" in
    straight) offset=0 ;;
    left-3.5m) offset=-3.5 ;;
    right-3.5m) offset=3.5 ;;
  esac
  python scripts/infer_f2lf.py --weights weights/f2lf \
    --a "$INPUT/first.png" --b "$INPUT/last.png" \
    --reference-latents "$INPUT/reference-latents.safetensors" \
    --camera "$INPUT/camera.json" --prompt-file "$INPUT/prompt.txt" \
    --scene-file "$INPUT/scene.txt" --seed 2026102991 --offset-m "$offset" --deterministic \
    --output "outputs/f2lf-$variant.mp4"
done
```

The recipe generates all 121 frames at 768×512 and 24 fps. The reference latent file keeps the VAE encoding of the two PNGs fixed across runs. The standalone camera file gives calibrated endpoint poses in a metric world frame. The runner derives the intermediate poses from the selected offset. [Camera JSON conventions](../../../src/f2lf/README.md#camera-json)
