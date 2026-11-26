# PAI-Bench-G · Autonomous driving

The reported scores come from the final Base AV checkpoint's native-resolution I2V run on 118 autonomous-driving scenes. The setup used the benchmark's English prompts and conditioning images, seed 42, 120 frames at 24 fps, and 768×512 resolution. The sampler used 50 Euler schedule points, shift 5 and text guidance 5. The gallery shows separate super-resolution renders from the same checkpoint. [Benchmark dataset](https://huggingface.co/datasets/shi-labs/physical-ai-bench-generation)

## Generate the AV subset

Run from this repository in the Base AV environment. Point `DATASET` at the downloaded PAI-Bench-G dataset.

```bash
DATASET=/absolute/path/to/physical-ai-bench-generation
RUN="$PWD/outputs/paibench-av"
mkdir -p "$RUN"

python - "$DATASET" "$RUN/av.json" <<'PY'
import json
import sys
from pathlib import Path

dataset, output = map(Path, sys.argv[1:])
source = dataset / "cosmos_predict2_bench_full_info.json"
rows = [
    {"video_id": row["video_id"], "prompt": row["prompt_en"], "img_id": row["image_name"]}
    for row in json.loads(source.read_text(encoding="utf-8"))
    if row["video_id"].startswith("av_")
]
assert len({row["video_id"] for row in rows}) == len(rows) == 118
assert sum(row["video_id"].startswith("av_accident_") for row in rows) == 8
output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

torchrun --standalone --nproc-per-node=8 scripts/infer_base_av.py --mode i2v \
  --model-dir weights/base-av --input "$RUN/av.json" \
  --image-dir "$DATASET/condition_image" --output-dir "$RUN/videos" \
  --seed 42 --output-frames 120 --visual-size 512 \
  --num-steps 50 --scheduler-scale 5 --guidance-weight 5
```

Use the archived 118 native-resolution MP4s when recomputing the reported run. A fresh generation and judge run can produce a different score.

To reproduce the showcase rendering workflow, use the original WM 1.5 checkout with its prepared `filename,prompt` AV CSV, weight bundle, inference Python and `ksr-infer` executable. The upscaled clips are written under `sr/`.

```bash
(
  cd "$WM15_SOURCE"
  NPROC_PER_NODE=8 PYTHON="$WM15_PYTHON" WM15_SR_CLI="$SR_CLI" \
    bash scripts/run_generate_physical_ai_bench_v169.sh --mode i2v \
    --model-dir "$WM15_BUNDLE" --input "$AV_CSV" \
    --output-dir "$RUN/showcase" --seed 42 --sr
)
```

## Run the Qwen judge

Use `Qwen/Qwen3-VL-235B-A22B-Instruct` with vLLM. The reference evaluator used tensor parallelism across eight GPUs, GPU memory utilization `0.92` and the default setting with expert parallelism disabled. Set those options in `generation/pbench/vqa_evaluation.py` when using the pinned upstream checkout. Install its Generation environment separately from Base AV. [PAI-Bench-G evaluator setup](https://github.com/SHI-Labs/physical-ai-bench/blob/5bf80d18bbebea2c6c9802b37da76e391c4e781e/generation/README.md)

The pinned evaluator recognizes bare scene filenames. Stage aliases for seed 42 and pass the AV-only manifest so it scores the same selection.

```bash
BENCH_REPO=/absolute/path/to/physical-ai-bench
VIDEOS="$RUN/videos"  # Or point to the archived native-resolution videos
mkdir -p "$RUN/judge-input"
for clip in "$VIDEOS"/*__42.mp4; do
  name=${clip##*/}
  ln -sfn "$clip" "$RUN/judge-input/${name%__42.mp4}.mp4"
done

cd "$BENCH_REPO/generation"
uv run python evaluate_vqa.py \
  --prompt_file "$RUN/av.json" --vqa_questions_dir "$DATASET/vqa" \
  --video_dir "$RUN/judge-input" --output_dir "$RUN/qwen" \
  --model_name Qwen/Qwen3-VL-235B-A22B-Instruct --tensor_parallel_size 8
```

## Recompute the table scores

Average `accuracy` from `vqa_detailed_results.json` over the AV scenes and multiply by 100. Apply the same mean to IDs beginning with `av_accident_` for the eight-scene score.

```bash
python - "$RUN/av.json" "$RUN/qwen/vqa_detailed_results.json" <<'PY'
import json
import sys
from pathlib import Path
from statistics import mean

manifest, report = [json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:]]
scores = {row["video_id"]: row["accuracy"] for row in report}
assert len(report) == len(scores) == 118
assert set(scores) == {row["video_id"] for row in manifest}
accidents = [score for scene, score in scores.items() if scene.startswith("av_accident_")]
assert len(accidents) == 8
print(f"All AV scenes: {100 * mean(scores.values()):.2f}")
print(f"Accident scenes: {100 * mean(accidents):.2f}")
PY
```

The supplementary ChatGPT column comes from judging sampled video frames. To recompute it from the archived answers, select the final SFT rows in the linked CSV, average `score_percent` over all 118 scenes, and repeat over the eight IDs beginning with `av_accident_`. [Archived per-scene scores](https://gigaeye-kandinsky-spark.obs.ru-moscow-1.hc.sbercloud.ru/bulygin/habr/physical_ai_av/research/pai_gpt_judge_20261116/per_scene_all_five.csv)

## Compare models on AV118

The Qwen3-VL results for Base AV, MiniMax H3, Cosmos3-Nano and Cosmos3-Edge share 118 scene IDs, 490 benchmark questions and the same answer keys. Scores are the mean of the per-video accuracies on a 0–100 scale. Eight scene IDs form the accident subset. The Cosmos runs used prompts adapted to their generator. Resolution, frame count and prompt processing can vary across the source runs. [Normalized per-scene scores](paibench-av118-scores.csv)

| Model | Qwen · AV | Qwen · Accidents | ChatGPT · AV | ChatGPT · Accidents |
| --- | ---: | ---: | ---: | ---: |
| Kandinsky WM 1.5 Base AV | 78.06 | 81.80 | 71.32 | 76.71 |
| MiniMax H3 | 76.84 | 76.07 | 71.42 | 76.15 |
| Cosmos3-Nano | 74.94 | 67.18 | 68.98 | 49.63 |
| Cosmos3-Edge | 65.91 | 34.79 | 60.15 | 15.00 |

The ChatGPT comparison uses sampled frames. The MiniMax H3 values come from a separate ten-frame-per-clip review and its public aggregate is linked below. The archived frame-review answer key is heavily skewed toward yes answers. A yes-only baseline scores 92.00 on the AV macro average, so these results measure answer agreement on this question set. [Frame review protocol and baseline](https://gigaeye-kandinsky-spark.obs.ru-moscow-1.hc.sbercloud.ru/bulygin/habr/physical_ai_av/research/pai_gpt_judge_20261116/report_all_five.txt) · [MiniMax H3 aggregate](https://gigaeye-kandinsky-spark.obs.ru-moscow-1.hc.sbercloud.ru/bulygin/habr/physical_ai_av/research/pai_h3_chatgpt_20261116/results.json)

To reproduce the table from the normalized CSV, group by `model`. Average `qwen_accuracy_percent` and, where available, `chatgpt_accuracy_percent`. Repeat with rows where `is_accident` is `1`. The published leaderboard uses its own generation and evaluation protocol. [Official leaderboard](https://huggingface.co/spaces/shi-labs/physical-ai-bench-leaderboard)
