# Example inputs

`reference.jpg` is the 768×512 AV sample frame from the previous Kandinsky WM release. The prompt and trajectories were prepared for this release. The example intrinsics are approximate and the trajectories are synthetic. [Reference frame and upstream attribution](https://github.com/kandinskylab/kandinsky-wm/blob/14ff77cf3d734f07a8b04a1a16ea67a0e9a64d9b/assets/av/frame_1.jpg)

| File | Use |
|---|---|
| `reference.jpg`, `prompt.txt` | Common I2V input |
| `straight.json`, `left.json`, `right.json`, `stationary.json` | Four 121-frame camera paths for the controlled UCPE comparison |
| `rollout.json` | 145 camera poses for a multi-chunk AR replay |
| `ar/long-drive/` | [Long autoregressive driving example](../README.md#autoregressive-continuation) |
| `f2lf/` | F2LF quickstart inputs and a three-path showcase with its own calibrated references and captions |
| `base-av/` | Animated Base AV clips, conditioning frames and exact prompts |
| `accidents/` | PAI-Bench-G collision clips, conditioning frames and exact prompts |
| `ucpe/` | Four animated UCPE clips |

Each trajectory is a standalone JSON file. `camera_to_world` contains one homogeneous 4×4 transform per RGB frame index. Camera axes follow OpenCV conventions with x right, y down and z forward. World coordinates in these examples use the first camera as their origin and are measured in metres. The translation is the fourth column and the 3×3 rotation maps camera axes into world axes. `intrinsics` gives the pinhole matrix in pixels at the declared `image_size`, ordered as width and height. `fps` gives the sampling rate of `frame_indices`. The files use a shared intrinsics matrix for every pose.

Run `python examples/make_trajectories.py` to regenerate the coordinates. Keep the prompt and seed fixed when comparing trajectories.

Each UCPE gallery clip uses the shared reference frame, prompt and seed 42 with deterministic VAE decoding. See the repository README to play the videos side by side. [Video gallery](../README.md#examples) · [GPU comparisons](../docs/reproducibility.md)

The F2LF camera file stores two endpoint poses and shared intrinsics in a local East-North-Up frame. The runtime derives the path between those endpoints. The F2LF gallery fixes the references, captions and seed and varies only the lateral offset. [F2LF input format](../src/f2lf/README.md#camera-json)
