# Example inputs

`reference.jpg` is the 768×512 AV sample frame from the previous Kandinsky WM release. The prompt and trajectories were prepared for this release. The example intrinsics are approximate and the trajectories are synthetic. [Reference frame and upstream attribution](https://github.com/kandinskylab/kandinsky-wm/blob/14ff77cf3d734f07a8b04a1a16ea67a0e9a64d9b/assets/av/frame_1.jpg)

| File | Use |
|---|---|
| `reference.jpg`, `prompt.txt` | Common I2V input |
| `straight.json`, `left.json` | 121-frame camera input for a controlled UCPE comparison |
| `rollout.json` | 145 camera poses for a multi-chunk PiFlow replay |

Each trajectory is a standalone JSON file. `camera_to_world` contains one homogeneous 4×4 transform per RGB frame index. Camera axes follow OpenCV conventions with x right, y down and z forward. World coordinates in these examples use the first camera as their origin and are measured in metres. The translation is the fourth column and the 3×3 rotation maps camera axes into world axes. `intrinsics` gives the pinhole matrix in pixels at the declared `image_size`, ordered as width and height. `fps` gives the sampling rate of `frame_indices`. The files use a shared intrinsics matrix for every pose.

Run `python examples/make_trajectories.py` to regenerate the coordinates. Keep the prompt and seed fixed when comparing trajectories.

The preview uses seed 42 and the shared reference frame in all three rows. Base AV uses the hard temporal prompt example. UCPE uses the left-turn trajectory and PiFlow uses the rollout trajectory. Both camera-conditioned rows use deterministic VAE decoding. [Temporal prompt example](../base_av/prompt_segments.example.json) · [GPU comparisons](../docs/reproducibility.md)
