"""Apply pinned Kandinsky VSR 2.0 to constant-frame-rate videos on one CUDA GPU."""

import argparse
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile

MODEL_ID = "kandinskylab/Kandinsky-6.0-VSR-2.0-distilled2steps-Diffusers"
MODEL_REVISION = "36929c88a2cecc6d436444def27a89c8af0a37e6"
RESOLUTION_SCALE = 2.25


def read_video(path):
    import av
    from PIL import Image

    frames, timestamps = [], []
    with av.open(path) as container:
        stream = container.streams.video[0]
        if stream.average_rate is None:
            raise ValueError(f"Missing frame rate for {path}")
        for frame in container.decode(stream):
            timestamps.append(frame.pts * frame.time_base if frame.pts is not None else None)
            frames.append(Image.fromarray(frame.to_ndarray(format="rgb24")))
        rate = stream.average_rate
    if not frames:
        raise ValueError(f"No video frames in {path}")
    if any(pts != index / rate for index, pts in enumerate(timestamps)):
        raise ValueError(f"Expected constant frame rate and a zero start timestamp in {path}")
    return frames, rate, tuple(timestamps)


def write_video(path, frames, rate):
    import av
    import numpy as np

    with av.open(path, mode="w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height = frames[0].size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "10"}
        for image in frames:
            frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def source_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_video(path, size, rate, timestamps):
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        saved_timestamps = tuple(frame.pts * frame.time_base for frame in container.decode(stream))
        if (stream.width, stream.height) != size or stream.average_rate != rate or saved_timestamps != timestamps:
            raise ValueError(f"Unexpected resolution, frame rate, or timestamps in {path}")


def upscale_video(pipe, source, output, *, generator):
    if source.resolve() == output.resolve():
        raise ValueError("Input and output must refer to different files")
    digest = source_digest(source)
    frames, rate, timestamps = read_video(source)
    count = len(frames)
    snap = pipe.vae_scale_factor_spatial
    expected_size = tuple(2 * snap * round(dim * 1.125 / snap) for dim in frames[0].size)

    # KVAE admits 1 + k * temporal_scale frames. Trim only the frames added here.
    padding = (1 - count) % pipe.vae_scale_factor_temporal
    frames.extend([frames[-1]] * padding)
    result = pipe(video=frames, resolution_scale=RESOLUTION_SCALE, num_inference_steps=2, generator=generator).frames[0]
    if len(result) != len(frames):
        raise ValueError(f"VSR returned {len(result)} frames for {len(frames)} input frames")

    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_suffix(output.suffix + ".source.sha256")
    with NamedTemporaryFile(dir=output.parent, suffix=".mp4", delete=False) as file:
        temporary = Path(file.name)
    receipt = temporary.with_suffix(".sha256")
    try:
        write_video(temporary, result[:count], rate)
        verify_video(temporary, expected_size, rate, timestamps)
        if source_digest(source) != digest:
            raise ValueError(f"Input video changed during super-resolution: {source}")
        receipt.write_text(f"{digest}\n{source_digest(temporary)}\n", encoding="ascii")
        sidecar.unlink(missing_ok=True)
        temporary.replace(output)
        receipt.replace(sidecar)
    finally:
        temporary.unlink(missing_ok=True)
        receipt.unlink(missing_ok=True)
    print(f"Saved {output} ({expected_size[0]}x{expected_size[1]}, {count} frames at {rate} fps)", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Native MP4. Repeat with --output.")
    parser.add_argument("--output", type=Path, action="append", required=True, help="VSR MP4. Repeat with --input.")
    parser.add_argument("--seed", type=int, default=42, help="Reset the CUDA random generator for each video")
    parser.add_argument("--model-dir", type=Path, help="Local copy of the pinned public checkpoint")
    args = parser.parse_args()
    if len(args.input) != len(args.output):
        parser.error("Give one --output for each --input")
    paths_in = {path.resolve() for path in args.input}
    paths_out = {path.resolve() for path in args.output}
    if paths_in & paths_out or len(paths_out) != len(args.output):
        parser.error("Choose distinct output paths that differ from every input path")
    for path in args.input:
        if not path.is_file():
            parser.error(f"Input video does not exist: {path}")
    return args


def main():
    args = parse_args()
    import torch
    from diffusers import Kandinsky6SRPipeline

    torch._inductor.config.max_autotune = True
    load_options = {"local_files_only": True} if args.model_dir else {"revision": MODEL_REVISION}
    model = str(args.model_dir.resolve()) if args.model_dir else MODEL_ID
    model_label = model if args.model_dir else f"{MODEL_ID}@{MODEL_REVISION}"
    pipe = Kandinsky6SRPipeline.from_pretrained(
        model, torch_dtype=torch.bfloat16, **load_options,
    )
    pipe.enable_model_cpu_offload()
    pipe.transformer.set_attention_backend("flex")
    pipe.transformer.compile_repeated_blocks(fullgraph=False)
    for source, output in zip(args.input, args.output, strict=True):
        print(f"Upscaling {source} with {model_label}", flush=True)
        generator = torch.Generator("cuda").manual_seed(args.seed)
        upscale_video(pipe, source, output, generator=generator)


if __name__ == "__main__":
    main()
