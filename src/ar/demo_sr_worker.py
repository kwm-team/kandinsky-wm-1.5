"""Run optional Kandinsky video SR in an independent Python environment."""

from contextlib import redirect_stdout
from pathlib import Path
import os
import sys

# Run directly with the SR interpreter, independent of the Torch install for AR.
from demo_control import read_rgb, write_rgb


def main():
    filename, device, scale = sys.argv[1:]
    output = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    with redirect_stdout(sys.stderr), output:
        import torch
        from kandinsky_sr.pipeline.factory import load_sr_pipeline

        torch.cuda.set_device(torch.device(device))
        pipeline = load_sr_pipeline(
            Path(filename), device, force=True, resolution_scale=int(scale)
        )
        previous = None
        with torch.inference_mode():
            while (rgb := read_rgb(sys.stdin.buffer)) is not None:
                pixels = torch.from_numpy(rgb).permute(0, 3, 1, 2)
                # The SR VAE takes one prefix plus a multiple of four.
                prefix = pixels[:1] if previous is None else previous
                result = pipeline(video=torch.cat((prefix, pixels))).frames[0]
                previous = pixels[-1:].clone()
                write_rgb(output, result[:, 1:].permute(1, 2, 3, 0).cpu().numpy())


if __name__ == "__main__":
    main()
