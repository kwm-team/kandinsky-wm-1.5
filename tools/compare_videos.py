"""Compare decoded RGB frames from two video files in timestamp order."""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from itertools import zip_longest
from pathlib import Path


@dataclass
class Comparison:
    width: int
    height: int
    fps: str
    frames: int
    different_frames: int
    max_rgb_delta: int
    mean_rgb_delta: float
    psnr_db: float | None


def compare(reference: Path, candidate: Path) -> Comparison:
    import av
    import numpy as np

    absolute_error = 0
    squared_error = 0
    different = 0
    count = 0
    max_delta = 0
    with av.open(str(reference)) as original, av.open(str(candidate)) as generated:
        a, b = original.streams.video[0], generated.streams.video[0]
        width, height, rate = a.width, a.height, a.average_rate
        if (width, height, rate) != (b.width, b.height, b.average_rate):
            raise ValueError("The videos have different resolution or frame rate")
        for left, right in zip_longest(original.decode(a), generated.decode(b)):
            if left is None or right is None:
                raise ValueError("The videos have different decoded frame counts")
            values = left.to_ndarray(format="rgb24").astype(np.int16)
            values -= right.to_ndarray(format="rgb24")
            errors = np.abs(values)
            peak = int(errors.max())
            max_delta = max(max_delta, peak)
            different += int(peak != 0)
            absolute_error += int(errors.sum(dtype=np.int64))
            squared_error += int(np.square(values, dtype=np.int32).sum(dtype=np.int64))
            count += 1
    if not count:
        raise ValueError("The videos contain no decodable frames")
    pixels = count * width * height * 3
    psnr = 10 * math.log10((255**2 * pixels) / squared_error) if squared_error else None
    return Comparison(width, height, str(rate), count, different, max_delta, absolute_error / pixels, psnr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--exact", action="store_true", help="Fail if any decoded RGB sample differs")
    args = parser.parse_args()
    try:
        result = compare(args.reference, args.candidate)
    except (OSError, ValueError, IndexError) as exc:
        parser.error(str(exc))
    print(json.dumps(asdict(result), indent=2))
    return int(args.exact and result.different_frames > 0)


if __name__ == "__main__":
    raise SystemExit(main())
