"""Read batch generation inputs and derive deterministic output names."""
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .temporal import TemporalPrompt, parse_temporal_prompt


@dataclass(frozen=True)
class Job:
    source: Path
    prompt: str | TemporalPrompt
    seed: int
    output: Path


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        if path.suffix.lower() == '.csv':
            reader = csv.DictReader(stream)
            if not {'filename', 'prompt'} <= set(reader.fieldnames or ()):
                raise ValueError('CSV requires filename and prompt columns')
            return list(reader)
        if path.suffix.lower() == '.json':
            rows = json.load(stream)
            if isinstance(rows, list):
                return rows
    raise ValueError('Expected a JSON array or a CSV manifest')


def find_source(candidates):
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError('Missing conditioning media. Tried ' + ', '.join(map(str, candidates)))


def load_jobs(args):
    jobs, outputs = [], set()
    manifest_dir = args.input.resolve().parent
    rows = read_rows(args.input)
    stop = None if args.limit is None else args.start + args.limit
    for index, row in enumerate(rows[args.start:stop], args.start):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} must be a JSON object")
        if args.temporal_text is not None:
            try:
                prompt = parse_temporal_prompt(row)
            except ValueError as error:
                raise ValueError(f"Invalid temporal prompt in row {index}. {error}") from error
        else:
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Row {index} requires a nonempty prompt")
        filename = row.get('filename')
        recorded_seed = 1137
        if filename is not None:
            if not filename or Path(filename).name != filename or Path(filename).suffix.lower() != '.mp4':
                raise ValueError(f'Row {index} requires a plain MP4 filename')
            stem = Path(filename).stem
            match = re.fullmatch(r'(.+)__([0-9]+)', stem)
            video_id = match[1] if match else stem
            recorded_seed = int(match[2]) if match else recorded_seed
        else:
            video_id = str(row.get('video_id', ''))
        if not video_id.strip() or video_id in ('.', '..') or '/' in video_id or '\\' in video_id:
            raise ValueError(f'Row {index} has an invalid video ID')

        if args.mode == 'v2v':
            root = args.video_dir or manifest_dir
            if row.get('video_path'):
                candidates = [root / row['video_path']]
            else:
                candidates = [root / filename] if filename else []
                candidates.append(root / f'{video_id}.mp4')
        elif row.get('img_id'):
            image = Path(row['img_id'])
            if args.image_dir:
                candidates = [args.image_dir / (image.name if image.is_absolute() else image)]
            else:
                candidates = [manifest_dir / image]
        else:
            root = args.image_dir or manifest_dir / 'condition_image'
            candidates = [root / f'{video_id}{ext}' for ext in ('.jpg', '.png', '.jpeg', '.webp')]
        source = find_source(candidates)
        seed = args.seed if args.seed is not None else recorded_seed
        output = args.output_dir / f'{video_id}__{seed}.mp4'
        if output.resolve() == source.resolve():
            raise ValueError(f'Output would overwrite its input video: {source}')
        if output.resolve() in outputs:
            raise ValueError(f'Duplicate output name: {output.name}')
        outputs.add(output.resolve())
        jobs.append(Job(source, prompt, seed, output))
    if {p.resolve() for p in outputs} & {job.source.resolve() for job in jobs}:
        raise ValueError('An output path overlaps a conditioning file')
    return jobs
