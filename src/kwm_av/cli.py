"""Fetch and verify native Kandinsky WM 1.5 inference weights."""

import argparse
import json
import re
import sys
from pathlib import Path

from .weights import MANIFEST, MODELS, inspect_bundle, read_manifest


def fetch(repo_id: str, revision: str, output: Path) -> None:
    if not re.fullmatch("[0-9a-fA-F]{40}", revision):
        raise ValueError("Pass a full 40-character Hub commit to --revision")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ValueError("Install the Hub extra with pip install -e '.[hub]'") from exc
    options = {"repo_id": repo_id, "revision": revision, "local_dir": output}
    snapshot_download(**options, allow_patterns=[MANIFEST])
    manifest = read_manifest(output)
    snapshot_download(**options, allow_patterns=list(manifest.files))
    inspect_bundle(output, expected_model=manifest.model, verify=True)
    print(output.resolve())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="Check a local weight bundle")
    inspect.add_argument("weights", type=Path)
    inspect.add_argument("--model", choices=MODELS)
    inspect.add_argument("--verify", action="store_true", help="Read all payloads and verify their SHA-256")

    hub = commands.add_parser("fetch", help="Download and verify the files in a pinned Hub manifest")
    hub.add_argument("--repo-id", required=True)
    hub.add_argument("--revision", required=True)
    hub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            bundle = inspect_bundle(args.weights, expected_model=args.model, verify=args.verify)
            print(json.dumps({
                "model": bundle.model,
                "files": len(bundle.files),
                "bytes": sum(file.size for file in bundle.files.values()),
                "sha256_verified": args.verify,
            }, indent=2))
        else:
            fetch(args.repo_id, args.revision, args.output)
    except (OSError, ValueError) as exc:
        print(f"kwm-av: {exc}", file=sys.stderr)
        return 2
    return 0
