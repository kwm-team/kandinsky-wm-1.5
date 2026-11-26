"""Read versioned weight bundles without importing a model runtime."""

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path

MODELS = ("base-av", "ucpe", "ar", "f2lf")
MANIFEST = "bundle.json"


@dataclass(frozen=True)
class BundleFile:
    size: int
    sha256: str


@dataclass(frozen=True)
class Bundle:
    model: str
    files: dict[str, BundleFile]


def read_manifest(root: Path) -> Bundle:
    document = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("format_version") != 1:
        raise ValueError("Unsupported weight bundle manifest version")
    model = document.get("model")
    if model == "piflow":
        model = "ar"
    if model not in MODELS:
        raise ValueError(f"Unknown model in bundle manifest: {model!r}")
    entries = document.get("files")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Bundle manifest must list its payload files")
    files = {}
    for name, metadata in entries.items():
        parts = name.split("/")
        if (
            any(part in ("", ".", "..") for part in parts)
            or name == MANIFEST
            or re.search(r"[*?\[\]\\:\x00-\x1f]", name)
        ):
            raise ValueError(f"Invalid bundle path: {name!r}")
        size = metadata.get("bytes") if isinstance(metadata, dict) else None
        digest = metadata.get("sha256") if isinstance(metadata, dict) else None
        if type(size) is not int or size <= 0:
            raise ValueError(f"Invalid file size for {name}")
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid SHA-256 for {name}")
        files[name] = BundleFile(size, digest)
    return Bundle(model, files)


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inspect_bundle(root: Path, *, expected_model: str | None = None, verify: bool = False) -> Bundle:
    bundle = read_manifest(root)
    if expected_model is not None and bundle.model != expected_model:
        raise ValueError(f"Expected {expected_model} bundle, found {bundle.model}")
    for name, info in bundle.files.items():
        path = root / name
        if not path.is_file():
            raise ValueError(f"Missing bundle file: {name}")
        if path.stat().st_size != info.size:
            raise ValueError(f"Size mismatch: {name}")
        if verify and sha256_file(path) != info.sha256:
            raise ValueError(f"SHA-256 mismatch: {name}")
    return bundle


def write_manifest(root: Path, model: str) -> Path:
    """Index a prepared directory of inference files."""
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model!r}")
    entries = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == root / MANIFEST:
            continue
        name = path.relative_to(root).as_posix()
        entries[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if not entries:
        raise ValueError(f"Empty weight directory: {root}")
    target = root / MANIFEST
    content = {"format_version": 1, "model": model, "files": entries}
    target.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    inspect_bundle(root, expected_model=model)
    return target
