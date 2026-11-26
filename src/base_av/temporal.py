"""Whole-scene captions for the five temporal I2V segments."""

import json
from dataclasses import dataclass
from pathlib import Path

TEMPORAL_OUTPUT_FRAMES = (120, 121)


@dataclass(frozen=True)
class TemporalPrompt:
    scene: str
    captions: tuple[str, ...]


def parse_temporal_prompt(value) -> TemporalPrompt:
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object with scene and captions")
    if any(key in value for key in ("objects", "tracks", "background")):
        raise ValueError("Temporal text supports whole-scene captions")
    scene, captions = value.get("scene"), value.get("captions")
    if not isinstance(scene, str) or not scene.strip():
        raise ValueError("Temporal text requires a nonempty scene")
    if (
        not isinstance(captions, list)
        or len(captions) != 5
        or any(
            not isinstance(caption, str) or not caption.strip() for caption in captions
        )
    ):
        raise ValueError(
            "Temporal text requires exactly five nonempty whole-scene captions"
        )
    return TemporalPrompt(scene.strip(), tuple(caption.strip() for caption in captions))


def read_temporal_prompt(path: Path) -> TemporalPrompt:
    with path.open(encoding="utf-8-sig") as stream:
        return parse_temporal_prompt(json.load(stream))
