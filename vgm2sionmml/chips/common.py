"""Shared helpers for the chip decoders."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..model import Track, VoiceLibrary
from ..vgm import VgmFile


@dataclass
class ExtractResult:
    tracks: list[Track] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


def db_gain(db: float) -> float:
    """Linear amplitude from dB attenuation (positive = quieter)."""
    return 10.0 ** (-db / 20.0)


def pan_from_lr(left: bool, right: bool) -> tuple[float, float]:
    """Return (pan 0..8, gain multiplier) for a left/right enable pair."""
    if left and right:
        return 4.0, 1.0
    if left:
        return 0.0, 1.0
    if right:
        return 8.0, 1.0
    return 4.0, 0.0


def tl_gain(db_delta: float) -> float:
    """Gain for a total-level change (positive dB_delta = attenuation)."""
    return max(0.0, min(1.0, db_gain(db_delta)))


def sorted_writes(vgm: VgmFile, chip: str, instance: int) -> list:
    return [w for w in vgm.writes if w.chip == chip and w.chip_id == instance]


def family_track_order(base: int, channel: int) -> int:
    return base + channel


def dedup_voice(voices: VoiceLibrary, voice):
    return voices.add(voice)
