"""Shared helpers for the chip decoders."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..model import Track, VoiceLibrary
from ..vgm import VgmFile

FM_CHIPS = frozenset(
    {
        "ym2151",
        "ym2203",
        "ym2612",
        "ym2608",
        "ym2610",
        "ym3812",
        "ym3526",
        "ymf262",
        "ym2413",
    }
)


def has_fm_chip(vgm: VgmFile) -> bool:
    """Whether the log also contains an FM chip.

    PSG chips are mixed by VGM players with headroom so they cannot bury a
    simultaneous FM chip.  That headroom is only meaningful when an FM chip
    is present; a standalone PSG log should use the chip's full output.
    """
    return any(w.chip in FM_CHIPS for w in vgm.writes)


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


def normalize_carrier_levels(
    ops: tuple[tuple[int, ...], ...],
    carriers: tuple[int, ...],
    tl_index: int,
) -> tuple[tuple[tuple[int, ...], ...], int]:
    """Move the common carrier floor out of a voice and return it.

    SiON's ``x`` expression carries the per-note level; the stored voice only
    needs the relative carrier balance.  The floor is the quietest (highest
    TL) of the loudest carrier set's values, i.e. the minimum.
    """
    if not carriers:
        return ops, 0
    floor = min(ops[i][tl_index] for i in carriers if 0 <= i < len(ops))
    if floor <= 0:
        return ops, 0
    out = []
    for i, op in enumerate(ops):
        if i in carriers:
            values = list(op)
            values[tl_index] = max(0, values[tl_index] - floor)
            out.append(tuple(values))
        else:
            out.append(op)
    return tuple(out), floor
