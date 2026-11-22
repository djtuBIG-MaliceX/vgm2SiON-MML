"""First-pass detection of reusable SiON table envelopes.

SiON's ``np``/``nt`` and ``na`` commands replay a ``#TABLE`` at a fixed
frame rate.  They are independent of the instrument and are an efficient
representation for repeatedly shaped pitch/volume automation: one table
definition replaces a long chain of slurred note segments.

This pass samples each note's control-point curve at 60 Hz, selects shapes
that are useful or repeated, and attaches ``EnvelopeSpec`` objects to the
notes.  The MML writer then suppresses those control points and emits the
table command before the note.  Notes that do not get a table keep their
existing tie-segment representation.
"""

from __future__ import annotations

from collections import defaultdict

from .model import (
    EnvelopeSpec,
    EnvelopeTableLibrary,
    Note,
    Track,
)

ENV_FPS = 60
FRAME_SAMPLES = 44_100 / ENV_FPS


def _value_at(points: list[tuple[int, float]], default: float, sample: int) -> float:
    value = default
    for t, v in points:
        if t <= sample:
            value = v
        else:
            break
    return value


def _sample_curve(
    note: Note,
    points: list[tuple[int, float]],
    default: float,
    total_samples: int,
    transform,
) -> tuple[int, ...]:
    end = note.end if note.end is not None else total_samples
    if end <= note.start:
        return ()
    frames = max(1, int(round((end - note.start) * ENV_FPS / 44_100.0)))
    values: list[int] = []
    # The MML writer treats a control point exactly at the gate as part of
    # the next note's setup, not as a first-segment move.  Start the table
    # from the note's own gate value so the table and tie paths agree.
    later = [p for p in points if p[0] > note.start]
    # Walk the control points in lockstep with the sampled frames.  A fresh
    # linear scan per frame makes dense automation quadratic.
    point_index = 0
    current = default
    for i in range(frames):
        t = int(round(note.start + i * FRAME_SAMPLES))
        if t >= end:
            t = end - 1
        while point_index < len(later) and later[point_index][0] <= t:
            current = later[point_index][1]
            point_index += 1
        value = default if i == 0 else current
        values.append(int(transform(value)))
    return tuple(values)


def _curve_matches(
    note: Note,
    points: list[tuple[int, float]],
    base_value: float,
    sampled: tuple[int, ...],
    total_samples: int,
    transform,
    tolerance: float,
) -> bool:
    """Reject a table when its 60 Hz sampling misses the source curve."""
    end = note.end if note.end is not None else total_samples
    if not sampled:
        return False
    for t, raw in points:
        if t <= note.start or t >= end:
            continue
        index = int(round((t - note.start) / FRAME_SAMPLES))
        index = max(0, min(len(sampled) - 1, index))
        expected = transform(raw)
        if abs(sampled[index] - expected) > tolerance:
            return False
    return True


def _pitch_candidate(note: Note, total_samples: int) -> tuple[str, tuple[int, ...]] | None:
    if not note.pitch_points or note.end is None:
        return None
    relative = _sample_curve(
        note,
        note.pitch_points,
        note.pitch,
        total_samples,
        lambda value: round(value - note.pitch),
    )
    if len(relative) < 2 or len(set(relative)) < 2:
        return None
    # nt is the compact semitone form; np covers fractional detune.
    if all(v % 64 == 0 and -128 * 64 <= v <= 127 * 64 for v in relative):
        if not _curve_matches(
            note, note.pitch_points, note.pitch, tuple(v // 64 for v in relative),
            total_samples, lambda value: round((value - note.pitch) / 64.0), 1,
        ):
            return None
        return "nt", tuple(v // 64 for v in relative)
    if any(v < -8192 or v > 8191 for v in relative):
        return None
    # Only nt is emitted.  nt reproduces the source's semitone offsets
    # exactly; np's frame-table cursor does not preserve arbitrary fractional
    # pitch curves reliably enough for the converter, so those stay as
    # slurred note segments.
    return None


def _gain_candidate(note: Note, total_samples: int) -> tuple[str, tuple[int, ...]] | None:
    if not note.gain_points or note.end is None:
        return None
    values = _sample_curve(
        note,
        note.gain_points,
        note.gain,
        total_samples,
        lambda value: max(0, min(128, round(value * 128.0))),
    )
    if len(values) < 2 or len(set(values)) < 2:
        return None
    if not _curve_matches(
        note, note.gain_points, note.gain, values, total_samples,
        lambda value: max(0, min(128, round(value * 128.0))), 2,
    ):
        return None
    return "na", values


def assign_envelopes(
    tracks: list[Track],
    total_samples: int,
    *,
    limit: int = 255,
) -> tuple[EnvelopeTableLibrary, dict[str, int]]:
    """Attach reusable pitch/gain table envelopes to notes.

    A candidate is selected when its shape repeats or when it is long enough
    that the table body is cheaper than the slurred control-point segments.
    The remaining notes are left untouched.
    """
    library = EnvelopeTableLibrary(limit=limit)
    candidates: dict[tuple[str, tuple[int, ...]], list[tuple[Note, str]]] = defaultdict(list)

    for track in tracks:
        for note in track.timeline.notes:
            pitch = _pitch_candidate(note, total_samples)
            if pitch is not None:
                candidates[pitch].append((note, "pitch"))
            gain = _gain_candidate(note, total_samples)
            if gain is not None:
                candidates[gain].append((note, "gain"))

    scored: list[tuple[float, tuple[str, tuple[int, ...]]]] = []
    for key, uses in candidates.items():
        _kind, values = key
        # A shared table is always worthwhile; a one-off table needs enough
        # entries to beat the equivalent tied-note token chain.
        if len(uses) < 2 and len(values) < 8:
            continue
        score = len(uses) * len(values) + len(values) * 0.25
        scored.append((score, key))
    scored.sort(key=lambda item: (-item[0], item[1]))

    stats = {"pitch_tables": 0, "gain_tables": 0, "table_notes": 0}
    selected: dict[tuple[str, tuple[int, ...]], int] = {}
    for _score, key in scored:
        kind, values = key
        idx = library.add(values)
        if idx is None:
            continue
        selected[key] = idx

    for (kind, values), uses in candidates.items():
        idx = selected.get((kind, values))
        if idx is None:
            continue
        spec = EnvelopeSpec(kind, idx, 1)
        for note, field in uses:
            if field == "pitch":
                note.pitch_env = spec
                stats["pitch_tables"] += 1
            else:
                note.gain_env = spec
                stats["gain_tables"] += 1
        stats["table_notes"] += len(uses)
    return library, stats
