"""Two-pass MML size optimization and tempo-grid selection."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .model import Track

VGM_RATE = 44100
_NOTE_DURATION = re.compile(r"^([a-g](?:\+)?)(\d+)$")
_REST_DURATION = re.compile(r"^(r)(\d+)$")


def _text_len(tokens: list[str]) -> int:
    return sum(len(t) for t in tokens) + max(0, len(tokens) - 1)


def _balanced_loops(tokens: list[str]) -> bool:
    depth = 0
    for token in tokens:
        if token == "[":
            depth += 1
        elif token.startswith("]"):
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _has_control(tokens: list[str]) -> bool:
    for token in tokens:
        if token in ("$",) or token.startswith("%") or token == "t" or token.startswith("t"):
            return True
        if token.startswith("#") or token.endswith(";"):
            return True
    return False


def compress_default_length(tokens: list[str]) -> tuple[list[str], int]:
    """Use ``l<n>`` for one common simple note length."""
    counts: dict[str, int] = {}
    for token in tokens:
        match = _NOTE_DURATION.match(token) or _REST_DURATION.match(token)
        if match:
            duration = match.group(2)
            counts[duration] = counts.get(duration, 0) + 1
    if not counts:
        return tokens, 0
    duration, count = max(counts.items(), key=lambda item: (item[1], -len(item[0])))
    if count < 4:
        return tokens, 0
    saved = count * (len(duration) - 0) - (len(duration) + 1)
    if saved <= 2:
        return tokens, 0
    out: list[str] = []
    inserted = False
    for token in tokens:
        if not inserted and token.startswith("v") and token[1:].isdigit():
            out.append(token)
            out.append("l" + duration)
            inserted = True
            continue
        match = _NOTE_DURATION.match(token) or _REST_DURATION.match(token)
        if match and match.group(2) == duration:
            out.append(match.group(1))
        else:
            out.append(token)
    if not inserted:
        out.insert(0, "l" + duration)
    return out, count


def _loop_savings(block: list[str], count: int) -> int:
    body = _text_len(block)
    old = body * count + (count - 1)
    new = body + 4  # [ body ]count
    return old - new


def compress_loops(tokens: list[str], max_block: int = 32) -> tuple[list[str], int]:
    """Replace immediately repeated token blocks with ``[block]n``."""
    loops = 0
    tokens = list(tokens)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(tokens):
            best = None
            max_n = min(max_block, (len(tokens) - i) // 2)
            for n in range(max_n, 1, -1):
                block = tokens[i : i + n]
                if _has_control(block):
                    continue
                if block[0] in ("&", "&&", "*"):
                    continue
                if tokens[i + n : i + 2 * n] != block:
                    continue
                count = 2
                while tokens[i + count * n : i + (count + 1) * n] == block:
                    count += 1
                after = i + count * n
                # A tie immediately after `]` is invalid in SiON: the parser's
                # last event is the loop terminator, not the note inside it.
                if after < len(tokens) and tokens[after] in ("&", "&&", "*"):
                    continue
                if _loop_savings(block, count) > 2:
                    best = (n, count, block)
                    break
            if best is None:
                i += 1
                continue
            n, count, block = best
            tokens[i : i + n * count] = ["["] + block + [f"]{count}"]
            loops += 1
            changed = True
            i += n + 2
    return tokens, loops


def _macro_savings(body: list[str], count: int) -> int:
    body_len = _text_len(body)
    definition = body_len + 3  # "#A=" + ";" 
    use = count  # one character per use
    return count * body_len - definition - use


def find_macros(tracks: list[list[str]], limit: int = 26) -> tuple[list[tuple[str, list[str]]], int]:
    """Greedy repeated-phrase macros shared by all tracks."""
    candidates: dict[tuple[str, ...], int] = {}
    lengths = (2, 3, 4, 6, 8, 12, 16, 24, 32)
    for tokens in tracks:
        n = len(tokens)
        for size in lengths:
            if size > n:
                continue
            for i in range(0, n - size + 1):
                body = tuple(tokens[i : i + size])
                if _has_control(body) or not _balanced_loops(list(body)):
                    continue
                if body[0] in ("&", "&&", "*"):
                    continue
                if any(len(t) == 1 and "A" <= t <= "Z" for t in body):
                    continue
                candidates[body] = candidates.get(body, 0) + 1
    ranked = []
    for body, count in candidates.items():
        if count < 2:
            continue
        saving = _macro_savings(list(body), count)
        if saving > 5:
            ranked.append((saving, count, body))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    chosen: list[tuple[str, list[str]]] = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    used_bodies: set[tuple[str, ...]] = set()
    for saving, _count, body in ranked:
        if len(chosen) >= limit:
            break
        if body in used_bodies:
            continue
        # A simple exact-token replacement cannot overlap an earlier macro
        # name, and definitions are emitted before the tracks.
        if any(t in letters for t in body):
            continue
        name = letters[len(chosen)]
        chosen.append((name, list(body)))
        used_bodies.add(body)
    # Apply the accepted macros to every track.  Long bodies first so a
    # shorter macro does not fragment the phrase before it is recognized.
    for name, body in sorted(
        chosen, key=lambda item: len(item[1]), reverse=True
    ):
        pattern = tuple(body)
        for index, tokens in enumerate(tracks):
            out: list[str] = []
            i = 0
            while i < len(tokens):
                if tuple(tokens[i : i + len(pattern)]) == pattern:
                    out.append(name)
                    i += len(pattern)
                else:
                    out.append(tokens[i])
                    i += 1
            tracks[index] = out
    return chosen, sum(1 for _ in chosen)


@dataclass
class OptimizeStats:
    original_chars: int = 0
    optimized_chars: int = 0
    loops: int = 0
    macros: int = 0
    default_lengths: int = 0


def optimize_tracks(tracks: list[list[str]]) -> tuple[list[list[str]], list[tuple[str, list[str]]], OptimizeStats]:
    stats = OptimizeStats(original_chars=sum(_text_len(t) for t in tracks))
    out: list[list[str]] = []
    for tokens in tracks:
        tokens, default_count = compress_default_length(tokens)
        if default_count:
            stats.default_lengths += 1
        tokens, loops = compress_loops(tokens)
        stats.loops += loops
        out.append(tokens)
    macros, _ = find_macros(out)
    stats.macros = len(macros)
    stats.optimized_chars = sum(_text_len(t) for t in out)
    for _name, body in macros:
        stats.optimized_chars += _text_len(body) + 3
    return out, macros, stats


def _event_samples(tracks: list[Track]) -> list[int]:
    """Times that should line up with the tick grid, weighted by kind."""
    events: list[int] = []
    for track in tracks:
        for note in track.timeline.notes:
            events.append(note.start)
            if note.end is not None:
                events.append(note.end)
            events.extend(t for t, _v in note.pitch_points)
            events.extend(t for t, _v in note.gain_points)
            events.extend(t for t, _v in note.pan_points)
            events.extend(t for t, _v in note.wave_points)
            events.extend(t for t, _op, _v in note.op_tl_points)
    return events


def detect_bpm(tracks: list[Track], total_samples: int, rate: int = 60) -> tuple[int, float]:
    """Pick a tick resolution that best aligns the source event grid.

    The converter's whole note has 1920 ticks.  A candidate is scored by the
    squared fractional tick error of note boundaries and control changes; a
    small preference for 120 BPM only breaks ties.  The caller can override
    this with an explicit BPM.
    """
    events = _event_samples(tracks)
    if not events:
        return 120, 0.0
    # Sample at most ~12000 events; this keeps auto-tempo cheap on huge logs.
    if len(events) > 12000:
        step = max(1, len(events) // 12000)
        events = events[::step]
    best_bpm = 120
    best_score = math.inf
    # Prefer tempos whose tick is an integral number of the source frame
    # (735 samples at 60 Hz, 882 at 50 Hz).  Without this a coarse exotic BPM
    # can hide a badly aligned frame grid in the aggregate fractional error.
    aligned = []
    if rate:
        for bpm in range(60, 151):
            frame_ticks = 8.0 * bpm / rate
            if abs(frame_ticks - round(frame_ticks)) < 1e-7:
                aligned.append(bpm)
    candidates = aligned or list(range(60, 151))
    for bpm in candidates:
        scale = 8.0 * bpm / VGM_RATE
        err = 0.0
        for sample in events:
            x = sample * scale
            err += (x - round(x)) ** 2
        err /= len(events)
        # Prefer 120 over an equally aligned exotic tempo.
        err += abs(bpm - 120) * 1e-7
        if err < best_score:
            best_score = err
            best_bpm = bpm
    # A finer grid always fits arbitrary write times slightly better.  Stay
    # with the conventional 120 BPM unless another musical tempo is clearly
    # better than it.
    if 120 in candidates:
        scale = 8.0 * 120 / VGM_RATE
        err120 = sum(
            (sample * scale - round(sample * scale)) ** 2
            for sample in events
        ) / len(events)
        if err120 <= best_score * 1.05:
            return 120, err120
    return best_bpm, best_score
