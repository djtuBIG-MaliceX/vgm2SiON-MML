"""SiON MML text renderer."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .model import (
    FmVoice,
    LfoState,
    Note,
    SimpleVoice,
    Song,
    Track,
)
from .pitch import units_to_note_k
from .optimize import optimize_tracks

TICKS_PER_WHOLE = 1920
MAX_LINE = 108
NOTE_NAMES = ("c", "c+", "d", "d+", "e", "f", "f+", "g", "g+", "a", "a+", "b")
DIVISORS = tuple(d for d in range(TICKS_PER_WHOLE, 0, -1) if TICKS_PER_WHOLE % d == 0)


@lru_cache(maxsize=None)
def _duration_pieces(ticks: int) -> tuple[int, ...]:
    """Minimal decomposition of 1..1920 ticks into divisors of 1920."""
    if ticks <= 0:
        return ()
    best: list[tuple[int, ...] | None] = [None] * (ticks + 1)
    best[0] = ()
    for n in range(1, ticks + 1):
        choice: tuple[int, ...] | None = None
        for d in DIVISORS:
            if d > n:
                continue
            prev = best[n - d]
            if prev is None:
                continue
            cand = prev + (d,)
            if choice is None or len(cand) < len(choice):
                choice = cand
        best[n] = choice
    return best[ticks] or (ticks,)


def duration_token(ticks: int) -> str:
    """MML duration *argument*, e.g. ``4`` or ``1^8``.

    ``c4`` is a quarter note, so the argument is ``1920 / tick_count``.  A
    duration that is not a note value is written as a chain of extensions.
    """
    if ticks <= 0:
        return "1920"
    wholes, rem = divmod(ticks, TICKS_PER_WHOLE)
    pieces: list[int] = [TICKS_PER_WHOLE] * wholes
    if rem:
        pieces.extend(_duration_pieces(rem))
    if not pieces:
        pieces = [TICKS_PER_WHOLE]
    out = str(TICKS_PER_WHOLE // pieces[0])
    for piece in pieces[1:]:
        out += "^" + str(TICKS_PER_WHOLE // piece)
    return out


def ticks_from_samples(sample: int, bpm: int) -> int:
    return int(round(sample * 8.0 * bpm / 44100.0))


def _pan_value(pan: float) -> int:
    return max(0, min(8, int(round(pan))))


def _gain_value(gain: float) -> int:
    return max(0, min(128, int(round(gain * 128.0))))


@dataclass
class RenderStats:
    notes: int = 0
    tied_segments: int = 0
    rests: int = 0
    loop_split: int = 0
    clamped: int = 0
    original_chars: int = 0
    optimized_chars: int = 0
    macros: int = 0
    loops: int = 0


class MmlRenderer:
    def __init__(self, song: Song, optimize: bool = True) -> None:
        self.song = song
        self.optimize = optimize
        self.stats = RenderStats()

    # ------------------------------------------------------------------ header
    def _title(self) -> str:
        title = (
            self.song.title.replace("\n", " ")
            .replace("{", "(")
            .replace("}", ")")
            .replace(";", ",")
        )
        if not title:
            title = self.song.title
        return f"#TITLE{{{title}}};"

    def _wavetable_headers(self) -> list[str]:
        return [
            table.mml(index)
            for index, table in enumerate(self.song.waves.tables)
        ]

    def _envelope_table_headers(self) -> list[str]:
        return [
            table.mml(index)
            for index, table in enumerate(self.song.tables.tables)
        ]

    def _voice_headers(self) -> list[str]:
        blocks: list[str] = []
        for family in ("opm", "opn", "opl"):
            for index, voice in self.song.voices.by_family(family):
                blocks.append(voice.mml(index))
        return blocks

    # ------------------------------------------------------------------ tracks
    def render(self) -> str:
        # Register every voice before writing the headers: the headers are
        # emitted before the tracks, so late registration would leave the
        # definitions out.
        for track in self.song.tracks_in_order():
            for note in track.sorted_notes():
                if isinstance(note.voice, FmVoice):
                    self.song.voices.add(note.voice)
        parts: list[str] = [self._title(), ""]
        wavetables = self._wavetable_headers()
        if wavetables:
            parts.extend(wavetables)
            parts.append("")
        envelope_tables = self._envelope_table_headers()
        if envelope_tables:
            parts.extend(envelope_tables)
            parts.append("")
        voices = self._voice_headers()
        if voices:
            parts.extend(voices)
            parts.append("")
        rendered: list[tuple[Track, list[str]]] = []
        for track in self.song.tracks_in_order():
            tokens = self._render_track_tokens(track, include_tempo=True)
            if tokens is None:
                continue
            rendered.append((track, tokens))
        macros: list[tuple[str, list[str]]] = []
        if rendered:
            if self.optimize:
                tracks_only = [tokens for _track, tokens in rendered]
                optimized, macros, opt_stats = optimize_tracks(tracks_only)
                rendered = [
                    (track, tokens) for (track, _old), tokens in zip(rendered, optimized)
                ]
                self.stats.original_chars = opt_stats.original_chars
                self.stats.optimized_chars = opt_stats.optimized_chars
                self.stats.macros = opt_stats.macros
                self.stats.loops = opt_stats.loops
            else:
                self.stats.original_chars = sum(
                    len(token) for _track, tokens in rendered for token in tokens
                )
                self.stats.optimized_chars = self.stats.original_chars
        if macros:
            for name, body in macros:
                parts.append(f"#{name}=" + " ".join(body) + ";")
            parts.append("")
        for track, tokens in rendered:
            parts.append(f"// {track.label}")
            parts.append(self._wrap(tokens))
            parts.append("")
        if not rendered:
            # No playable tracks; keep a syntactically valid empty sequence.
            parts.append("%0 v0 q8 r1;")
        return "\n".join(parts).rstrip() + "\n"

    def _render_track_tokens(self, track: Track, include_tempo: bool) -> list[str] | None:
        notes = track.sorted_notes()
        if not notes:
            return None
        bpm = self.song.bpm
        loop_tick = (
            ticks_from_samples(self.song.loop_sample, bpm)
            if self.song.loop_sample is not None
            else None
        )
        if loop_tick is not None and loop_tick <= 0:
            loop_tick = 0
        if loop_tick is not None and loop_tick >= ticks_from_samples(self.song.total_samples, bpm):
            loop_tick = None

        loop_tick = self._split_notes(track, loop_tick)
        notes = track.timeline.notes
        if loop_tick is not None:
            pre = [n for n in notes if n.start < self.song.loop_sample]
            post = [n for n in notes if n.start >= self.song.loop_sample]
        else:
            pre, post = notes, []
        end_tick = ticks_from_samples(self.song.total_samples, bpm)

        tokens: list[str] = []
        init = self._track_init(track, include_tempo)
        tokens.extend(init)
        state = _TrackState()

        if loop_tick is not None and loop_tick == 0:
            tokens.append("$")
            self._emit_notes(track, post, 0, state, tokens)
            if not post:
                tokens.append("r1")
            else:
                cursor = self._cursor_after(post)
                if end_tick > cursor:
                    tokens.append("r" + duration_token(end_tick - cursor))
                    self.stats.rests += 1
            return tokens

        self._emit_notes(track, pre, 0, state, tokens)
        if loop_tick is not None:
            cursor = self._cursor_after(pre)
            gap = loop_tick - cursor
            if gap > 0:
                tokens.append("r" + duration_token(gap))
                self.stats.rests += 1
            tokens.append("$")
            self._emit_notes(track, post, loop_tick, state, tokens)
            if not post:
                # A non-empty loop body keeps the repeat well-defined.
                tokens.append("r" + duration_token(max(1, end_tick - loop_tick)))
            else:
                cursor = self._cursor_after(post)
                if end_tick > cursor:
                    tokens.append("r" + duration_token(end_tick - cursor))
                    self.stats.rests += 1
        else:
            cursor = self._cursor_after(pre)
            if end_tick > cursor + TICKS_PER_WHOLE:
                # Keep a little tail so every track reaches the song's end.
                tokens.append("r" + duration_token(end_tick - cursor))
                self.stats.rests += 1
        return tokens

    def _track_init(self, track: Track, include_tempo: bool) -> list[str]:
        tokens = [f"%{track.module}"]
        if include_tempo:
            tokens.append(f"t{self.song.bpm}")
        tokens.extend(["q8", "v16"])
        return tokens

    def _split_notes(self, track: Track, loop_tick: int | None) -> int | None:
        if loop_tick is None:
            return None
        loop_sample = self.song.loop_sample
        if loop_sample is None:
            return None
        notes = track.timeline.notes
        changed = False
        out: list[Note] = []
        for note in notes:
            if note.start < loop_sample and (note.end is None or note.end > loop_sample):
                # Close the intro copy at the loop point.
                end = note.end
                intro = Note(
                    start=note.start,
                    end=loop_sample,
                    pitch=note.pitch,
                    voice=note.voice,
                    gain=note.gain,
                    pan=note.pan,
                    pitch_points=[p for p in note.pitch_points if p[0] < loop_sample],
                    gain_points=[p for p in note.gain_points if p[0] < loop_sample],
                    pan_points=[p for p in note.pan_points if p[0] < loop_sample],
                    wave_points=[p for p in note.wave_points if p[0] < loop_sample],
                    op_tl=note.op_tl,
                    op_tl_points=[p for p in note.op_tl_points if p[0] < loop_sample],
                    pitch_env=None,
                    gain_env=None,
                    tag=note.tag,
                )
                post = Note(
                    start=loop_sample,
                    end=end,
                    pitch=_value_at(note.pitch_points, note.pitch, loop_sample),
                    voice=note.voice,
                    gain=_value_at(note.gain_points, note.gain, loop_sample),
                    pan=_value_at(note.pan_points, note.pan, loop_sample),
                    pitch_points=[p for p in note.pitch_points if p[0] > loop_sample],
                    gain_points=[p for p in note.gain_points if p[0] > loop_sample],
                    pan_points=[p for p in note.pan_points if p[0] > loop_sample],
                    wave_points=[p for p in note.wave_points if p[0] > loop_sample],
                    op_tl=tuple(
                        (op, v) for op, v in note.op_tl
                    ),
                    op_tl_points=[p for p in note.op_tl_points if p[0] > loop_sample],
                    pitch_env=None,
                    gain_env=None,
                    tag=note.tag,
                )
                if isinstance(post.voice, SimpleVoice):
                    wave = note.voice.wave if isinstance(note.voice, SimpleVoice) else 0
                    for t, value in note.wave_points:
                        if t <= loop_sample:
                            wave = value
                    post.voice = SimpleVoice(
                        post.voice.module, wave, post.voice.name
                    )
                if post.op_tl:
                    levels = dict(note.op_tl)
                    for t, operator, value in note.op_tl_points:
                        if t <= loop_sample:
                            levels[operator] = value
                    post.op_tl = tuple(sorted(levels.items()))
                out.extend((intro, post))
                changed = True
                self.stats.loop_split += 1
            else:
                out.append(note)
        if changed:
            track.timeline.notes = sorted(out, key=lambda n: (n.start, n.end or n.start))
        return loop_tick

    def _cursor_after(self, notes: list[Note]) -> int:
        bpm = self.song.bpm
        end = 0
        for note in notes:
            if note.end is not None:
                end = max(end, ticks_from_samples(note.end, bpm))
        return end

    # ------------------------------------------------------------------ notes
    def _emit_notes(
        self,
        track: Track,
        notes: list[Note],
        start_tick: int,
        state: "_TrackState",
        tokens: list[str],
    ) -> None:
        bpm = self.song.bpm
        cursor = start_tick
        for note in sorted(notes, key=lambda n: (n.start, n.end or n.start)):
            start = ticks_from_samples(note.start, bpm)
            if note.end is None:
                end = ticks_from_samples(self.song.total_samples, bpm)
            else:
                end = ticks_from_samples(note.end, bpm)
            end = min(end, ticks_from_samples(self.song.total_samples, bpm))
            if end <= start:
                continue
            if start < cursor:
                start = cursor
            if start > cursor:
                tokens.append("r" + duration_token(start - cursor))
                self.stats.rests += 1
            self._emit_note(track, note, start, end, state, tokens)
            cursor = max(cursor, end)
        # Leave the cursor at the end of the last note; the caller pads to the
        # loop/end point when needed.
        state.cursor = cursor

    def _emit_note(
        self,
        track: Track,
        note: Note,
        start: int,
        end: int,
        state: "_TrackState",
        tokens: list[str],
    ) -> None:
        bpm = self.song.bpm
        # Collect control points, keeping only the ones strictly inside the
        # gate and quantising them to the same tick grid as the note.
        points: list[tuple[int, str, object]] = []
        if note.pitch_env is None:
            for t, value in note.pitch_points:
                if note.start < t < (note.end if note.end is not None else self.song.total_samples):
                    tick = ticks_from_samples(t, bpm)
                    if start < tick < end:
                        points.append((tick, "pitch", value))
        if note.gain_env is None:
            for t, value in note.gain_points:
                if note.start < t < (note.end if note.end is not None else self.song.total_samples):
                    tick = ticks_from_samples(t, bpm)
                    if start < tick < end:
                        points.append((tick, "gain", value))
        for t, value in note.pan_points:
            if note.start < t < (note.end if note.end is not None else self.song.total_samples):
                tick = ticks_from_samples(t, bpm)
                if start < tick < end:
                    points.append((tick, "pan", value))
        for t, value in note.wave_points:
            if note.start < t < (note.end if note.end is not None else self.song.total_samples):
                tick = ticks_from_samples(t, bpm)
                if start < tick < end:
                    points.append((tick, "wave", value))
        for t, operator, value in note.op_tl_points:
            if note.start < t < (note.end if note.end is not None else self.song.total_samples):
                tick = ticks_from_samples(t, bpm)
                if start < tick < end:
                    points.append((tick, "optl", (operator, value)))
        points.sort(key=lambda item: item[0])

        # Merge same-tick points; keep a deterministic order.
        merged: list[tuple[int, list[tuple[str, object]]]] = []
        for tick, kind, value in points:
            if merged and merged[-1][0] == tick:
                merged[-1][1].append((kind, value))
            else:
                merged.append((tick, [(kind, value)]))

        boundaries: list[tuple[int, list[tuple[str, object]]]] = [(start, [])]
        boundaries.extend(merged)
        boundaries.append((end, []))
        boundaries = [b for b in boundaries if b[0] >= start]
        # Drop a zero-length duplicate at the gate.
        boundaries = [b for b in boundaries if b[0] < end] + [(end, [])]

        current_tick = start
        current_pitch = note.pitch
        current_gain = note.gain
        current_pan = note.pan
        current_op_tl = dict(note.op_tl)
        current_wave = None
        first_segment = True
        for index, (tick, changes) in enumerate(boundaries[:-1]):
            next_tick = boundaries[index + 1][0]
            duration = next_tick - tick
            if duration <= 0:
                continue
            for kind, value in changes:
                if kind == "pitch":
                    current_pitch = float(value)
                elif kind == "gain":
                    current_gain = float(value)
                elif kind == "pan":
                    current_pan = float(value)
                elif kind == "wave":
                    current_wave = int(value)
                elif kind == "optl":
                    operator, level = value
                    current_op_tl[int(operator)] = int(level)

            if first_segment:
                self._emit_voice(track, note, state, tokens)
                self._emit_operator_levels(
                    note.op_tl, state, tokens, initialize=True
                )
                self._emit_envelopes(note, state, tokens)
                self._emit_lfo(track, note.start, state, tokens)
                if note.gain_env is None:
                    self._emit_gain(current_gain, state, tokens)
                self._emit_pan(current_pan, state, tokens)
            else:
                if not tokens or tokens[-1] != "&":
                    tokens.append("&")
                self._emit_operator_levels(
                    tuple(current_op_tl.items()), state, tokens,
                    initialize=False,
                    only=[value[0] for kind, value in changes if kind == "optl"],
                )
                if current_wave is not None:
                    self._emit_wave(current_wave, state, tokens)
                if note.gain_env is None:
                    self._emit_gain(current_gain, state, tokens)
                self._emit_pan(current_pan, state, tokens)
                self.stats.tied_segments += 1

            self._emit_pitch_note(current_pitch, duration, state, tokens)
            self.stats.notes += 1
            first_segment = False
            current_tick = next_tick
        state.cursor = end

    def _emit_voice(
        self,
        track: Track,
        note: Note,
        state: "_TrackState",
        tokens: list[str],
    ) -> None:
        voice = note.voice or track.default_voice
        if voice is None:
            return
        if isinstance(voice, FmVoice):
            idx = self.song.voices.add(voice)
            token = f"@{idx}"
        elif isinstance(voice, SimpleVoice):
            token = f"@{voice.wave}"
        else:
            return
        if token != state.module_voice:
            tokens.append(token)
            state.module_voice = token
            state.op_tl.clear()

    def _emit_envelopes(
        self, note: Note, state: "_TrackState", tokens: list[str]
    ) -> None:
        pitch = note.pitch_env
        old_pitch = state.pitch_env
        if pitch is None:
            if old_pitch is not None:
                tokens.append(f"{old_pitch[:2]}255")
                state.pitch_env = None
                state.detune = None
        else:
            if old_pitch is not None and old_pitch[:2] != pitch.command:
                tokens.append(f"{old_pitch[:2]}255")
            token = pitch.token
            if old_pitch != token:
                tokens.append(token)
                state.pitch_env = token
                # The table controls pitch; force a base k after it ends.
                state.detune = None
        gain = note.gain_env
        old_gain = state.gain_env
        if gain is None:
            if old_gain is not None:
                tokens.append("na255")
                state.gain_env = None
                state.gain = None
        else:
            token = gain.token
            if old_gain != token:
                tokens.append(token)
                state.gain_env = token
                state.gain = None

    def _emit_operator_levels(
        self,
        levels,
        state: "_TrackState",
        tokens: list[str],
        *,
        initialize: bool,
        only=None,
    ) -> None:
        allowed = None if only is None else set(only)
        for operator, value in levels:
            if allowed is not None and operator not in allowed:
                continue
            key = (operator, value)
            if state.op_tl.get(operator) == value:
                continue
            tokens.append(f"i{operator}@tl{value}")
            state.op_tl[operator] = value

    def _emit_wave(self, index: int, state: "_TrackState", tokens: list[str]) -> None:
        token = f"@{index}"
        if token != state.module_voice:
            tokens.append(token)
            state.module_voice = token
            state.op_tl.clear()

    def _emit_lfo(
        self, track: Track, sample: int, state: "_TrackState", tokens: list[str]
    ) -> None:
        if track.kind != "fm":
            return
        lfo = track.lfo_at(sample)
        if lfo is None:
            if state.lfo_active:
                tokens.extend(["mp0", "ma0"])
                state.lfo_active = False
            return
        if not lfo.active:
            if state.lfo_active:
                tokens.extend(["mp0", "ma0"])
                state.lfo_active = False
            return
        if state.lfo is None or (state.lfo.period, state.lfo.wave) != (lfo.period, lfo.wave):
            tokens.append(f"@lfo{lfo.period},{lfo.wave}")
            state.lfo = lfo
            # @lfo initializes mp/ma to 0, so re-state the depths below.
            state.lfo_pitch = 0
            state.lfo_amp = 0
        if state.lfo_pitch != lfo.pitch:
            tokens.append(f"mp{lfo.pitch}")
            state.lfo_pitch = lfo.pitch
        if state.lfo_amp != lfo.amp:
            tokens.append(f"ma{lfo.amp}")
            state.lfo_amp = lfo.amp
        state.lfo_active = True

    def _emit_gain(self, gain: float, state: "_TrackState", tokens: list[str]) -> None:
        value = _gain_value(gain)
        if value != state.gain:
            tokens.append(f"x{value}")
            state.gain = value

    def _emit_pan(self, pan: float, state: "_TrackState", tokens: list[str]) -> None:
        value = _pan_value(pan)
        if value != state.pan:
            tokens.append(f"p{value}")
            state.pan = value

    def _emit_pitch_note(
        self, pitch: float, duration: int, state: "_TrackState", tokens: list[str]
    ) -> None:
        # Extractors store pitch in SiON's 1/64-semitone units, not MIDI.
        units = int(round(pitch))
        if units > 119 * 64 + 63:
            self.stats.clamped += 1
        note, detune = units_to_note_k(units)
        octave = note // 12
        name = NOTE_NAMES[note % 12]
        if octave != state.octave:
            tokens.append(f"o{octave}")
            state.octave = octave
        if detune != state.detune:
            tokens.append(f"k{detune}")
            state.detune = detune
        tokens.append(name + duration_token(duration))

    # ------------------------------------------------------------------ layout
    def _wrap(self, tokens: list[str]) -> str:
        lines: list[str] = []
        current = ""
        for token in tokens:
            if not current:
                current = token
            elif len(current) + 1 + len(token) <= MAX_LINE:
                current += " " + token
            else:
                lines.append(current)
                current = token
        if current:
            lines.append(current)
        if not lines:
            lines = ["r1"]
        lines[-1] += ";"
        return "\n".join(lines)


class _TrackState:
    __slots__ = (
        "module_voice",
        "octave",
        "detune",
        "gain",
        "pan",
        "lfo",
        "lfo_pitch",
        "lfo_amp",
        "lfo_active",
        "op_tl",
        "pitch_env",
        "gain_env",
        "cursor",
    )

    def __init__(self) -> None:
        self.module_voice: str | None = None
        self.octave = 5
        self.detune = 0
        self.gain = 128
        self.pan = 4
        self.lfo: LfoState | None = None
        self.lfo_pitch = 0
        self.lfo_amp = 0
        self.lfo_active = False
        self.op_tl: dict[int, int] = {}
        self.pitch_env: str | None = None
        self.gain_env: str | None = None
        self.cursor = 0


def _value_at(points: list[tuple[int, float]], default: float, sample: int) -> float:
    value = default
    for t, v in points:
        if t <= sample:
            value = v
        else:
            break
    return value


def render_song(song: Song) -> str:
    return MmlRenderer(song).render()
