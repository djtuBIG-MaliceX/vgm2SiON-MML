"""Musical model shared by the chip extractors and the MML writer."""

from __future__ import annotations

from dataclasses import dataclass, field

FAMILY_COMMAND = {
    "opm": "#OPM@",
    "opn": "#OPN@",
    "opl": "#OPL@",
}

FAMILY_MODULE = {
    "opm": 6,
    "opn": 6,
    "opl": 6,
}

# Operator parameter counts in the SiON voice syntaxes.
FAMILY_OP_PARAMS = {"opm": 11, "opn": 10, "opl": 11}


@dataclass(frozen=True)
class FmVoice:
    """One FM timbre in SiON's #OPM@ / #OPN@ / #OPL@ parameter layout."""

    family: str
    alg: int
    fb: int
    ops: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        expected = FAMILY_OP_PARAMS[self.family]
        for op in self.ops:
            if len(op) != expected:
                raise ValueError(
                    f"{self.family} operator needs {expected} parameters, got {len(op)}"
                )
        if not 1 <= len(self.ops) <= 4:
            raise ValueError("an FM voice needs 1-4 operators")

    @property
    def key(self) -> tuple:
        return (self.family, self.alg, self.fb, self.ops)

    @property
    def n_ops(self) -> int:
        return len(self.ops)

    def mml_params(self) -> list[int]:
        out: list[int] = [self.alg, self.fb]
        for op in self.ops:
            out.extend(op)
        return out

    def mml(self, index: int) -> str:
        command = FAMILY_COMMAND[self.family]
        per_op = FAMILY_OP_PARAMS[self.family]
        values = self.mml_params()
        head = f"{command}{index}{{"
        lines = [head, f"{values[0]}, {values[1]};"]
        pos = 2
        for _ in range(self.n_ops):
            chunk = values[pos : pos + per_op]
            pos += per_op
            lines.append(", ".join(str(v) for v in chunk) + ";")
        lines.append("};")
        return "\n".join(lines)


@dataclass(frozen=True)
class SimpleVoice:
    """A PSG/NES wave selection (``%0@0``, ``%1@4`` ...)."""

    module: int
    wave: int
    name: str = ""

    @property
    def key(self) -> tuple:
        return (self.module, self.wave)


@dataclass(frozen=True)
class WaveTable:
    """One custom SiON wavetable (``#WAVB``), signed 8-bit samples."""

    samples: tuple[int, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.samples) <= 256:
            raise ValueError("a SiON wavetable needs 1-256 samples")
        for value in self.samples:
            if not -128 <= value <= 127:
                raise ValueError("wavetable samples must be signed 8-bit")

    @property
    def key(self) -> tuple[int, ...]:
        return self.samples

    def mml(self, index: int) -> str:
        data = "".join(f"{value & 0xFF:02x}" for value in self.samples)
        return f"#WAVB{index}{{{data}}};"


class WaveTableLibrary:
    """Deduplicated wavetables, capped at SiON's 256 SCC slots."""

    def __init__(self, limit: int = 256) -> None:
        self.limit = limit
        self.tables: list[WaveTable] = []
        self._index: dict[tuple[int, ...], int] = {}
        self.overflow = 0

    def add(self, samples) -> int:
        table = WaveTable(tuple(int(v) for v in samples))
        hit = self._index.get(table.key)
        if hit is not None:
            return hit
        if len(self.tables) < self.limit:
            idx = len(self.tables)
            self.tables.append(table)
            self._index[table.key] = idx
            return idx
        # Keep the most plausible timbre rather than dropping the note.
        self.overflow += 1
        best = 0
        best_err = None
        for i, candidate in enumerate(self.tables):
            err = sum(
                (a - b) * (a - b)
                for a, b in zip(candidate.samples, table.samples)
            )
            err += abs(len(candidate.samples) - len(table.samples)) * 16384
            if best_err is None or err < best_err:
                best, best_err = i, err
        return best

    def get(self, index: int) -> WaveTable | None:
        if 0 <= index < len(self.tables):
            return self.tables[index]
        return None


@dataclass
class Note:
    start: int
    end: int | None
    pitch: float  # SiON 1/64-semitone units
    voice: FmVoice | SimpleVoice | None
    gain: float = 1.0
    pan: float = 4.0
    pitch_points: list[tuple[int, float]] = field(default_factory=list)
    gain_points: list[tuple[int, float]] = field(default_factory=list)
    pan_points: list[tuple[int, float]] = field(default_factory=list)
    # Wavetable index changes during the held note (SCC/HuC6280/VSU morphs).
    wave_points: list[tuple[int, int]] = field(default_factory=list)
    # OPL operator total-level overrides at the note's gate, and later moves.
    op_tl: tuple[tuple[int, int], ...] = ()
    op_tl_points: list[tuple[int, int, int]] = field(default_factory=list)
    # Diagnostic tags (never emitted).
    tag: str = ""

    @property
    def duration(self) -> int:
        return (self.end or self.start) - self.start

    def close(self, sample: int) -> None:
        if self.end is None or sample < self.end:
            self.end = sample

    def add_pitch(self, sample: int, value: float, epsilon: float = 0.01) -> None:
        if value <= 0:
            return
        if abs(value - self.pitch) <= epsilon and not self.pitch_points:
            return
        if self.pitch_points and abs(value - self.pitch_points[-1][1]) <= epsilon:
            return
        self.pitch_points.append((sample, value))

    def add_gain(self, sample: int, value: float, epsilon: float = 0.004) -> None:
        value = max(0.0, min(1.0, value))
        if abs(value - self.gain) <= epsilon and not self.gain_points:
            return
        if self.gain_points and abs(value - self.gain_points[-1][1]) <= epsilon:
            return
        self.gain_points.append((sample, value))

    def add_pan(self, sample: int, value: float, epsilon: float = 0.05) -> None:
        value = max(0.0, min(8.0, value))
        if abs(value - self.pan) <= epsilon and not self.pan_points:
            return
        if self.pan_points and abs(value - self.pan_points[-1][1]) <= epsilon:
            return
        self.pan_points.append((sample, value))

    def add_wave(self, sample: int, index: int) -> None:
        if not self.wave_points or self.wave_points[-1][1] != index:
            self.wave_points.append((sample, index))

    def add_op_tl(self, sample: int, operator: int, value: int) -> None:
        if (
            self.op_tl_points
            and self.op_tl_points[-1][1] == operator
            and self.op_tl_points[-1][2] == value
        ):
            return
        self.op_tl_points.append((sample, operator, value))


class ChannelTimeline:
    """Monophonic note stream for one hardware channel.

    A key-on closes whatever was open (a retrigger), and a key-off closes the
    current note.  Pitch/gain/pan writes while a note is held become control
    points, which the MML writer turns into tied note segments.
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.notes: list[Note] = []
        self.open: Note | None = None

    def note_on(
        self,
        sample: int,
        pitch: float,
        voice: FmVoice | SimpleVoice | None,
        gain: float = 1.0,
        pan: float = 4.0,
        tag: str = "",
    ) -> Note:
        if self.open is not None:
            self.open.close(sample)
            self.open = None
        note = Note(
            start=sample,
            end=None,
            pitch=pitch,
            voice=voice,
            gain=max(0.0, min(1.0, gain)),
            pan=max(0.0, min(8.0, pan)),
            tag=tag or self.label,
        )
        self.notes.append(note)
        self.open = note
        return note

    def note_off(self, sample: int) -> None:
        if self.open is not None:
            self.open.close(sample)
            self.open = None

    def pitch(self, sample: int, value: float) -> None:
        if self.open is not None:
            self.open.add_pitch(sample, value)

    def gain(self, sample: int, value: float) -> None:
        if self.open is not None:
            self.open.add_gain(sample, value)

    def pan(self, sample: int, value: float) -> None:
        if self.open is not None:
            self.open.add_pan(sample, value)

    def wave(self, sample: int, index: int) -> None:
        if self.open is not None:
            self.open.add_wave(sample, index)

    def op_tl(self, sample: int, operator: int, value: int) -> None:
        if self.open is not None:
            self.open.add_op_tl(sample, operator, value)

    def close_all(self, sample: int) -> None:
        if self.open is not None:
            self.open.close(sample)
            self.open = None

    def shift(self, samples: int) -> None:
        """Shift every timestamp; used when one chip instance starts late."""
        if not samples:
            return
        for note in self.notes:
            note.start += samples
            if note.end is not None:
                note.end += samples
            note.pitch_points = [(t + samples, v) for t, v in note.pitch_points]
            note.gain_points = [(t + samples, v) for t, v in note.gain_points]
            note.pan_points = [(t + samples, v) for t, v in note.pan_points]
            note.wave_points = [(t + samples, v) for t, v in note.wave_points]
            note.op_tl_points = [
                (t + samples, op, v) for t, op, v in note.op_tl_points
            ]


@dataclass
class LfoState:
    """SiON's @lfo/mp/ma parameters."""

    period: int = 20
    wave: int = 2
    pitch: int = 0
    amp: int = 0

    @property
    def active(self) -> bool:
        return self.pitch > 0 or self.amp > 0

    @property
    def key(self) -> tuple:
        return (self.period, self.wave, self.pitch, self.amp)


@dataclass
class Track:
    label: str
    module: int
    kind: str  # "fm" | "psg" | "nes"
    timeline: ChannelTimeline
    default_voice: FmVoice | SimpleVoice | None = None
    lfo_points: list[tuple[int, LfoState]] = field(default_factory=list)
    order: int = 0

    def lfo_at(self, sample: int) -> LfoState | None:
        current = None
        for t, state in self.lfo_points:
            if t > sample:
                break
            current = state
        return current

    def sorted_notes(self) -> list[Note]:
        return sorted(self.timeline.notes, key=lambda n: (n.start, n.end or n.start))


class VoiceLibrary:
    """Deduplicated FM voices, capped at SiON's 256 voice slots."""

    def __init__(self, limit: int = 256) -> None:
        self.limit = limit
        self.voices: list[FmVoice] = []
        self._index: dict[tuple, int] = {}
        self.overflow = 0

    def add(self, voice: FmVoice) -> int:
        key = voice.key
        hit = self._index.get(key)
        if hit is not None:
            return hit
        if len(self.voices) < self.limit:
            idx = len(self.voices)
            self.voices.append(voice)
            self._index[key] = idx
            return idx
        # Reuse the closest timbre rather than dropping the note.
        self.overflow += 1
        target = voice.mml_params()
        best = 0
        best_err = None
        for i, candidate in enumerate(self.voices):
            if candidate.family != voice.family:
                continue
            err = sum(abs(a - b) for a, b in zip(candidate.mml_params(), target))
            if best_err is None or err < best_err:
                best, best_err = i, err
        return best

    def by_family(self, family: str) -> list[tuple[int, FmVoice]]:
        return [(i, v) for i, v in enumerate(self.voices) if v.family == family]


@dataclass
class Song:
    """Everything the MML writer needs."""

    title: str
    tracks: list[Track]
    voices: VoiceLibrary
    total_samples: int
    waves: WaveTableLibrary = field(default_factory=WaveTableLibrary)
    loop_sample: int | None = None
    bpm: int = 120
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    def tracks_in_order(self) -> list[Track]:
        return sorted(self.tracks, key=lambda t: t.order)
