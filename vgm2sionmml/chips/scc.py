"""Konami K051649 / K052539 (SCC / SCC+) wavetable decoder."""

from __future__ import annotations

from dataclasses import dataclass

from ..model import ChannelTimeline, SimpleVoice, Track, WaveTableLibrary
from ..pitch import freq_to_midi, midi_to_units, scc_freq
from ..vgm import VgmFile
from .common import ExtractResult, sorted_writes

SCC_VOICES = 5
SCC_WAVE_LEN = 32
SCC_MIN_PERIOD = 9
SCC_REFIT = 735
WAVE_BURST = 64


def _signed(raw) -> tuple[int, ...]:
    return tuple(v if v < 0x80 else v - 0x100 for v in raw)


def _has_signal(wave: tuple[int, ...]) -> bool:
    return any(wave)


@dataclass
class _Voice:
    wave: bytearray
    freq: int = 0
    vol: int = 0
    enabled: bool = False
    note: object = None
    refit_until: int = 0
    dirty_start: int | None = None


def extract_scc(
    vgm: VgmFile,
    waves: WaveTableLibrary,
    *,
    instance: int = 0,
    order_base: int = 0,
    label: str | None = None,
) -> ExtractResult:
    result = ExtractResult()
    writes = sorted_writes(vgm, "k051649", instance)
    if not writes:
        return result
    clock = vgm.clock("k051649", instance) or 1_789_772
    plus = bool(vgm.chip_flags.get("k051649_plus"))
    timelines = [
        ChannelTimeline(f"{label or 'SCC'} {instance + 1} ch {v + 1}")
        for v in range(SCC_VOICES)
    ]
    voices = [_Voice(bytearray(SCC_WAVE_LEN)) for _ in range(SCC_VOICES)]
    gate = [False] * SCC_VOICES
    test = 0
    stats = {"notes": 0, "wave_changes": 0, "glitches": 0}

    def index_for(v: int) -> int:
        return waves.add(_signed(voices[v].wave))

    def settle(v: int, sample: int) -> None:
        voice = voices[v]
        if voice.dirty_start is None:
            return
        started = voice.dirty_start
        voice.dirty_start = None
        note = voice.note
        if note is None:
            return
        image = _signed(voice.wave)
        if not _has_signal(image):
            return
        idx = waves.add(image)
        if started <= voice.refit_until:
            note.voice = SimpleVoice(4, idx, "SCC wave")
            return
        if note.wave_points and note.wave_points[-1][1] == idx:
            return
        note.add_wave(started, idx)
        stats["wave_changes"] += 1

    def settle_all(sample: int) -> None:
        for v in range(SCC_VOICES):
            settle(v, sample)

    def close(v: int, sample: int) -> None:
        settle(v, sample)
        note = voices[v].note
        if note is None:
            return
        note.close(sample)
        voices[v].note = None

    def start(v: int, sample: int) -> None:
        voice = voices[v]
        if voice.note is not None or voice.freq < SCC_MIN_PERIOD:
            return
        image = _signed(voice.wave)
        if not _has_signal(image):
            return
        idx = waves.add(image)
        note = timelines[v].note_on(
            sample,
            midi_to_units(freq_to_midi(scc_freq(voice.freq, clock))),
            SimpleVoice(4, idx, "SCC wave"),
            gain=voice.vol / 15.0,
            pan=4.0,
            tag=f"SCC {v + 1}",
        )
        voice.note = note
        voice.refit_until = sample + SCC_REFIT
        stats["notes"] += 1

    def update_gate(v: int, sample: int) -> None:
        on = gate[v] and voices[v].vol > 0 and voices[v].freq >= SCC_MIN_PERIOD
        if on and voices[v].note is None:
            start(v, sample)
        elif not on and voices[v].note is not None:
            close(v, sample)

    for w in writes:
        bank = w.reg >> 8
        addr = w.reg & 0xFF
        val = w.val & 0xFF
        sample = w.sample
        if bank in (0x0, 0x4):
            # A wave-RAM byte is part of a burst; do not allocate a table
            # until another register write (or the gate) ends the burst.
            if test & 0x40:
                continue
            if bank == 0x4 and not plus:
                addr = 0x60 | (addr & 0x1F)
            if plus and bank == 0x4:
                if not 0x80 <= addr <= 0x9F:
                    continue
                targets = (4,)
            elif plus:
                if addr >= 0x80:
                    continue
                targets = (addr >> 5,)
            elif addr >= 0x60:
                targets = (3, 4)
            else:
                targets = (addr >> 5,)
            for v in targets:
                voice = voices[v]
                if voice.dirty_start is None:
                    voice.dirty_start = sample
                voice.wave[addr & 0x1F] = val
            continue

        settle_all(sample)
        if bank == 0x1:
            v = (addr >> 1) % SCC_VOICES
            voice = voices[v]
            if addr & 1:
                voice.freq = (voice.freq & 0x0FF) | ((val & 0x0F) << 8)
            else:
                voice.freq = (voice.freq & 0xF00) | val
            note = voice.note
            if note is not None:
                pitch = midi_to_units(freq_to_midi(scc_freq(voice.freq, clock)))
                if sample <= voice.refit_until:
                    note.pitch = pitch
                elif voice.freq >= SCC_MIN_PERIOD:
                    note.add_pitch(sample, pitch)
            update_gate(v, sample)
        elif bank == 0x2:
            v = addr % SCC_VOICES
            voice = voices[v]
            voice.vol = val & 0x0F
            note = voice.note
            if note is not None and voice.vol > 0:
                gain = voice.vol / 15.0
                if sample <= voice.refit_until:
                    note.gain = gain
                else:
                    note.add_gain(sample, gain)
            update_gate(v, sample)
        elif bank == 0x3:
            for v in range(SCC_VOICES):
                gate[v] = bool(val & (1 << v))
            for v in range(SCC_VOICES):
                update_gate(v, sample)
        elif bank == 0x5:
            test = val

    end = vgm.total_samples
    for v in range(SCC_VOICES):
        close(v, end)
        tl = timelines[v]
        if tl.notes:
            result.tracks.append(
                Track(
                    label=tl.label,
                    module=4,
                    kind="wavetable",
                    timeline=tl,
                    default_voice=SimpleVoice(4, 0, "SCC wave"),
                    order=order_base + v,
                )
            )
    result.stats.update(stats)
    if stats["wave_changes"]:
        result.stats["wave_changes"] = stats["wave_changes"]
    return result
