"""Konami K005289 (Bubble System WSG) two-voice wavetable decoder."""

from __future__ import annotations

from dataclasses import dataclass

from ..model import ChannelTimeline, SimpleVoice, Track, WaveTableLibrary
from ..pitch import freq_to_midi, midi_to_units
from ..vgm import VgmFile
from .common import ExtractResult, sorted_writes

K005289_VOICES = 2
K005289_WAVE_LEN = 32
K005289_DEFAULT_CLOCK = 3_579_545
K005289_REFIT = 735
WAVE_BURST = 64


@dataclass
class _Voice:
    index: int
    latch: int = 0
    period: int = 0
    vol: int = 0
    bank: int = 0
    note: object = None
    refit_until: int = 0


def _wave_samples(prom: bytes, voice: int, bank: int) -> tuple[int, ...]:
    off = voice * 0x100 + (bank & 7) * K005289_WAVE_LEN
    return tuple(
        max(-128, min(127, ((prom[off + i] & 0x0F) - 8) * 16))
        for i in range(K005289_WAVE_LEN)
    )


def extract_k005289(
    vgm: VgmFile,
    waves: WaveTableLibrary,
    *,
    instance: int = 0,
    order_base: int = 0,
    label: str | None = None,
) -> ExtractResult:
    result = ExtractResult()
    writes = sorted_writes(vgm, "k005289", instance)
    if not writes:
        return result
    prom = bytearray([0xFF]) * 0x200
    loaded = False
    for block in vgm.data_blocks:
        if block.block_type != 0xC3:
            continue
        start = max(0, min(0x200, block.start))
        end = min(0x200, start + len(block.data))
        if end > start:
            prom[start:end] = block.data[: end - start]
            loaded = True
    if not loaded:
        result.warnings.append(
            "K005289: no 0xC3 waveform PROM data block was present; "
            "the voices were ignored"
        )
        return result

    clock = vgm.clock("k005289", instance) or K005289_DEFAULT_CLOCK
    timelines = [
        ChannelTimeline(f"{label or 'K005289'} {instance + 1} ch {v + 1}")
        for v in range(K005289_VOICES)
    ]
    voices = [_Voice(i) for i in range(K005289_VOICES)]
    stats = {"notes": 0, "wave_changes": 0}

    def add_wave(v: int) -> int:
        return waves.add(_wave_samples(prom, v, voices[v].bank))

    def close(v: int, sample: int) -> None:
        note = voices[v].note
        if note is None:
            return
        note.close(sample)
        voices[v].note = None

    def start(v: int, sample: int) -> None:
        voice = voices[v]
        if voice.note is not None or voice.vol == 0:
            return
        period = voice.period & 0xFFF
        pitch = midi_to_units(
            freq_to_midi(clock / (32.0 * (period + 1)))
        )
        note = timelines[v].note_on(
            sample,
            pitch,
            SimpleVoice(4, add_wave(v), "K005289 wave"),
            gain=voice.vol / 15.0,
            pan=4.0,
            tag=f"K005289 {v + 1}",
        )
        voice.note = note
        voice.refit_until = sample + K005289_REFIT
        stats["notes"] += 1

    for w in writes:
        off = w.reg & 0x0F
        val = w.val & 0xFFF
        sample = w.sample
        if off in (0x0, 0x1):
            v = off & 1
            voice = voices[v]
            old_vol = voice.vol
            voice.vol = val & 0x0F
            voice.bank = (val >> 5) & 7
            if voice.note is not None:
                if voice.vol == 0:
                    close(v, sample)
                else:
                    note = voice.note
                    idx = add_wave(v)
                    if sample <= voice.refit_until:
                        note.voice = SimpleVoice(4, idx, "K005289 wave")
                    elif not note.wave_points or note.wave_points[-1][1] != idx:
                        note.add_wave(sample, idx)
                        stats["wave_changes"] += 1
                    note.add_gain(sample, voice.vol / 15.0)
            elif old_vol == 0 and voice.vol > 0:
                start(v, sample)
        elif off in (0x2, 0x3):
            v = off & 1
            voices[v].latch = 0xFFF - (val & 0x0FFF)
        elif off in (0x4, 0x5):
            v = off & 1
            voice = voices[v]
            voice.period = voice.latch
            if voice.note is not None:
                period = voice.period & 0xFFF
                pitch = midi_to_units(
                    freq_to_midi(clock / (32.0 * (period + 1)))
                )
                note = voice.note
                if sample <= voice.refit_until:
                    note.pitch = pitch
                else:
                    note.add_pitch(sample, pitch)
            else:
                start(v, sample)

    for v in range(K005289_VOICES):
        close(v, vgm.total_samples)
        tl = timelines[v]
        if tl.notes:
            result.tracks.append(
                Track(
                    label=tl.label,
                    module=4,
                    kind="wavetable",
                    timeline=tl,
                    default_voice=SimpleVoice(4, 0, "K005289 wave"),
                    order=order_base + v,
                )
            )
    result.stats.update(stats)
    return result
