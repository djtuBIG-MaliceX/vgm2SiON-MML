"""Game Boy DMG (LR35902) APU decoder: pulse, wave and noise."""

from __future__ import annotations

from dataclasses import dataclass

from ..model import ChannelTimeline, SimpleVoice, Track, WaveTableLibrary
from ..pitch import freq_to_midi, gb_noise_note, gb_pulse_freq, gb_wave_freq, midi_to_units
from ..vgm import VgmFile
from .common import ExtractResult, sorted_writes

GB_CHANNELS = 4
GB_REFIT = 735
GB_ENV_STEP = 689  # 44100 / 64 Hz frame sequencer
GB_DUTY = (2, 4, 8, 12)  # %8 pulse widths: 12.5, 25, 50, 75%

NR10, NR11, NR12, NR13, NR14 = 0x00, 0x01, 0x02, 0x03, 0x04
NR21, NR22, NR23, NR24 = 0x06, 0x07, 0x08, 0x09
NR30, NR31, NR32, NR33, NR34 = 0x0A, 0x0B, 0x0C, 0x0D, 0x0E
NR41, NR42, NR43, NR44 = 0x10, 0x11, 0x12, 0x13
NR50, NR51, NR52 = 0x14, 0x15, 0x16
WAVE_RAM = 0x20
TRIGGERS = (NR14, NR24, NR34, NR44)


@dataclass
class _Voice:
    index: int
    duty: int = 0
    env: int = 0
    period: int = 0
    noise: int = 0
    trigger: bool = False
    note: object = None
    refit_until: int = 0
    env_value: int = 0
    env_dir: int = 0
    env_period: int = 0
    env_next: int = 0


def _is_trigger(w) -> bool:
    return (w.reg & 0x7F) in TRIGGERS and bool(w.val & 0x80)


def _pan_for(mask: int, voice: int) -> tuple[float, float]:
    left = bool(mask & (0x10 << voice))
    right = bool(mask & (1 << voice))
    if left and right:
        return 4.0, 1.0
    if left:
        return 0.0, 1.0
    if right:
        return 8.0, 1.0
    return 4.0, 0.0


def _wave_image(image: bytearray) -> tuple[int, ...]:
    # DMG wave RAM is 32 unsigned 4-bit samples.  Center at 8 and scale to
    # SiON's signed 8-bit custom table.
    return tuple(max(-128, min(127, ((v & 0x0F) - 8) * 16)) for v in image)


def extract_gb(
    vgm: VgmFile,
    waves: WaveTableLibrary,
    *,
    instance: int = 0,
    order_base: int = 0,
    label: str | None = None,
) -> ExtractResult:
    result = ExtractResult()
    writes = sorted_writes(vgm, "gb", instance)
    if not writes:
        return result
    clock = vgm.clock("gb", instance) or 4_194_304
    timelines = [
        ChannelTimeline(f"{label or 'Game Boy'} {instance + 1} ch {v + 1}")
        for v in range(GB_CHANNELS)
    ]
    voices = [_Voice(i) for i in range(GB_CHANNELS)]
    wave = bytearray(16)
    wave_dirty_start: int | None = None
    wave_index = 0
    wave_dac = False
    wave_vol = 0
    wave_triggered = False
    pan = 0xFF
    power = True
    stats = {"notes": 0, "wave_changes": 0, "noise_notes": 0}

    def pan_gain(v: int) -> tuple[float, float]:
        return _pan_for(pan, v)

    def wave_image() -> tuple[int, ...]:
        return _wave_image(
            bytearray(
                (wave[i // 2] >> (4 if i % 2 == 0 else 0)) & 0x0F
                for i in range(32)
            )
        )

    def settle_wave(sample: int) -> None:
        nonlocal wave_dirty_start, wave_index
        if wave_dirty_start is None:
            return
        started = wave_dirty_start
        wave_dirty_start = None
        image = wave_image()
        wave_index = waves.add(image)
        note = voices[2].note
        if note is None:
            return
        if started <= voices[2].refit_until:
            note.voice = SimpleVoice(4, wave_index, "GB wave")
        elif not note.wave_points or note.wave_points[-1][1] != wave_index:
            note.add_wave(started, wave_index)
            stats["wave_changes"] += 1

    def close(v: int, sample: int) -> None:
        note = voices[v].note
        if note is None:
            return
        note.close(sample)
        voices[v].note = None

    def start_pulse(v: int, sample: int) -> None:
        voice = voices[v]
        if not power or not (voice.env & 0xF8):
            return
        volume = voice.env >> 4
        if volume == 0 and not (voice.env & 0x08):
            return
        gain, p = pan_gain(v)
        note = timelines[v].note_on(
            sample,
            midi_to_units(freq_to_midi(gb_pulse_freq(voice.period, clock))),
            SimpleVoice(8, GB_DUTY[voice.duty & 3], "GB pulse"),
            gain=gain * volume / 15.0,
            pan=p,
            tag=f"GB pulse {v + 1}",
        )
        voice.note = note
        voice.refit_until = sample + GB_REFIT
        voice.env_value = volume
        voice.env_dir = (voice.env >> 3) & 1
        voice.env_period = voice.env & 7
        voice.env_next = sample + max(1, voice.env_period) * GB_ENV_STEP
        stats["notes"] += 1

    def start_wave(sample: int) -> None:
        voice = voices[2]
        if not power or not wave_dac or wave_vol == 0:
            return
        idx = waves.add(wave_image())
        gain, p = pan_gain(2)
        note = timelines[2].note_on(
            sample,
            midi_to_units(freq_to_midi(gb_wave_freq(voice.period, clock))),
            SimpleVoice(4, idx, "GB wave"),
            gain=gain * (15 if wave_vol == 1 else 10 if wave_vol == 2 else 5) / 15.0,
            pan=p,
            tag="GB wave",
        )
        voice.note = note
        voice.refit_until = sample + GB_REFIT
        stats["notes"] += 1

    def start_noise(sample: int) -> None:
        voice = voices[3]
        if not power or not (voice.env & 0xF8):
            return
        volume = voice.env >> 4
        if volume == 0 and not (voice.env & 0x08):
            return
        gain, p = pan_gain(3)
        note = timelines[3].note_on(
            sample,
            gb_noise_note(voice.noise) * 64,
            SimpleVoice(2, 2, "GB noise"),
            gain=gain * volume / 15.0,
            pan=p,
            tag="GB noise",
        )
        voice.note = note
        voice.refit_until = sample + GB_REFIT
        voice.env_value = volume
        voice.env_dir = (voice.env >> 3) & 1
        voice.env_period = voice.env & 7
        voice.env_next = sample + max(1, voice.env_period) * GB_ENV_STEP
        stats["noise_notes"] += 1

    def envelope(v: int, sample: int) -> None:
        voice = voices[v]
        note = voice.note
        if note is None or not voice.env_period:
            return
        guard = 0
        while voice.env_next <= sample and guard < 256:
            guard += 1
            tick = int(round(voice.env_next))
            voice.env_next += voice.env_period * GB_ENV_STEP
            value = voice.env_value
            if voice.env_dir and value < 15:
                value += 1
            elif not voice.env_dir and value > 0:
                value -= 1
            else:
                continue
            if value != voice.env_value:
                voice.env_value = value
                note.add_gain(tick, value / 15.0)

    def trigger(v: int, sample: int) -> None:
        close(v, sample)
        if v == 2:
            start_wave(sample)
        elif v == 3:
            start_noise(sample)
        else:
            start_pulse(v, sample)

    def update_pitch(v: int, sample: int) -> None:
        voice = voices[v]
        note = voice.note
        if note is None:
            return
        freq = gb_wave_freq(voice.period, clock) if v == 2 else gb_pulse_freq(voice.period, clock)
        pitch = midi_to_units(freq_to_midi(freq))
        if sample <= voice.refit_until:
            note.pitch = pitch
        else:
            note.add_pitch(sample, pitch)

    # Same-sample trigger bytes are applied last, so the register state that
    # the chip sees is the one preceding the trigger (and the note snapshot is
    # the one the driver intended).
    ordered = sorted(
        writes,
        key=lambda w: (w.sample, 1 if _is_trigger(w) else 0),
    )
    for w in ordered:
        reg = w.reg & 0x7F
        val = w.val & 0xFF
        sample = w.sample
        if not (WAVE_RAM <= reg <= WAVE_RAM + 0x0F):
            settle_wave(sample)
        if reg == NR50:
            continue
        if reg == NR51:
            pan = val
            for v in range(GB_CHANNELS):
                gain, p = pan_gain(v)
                note = voices[v].note
                if note is not None:
                    if sample <= voices[v].refit_until:
                        note.gain = note.gain
                        note.pan = p
                    else:
                        note.add_pan(sample, p)
            continue
        if reg == NR52:
            power = bool(val & 0x80)
            if not power:
                for v in range(GB_CHANNELS):
                    close(v, sample)
            continue
        if reg in (NR10, NR30):
            if reg == NR30:
                old = wave_dac
                wave_dac = bool(val & 0x80)
                if old and not wave_dac:
                    close(2, sample)
                    wave_triggered = False
            continue
        if reg in (NR11, NR21):
            v = 0 if reg == NR11 else 1
            voices[v].duty = (val >> 6) & 3
            note = voices[v].note
            if note is not None:
                duty_wave = GB_DUTY[voices[v].duty]
                if sample <= voices[v].refit_until:
                    note.voice = SimpleVoice(8, duty_wave, "GB pulse")
                else:
                    note.add_wave(sample, duty_wave)
            continue
        if reg in (NR12, NR22, NR42):
            v = 0 if reg == NR12 else 1 if reg == NR22 else 3
            voices[v].env = val
            note = voices[v].note
            if note is not None:
                gain, _p = pan_gain(v)
                vol = (val >> 4) & 0x0F
                note.add_gain(sample, gain * vol / 15.0)
                voices[v].env_value = vol
                voices[v].env_dir = (val >> 3) & 1
                voices[v].env_period = val & 7
            continue
        if reg in (NR13, NR23, NR33):
            v = 0 if reg == NR13 else 1 if reg == NR23 else 2
            voices[v].period = (voices[v].period & 0x700) | val
            update_pitch(v, sample)
            continue
        if reg in (NR14, NR24, NR34):
            v = 0 if reg == NR14 else 1 if reg == NR24 else 2
            voices[v].period = (voices[v].period & 0xFF) | ((val & 7) << 8)
            update_pitch(v, sample)
            if val & 0x80:
                if v == 2:
                    wave_triggered = True
                trigger(v, sample)
            continue
        if reg == NR32:
            wave_vol = (val >> 5) & 3
            note = voices[2].note
            if note is not None:
                gain, _p = pan_gain(2)
                note.add_gain(sample, gain * (15 if wave_vol == 1 else 10 if wave_vol == 2 else 5) / 15.0)
            elif wave_dac and wave_triggered and voices[2].note is None:
                start_wave(sample)
            continue
        if reg in (NR41, NR42, NR43, NR44):
            v = 3
            if reg == NR41:
                continue
            if reg == NR42:
                voices[v].env = val
            elif reg == NR43:
                voices[v].noise = val
            elif reg == NR44 and (val & 0x80):
                trigger(v, sample)
            continue
        if WAVE_RAM <= reg <= WAVE_RAM + 0x0F:
            idx = (reg - WAVE_RAM) * 2
            wave[idx // 2] = ((val >> 4) << 4) | (val & 0x0F)
            if wave_dirty_start is None:
                wave_dirty_start = sample
            continue

    settle_wave(vgm.total_samples)
    for v in range(GB_CHANNELS):
        envelope(v, vgm.total_samples)
        close(v, vgm.total_samples)
        tl = timelines[v]
        if tl.notes:
            if v == 2:
                voice = SimpleVoice(4, 0, "GB wave")
            elif v == 3:
                voice = SimpleVoice(2, 2, "GB noise")
            else:
                voice = SimpleVoice(8, 4, "GB pulse")
            result.tracks.append(
                Track(
                    label=tl.label,
                    module=4 if v == 2 else (2 if v == 3 else 8),
                    kind="wavetable" if v == 2 else "psg",
                    timeline=tl,
                    default_voice=voice,
                    order=order_base + v,
                )
            )
    result.stats.update(stats)
    return result
