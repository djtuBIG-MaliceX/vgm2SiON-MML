"""Hudson HuC6280 (PC Engine) six-voice wavetable decoder."""

from __future__ import annotations

from dataclasses import dataclass

from ..model import ChannelTimeline, SimpleVoice, Track, WaveTableLibrary
from ..pitch import freq_to_midi, huc6280_freq, midi_to_units
from ..vgm import VgmFile
from .common import ExtractResult, sorted_writes

HUC_CHANNELS = 6
HUC_WAVE_LEN = 32
HUC_REFIT = 735
HUC_ATTACK_WINDOW = 64
HUC_MIN_NOTE = 100
WAVE_BURST = 64
BALANCE_SCALE = (
    0x00, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F,
    0x10, 0x13, 0x15, 0x17, 0x19, 0x1B, 0x1D, 0x1F,
)

REG_SELECT = 0x00
REG_GLOBAL_BALANCE = 0x01
REG_FREQ_LO = 0x02
REG_FREQ_HI = 0x03
REG_CONTROL = 0x04
REG_CH_BALANCE = 0x05
REG_WAVE = 0x06
REG_NOISE = 0x07
REG_LFO_FREQ = 0x08
REG_LFO_CTRL = 0x09
CTRL_ENABLE = 0x80
CTRL_DDA = 0x40
CTRL_VOLUME = 0x1F
NOISE_ENABLE = 0x80


def _wave_samples(raw) -> tuple[int, ...]:
    # The PCE DAC is 5-bit unipolar.  Centering at 16 gives signed data in
    # the range SiON's SCC module expects; the sign matches the hardware's
    # 17 - data output.
    return tuple(max(-128, min(127, (16 - (v & 0x1F)) * 8)) for v in raw)


@dataclass
class _Voice:
    wave: bytearray
    widx: int = 0
    freq: int = 0
    control: int = 0
    balance: int = 0
    global_balance: int = 0
    noise: int = 0
    note: object = None
    noise_note: object = None
    refit_until: int = 0
    dirty_start: int | None = None


def _levels(control: int, balance: int, global_balance: int) -> tuple[float, float]:
    """Return (gain, pan) for the three PCE attenuators."""
    al = 0x1F - (control & CTRL_VOLUME)
    ga_l = 0x1F - BALANCE_SCALE[(global_balance >> 4) & 0x0F]
    ga_r = 0x1F - BALANCE_SCALE[global_balance & 0x0F]
    bl = 0x1F - BALANCE_SCALE[(balance >> 4) & 0x0F]
    br = 0x1F - BALANCE_SCALE[balance & 0x0F]
    left = 0x1F - min(0x1F, ga_l + bl + al)
    right = 0x1F - min(0x1F, ga_r + br + al)
    peak = max(left, right)
    if peak <= 0:
        return 0.0, 4.0
    pan = 4.0 + 4.0 * (right - left) / peak
    return peak / 0x1F, pan


def extract_huc6280(
    vgm: VgmFile,
    waves: WaveTableLibrary,
    *,
    instance: int = 0,
    order_base: int = 0,
    label: str | None = None,
) -> ExtractResult:
    result = ExtractResult()
    writes = sorted_writes(vgm, "huc6280", instance)
    if not writes:
        return result
    clock = vgm.clock("huc6280", instance) or 3_579_545
    timelines = [
        ChannelTimeline(f"{label or 'HuC6280'} {instance + 1} ch {v + 1}")
        for v in range(HUC_CHANNELS)
    ]
    noise_timelines = [
        ChannelTimeline(f"{label or 'HuC6280'} {instance + 1} noise {v + 1}")
        for v in range(HUC_CHANNELS)
    ]
    voices = [_Voice(bytearray(HUC_WAVE_LEN)) for _ in range(HUC_CHANNELS)]
    select = 0
    stats = {"notes": 0, "noise_notes": 0, "wave_changes": 0, "glitches": 0}

    def tone_on(v: int) -> bool:
        voice = voices[v]
        if not (voice.control & CTRL_ENABLE) or voice.control & CTRL_DDA:
            return False
        if not (voice.control & CTRL_VOLUME):
            return False
        if v >= 4 and voice.noise & NOISE_ENABLE:
            return False
        return True

    def settle(v: int, sample: int) -> None:
        voice = voices[v]
        if voice.dirty_start is None:
            return
        started = voice.dirty_start
        voice.dirty_start = None
        note = voice.note
        if note is None:
            return
        image = _wave_samples(voice.wave)
        if not any(image):
            return
        idx = waves.add(image)
        if started <= voice.refit_until:
            note.voice = SimpleVoice(4, idx, "PCE wave")
            return
        if note.wave_points and note.wave_points[-1][1] == idx:
            return
        note.add_wave(started, idx)
        stats["wave_changes"] += 1

    def settle_all(sample: int) -> None:
        for v in range(HUC_CHANNELS):
            settle(v, sample)

    def finish_short_note(note, timeline, sample: int) -> bool:
        """Discard driver housekeeping clicks below HUC_MIN_NOTE."""
        if sample - note.start >= HUC_MIN_NOTE:
            return False
        note.end = note.start
        try:
            timeline.notes.remove(note)
        except ValueError:
            pass
        stats["glitches"] += 1
        return True

    def close(v: int, sample: int) -> None:
        settle(v, sample)
        note = voices[v].note
        if note is None:
            return
        note.close(sample)
        finish_short_note(note, timelines[v], sample)
        voices[v].note = None

    def close_noise(v: int, sample: int) -> None:
        note = voices[v].noise_note
        if note is None:
            return
        note.close(sample)
        finish_short_note(note, noise_timelines[v], sample)
        voices[v].noise_note = None

    def start_noise(v: int, sample: int) -> None:
        voice = voices[v]
        if voice.noise_note is not None:
            return
        if not (voice.control & CTRL_ENABLE) or not (
            voice.control & CTRL_VOLUME
        ) or not (voice.noise & NOISE_ENABLE):
            return
        gain, pan = _levels(
            voice.control, voice.balance, voice.global_balance
        )
        note = noise_timelines[v].note_on(
            sample,
            (1 + (31 - (voice.noise & 0x1F))) * 64,
            SimpleVoice(2, 0, "PCE noise"),
            gain=gain,
            pan=pan,
            tag=f"HuC6280 noise {v + 1}",
        )
        voice.noise_note = note
        voice.refit_until = sample + HUC_REFIT
        stats["noise_notes"] += 1

    def start(v: int, sample: int) -> None:
        voice = voices[v]
        if voice.note is not None:
            return
        image = _wave_samples(voice.wave)
        if not any(image):
            return
        idx = waves.add(image)
        gain, pan = _levels(voice.control, voice.balance, voice.global_balance)
        pitch = midi_to_units(freq_to_midi(huc6280_freq(voice.freq, clock)))
        note = timelines[v].note_on(
            sample, pitch, SimpleVoice(4, idx, "PCE wave"),
            gain=gain, pan=pan, tag=f"HuC6280 {v + 1}",
        )
        voice.note = note
        voice.refit_until = sample + HUC_REFIT
        stats["notes"] += 1

    def update_gate(v: int, sample: int) -> None:
        noise_on = (
            v >= 4
            and bool(voices[v].noise & NOISE_ENABLE)
            and bool(voices[v].control & CTRL_ENABLE)
            and bool(voices[v].control & CTRL_VOLUME)
        )
        if noise_on:
            close(v, sample)
            start_noise(v, sample)
        else:
            close_noise(v, sample)
            if tone_on(v) and voices[v].note is None:
                start(v, sample)
            elif not tone_on(v):
                close(v, sample)

    def update_pitch(v: int, sample: int) -> None:
        voice = voices[v]
        if voice.note is not None and not (
            voice.note.end is not None and sample >= voice.note.end
        ):
            pitch = midi_to_units(freq_to_midi(huc6280_freq(voice.freq, clock)))
            if sample < voice.refit_until:
                voice.note.pitch = pitch
            else:
                voice.note.add_pitch(sample, pitch)
        if voice.noise_note is not None and not (
            voice.noise_note.end is not None and sample >= voice.noise_note.end
        ):
            pitch = (1 + (31 - (voice.noise & 0x1F))) * 64
            if sample < voice.refit_until:
                voice.noise_note.pitch = pitch
            else:
                voice.noise_note.add_pitch(sample, pitch)

    def update_levels(v: int, sample: int) -> None:
        voice = voices[v]
        gain, pan = _levels(voice.control, voice.balance, voice.global_balance)
        for note in (voice.note, voice.noise_note):
            if note is None:
                continue
            # A register write at the exact close sample belongs to the next
            # gate, not the note that is ending.  Overwriting the base gain
            # there made short percussion notes silent from their first tick.
            if note.end is not None and sample >= note.end:
                continue
            if sample < note.start + HUC_ATTACK_WINDOW:
                note.gain = gain
                note.pan = pan
            else:
                note.add_gain(sample, gain)
                note.add_pan(sample, pan)

    for w in writes:
        reg = w.reg & 0x7F
        val = w.val & 0xFF
        sample = w.sample
        if reg == REG_SELECT:
            select = val & 0x07
            continue
        if reg == REG_GLOBAL_BALANCE:
            for v in range(HUC_CHANNELS):
                voices[v].global_balance = val
                update_levels(v, sample)
            continue
        if reg in (REG_LFO_FREQ, REG_LFO_CTRL):
            continue
        if select >= HUC_CHANNELS:
            continue
        voice = voices[select]
        if reg == REG_FREQ_LO:
            voice.freq = (voice.freq & 0x0F00) | val
            update_pitch(select, sample)
        elif reg == REG_FREQ_HI:
            voice.freq = (voice.freq & 0x00FF) | ((val & 0x0F) << 8)
            update_pitch(select, sample)
        elif reg == REG_CONTROL:
            old = voice.control
            if (old & CTRL_DDA) and not (val & CTRL_DDA):
                voice.widx = 0
            if not (old & CTRL_ENABLE) and (val & CTRL_ENABLE):
                if not (val & CTRL_DDA):
                    voice.widx = (voice.widx + 1) & 0x1F
            voice.control = val
            update_gate(select, sample)
            update_levels(select, sample)
        elif reg == REG_CH_BALANCE:
            voice.balance = val
            update_levels(select, sample)
        elif reg == REG_WAVE:
            if not (voice.control & CTRL_DDA):
                if voice.dirty_start is None:
                    voice.dirty_start = sample
                voice.wave[voice.widx] = val & 0x1F
                if (voice.control & 0xC0) == 0:
                    voice.widx = (voice.widx + 1) & 0x1F
        elif reg == REG_NOISE:
            voice.noise = val
            update_pitch(select, sample)
            update_gate(select, sample)

    end = vgm.total_samples
    for v in range(HUC_CHANNELS):
        close(v, end)
        close_noise(v, end)
        tl = timelines[v]
        if tl.notes:
            result.tracks.append(
                Track(
                    label=tl.label,
                    module=4,
                    kind="wavetable",
                    timeline=tl,
                    default_voice=SimpleVoice(4, 0, "PCE wave"),
                    order=order_base + v,
                )
            )
        ntl = noise_timelines[v]
        if ntl.notes:
            result.tracks.append(
                Track(
                    label=ntl.label,
                    module=2,
                    kind="psg",
                    timeline=ntl,
                    default_voice=SimpleVoice(2, 0, "PCE noise"),
                    order=order_base + HUC_CHANNELS + v,
                )
            )
    if stats["glitches"]:
        result.warnings.append(
            f"HuC6280: ignored {stats['glitches']} sub-frame tone/noise gates"
        )
    result.stats.update(stats)
    return result
