"""NES APU (2A03) decoder.

Pulse 1, pulse 2, triangle and noise are converted.  DPCM writes are recorded
by the VGM reader as ``nes_dmc`` and are intentionally ignored.  The hardware
length counter is used to schedule note ends; envelope decay is written as
``x`` expression points.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model import ChannelTimeline, SimpleVoice, Track
from ..pitch import freq_to_midi, midi_to_units, nes_pulse_freq, nes_triangle_freq, ntsc_noise_note
from ..vgm import VgmFile
from .common import ExtractResult, sorted_writes

# Length counter values, clocked at 240 Hz.
LENGTH_TABLE = (
    10, 254, 20, 2, 40, 4, 80, 6, 160, 8, 60, 10, 14, 12, 26, 14,
    18, 16, 30, 18, 42, 20, 50, 22, 62, 24, 74, 26, 86, 28, 94, 30,
)

DUTY_WAVE = {0: 1, 1: 2, 2: 4, 3: 6}
NES_SAMPLE_RATE = 44100.0


def _length_samples(index: int) -> int:
    index &= 0x1F
    if index == 0:
        return 0
    return int(round(LENGTH_TABLE[index] * NES_SAMPLE_RATE / 240.0))


def _env_gain(level: int) -> float:
    return max(0.0, min(1.0, level / 15.0))


@dataclass
class _NesEnvelope:
    period: int = 0
    loop: bool = False
    step_samples: int = 1
    level: int = 15
    next_sample: int = 0
    active: bool = False

    def reset(self, sample: int, period: int, loop: bool) -> None:
        self.period = period & 0x0F
        self.loop = loop
        self.step_samples = max(1, int(round((self.period + 1) * NES_SAMPLE_RATE / 240.0)))
        self.level = 15
        self.next_sample = sample + self.step_samples
        self.active = True

    def points_until(self, end_sample: int, limit: int = 4096) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        if not self.active:
            return out
        count = 0
        while self.next_sample <= end_sample and count < limit:
            sample = self.next_sample
            if self.level > 0:
                self.level -= 1
            elif self.loop:
                self.level = 15
            else:
                self.active = False
                out.append((sample, 0))
                break
            out.append((sample, self.level))
            self.next_sample = sample + self.step_samples
            count += 1
        if count >= limit:
            self.active = False
        return out


@dataclass
class _NesChannel:
    kind: str  # pulse, triangle, noise
    index: int
    timeline: ChannelTimeline
    duty: int = 0
    env_period: int = 0
    const_volume: bool = True
    volume: int = 0
    length_halt: bool = False
    timer: int = 0
    noise_period: int = 0
    short_noise: bool = False
    linear_control: bool = False
    linear_reload: int = 0
    length_index: int = 0
    env: _NesEnvelope | None = None
    held: bool = False
    note_start: int = 0
    pending_end: int | None = None

    def module(self) -> int:
        # SiON's NES module uses the 16-entry PSG pitch table, which folds and
        # mistunes bass notes.  The MA3 triangle/square generator has the full
        # pitch range, so the triangle uses it instead.
        return 3 if self.kind == "triangle" else 1

    def wave(self) -> int:
        if self.kind == "triangle":
            return 16
        if self.kind == "noise":
            return 10 if self.short_noise else 9
        return DUTY_WAVE.get(self.duty & 3, 4)

    def pitch_units(self, clock: int) -> int:
        if self.kind == "noise":
            return ntsc_noise_note(self.noise_period) * 64
        if self.kind == "triangle":
            midi = freq_to_midi(nes_triangle_freq(self.timer, clock))
            # %3's triangle pitch table is accurate from C2 upward.  Fold
            # lower notes up by octaves; the rendered pitch stays the same.
            while midi < 24.0:
                midi += 12.0
            while midi > 119.0:
                midi -= 12.0
            return midi_to_units(midi)
        return midi_to_units(freq_to_midi(nes_pulse_freq(self.timer, clock)))

    def level(self) -> int:
        if self.kind == "triangle":
            return 15
        if self.env is not None and not self.const_volume:
            return self.env.level
        return self.volume & 0x0F

    def gain(self) -> float:
        return _env_gain(self.level())


class _NesApu:
    def __init__(self, clock: int, label: str, order_base: int) -> None:
        self.clock = clock
        self.label = label
        self.order_base = order_base
        self.channels = [
            _NesChannel("pulse", 0, ChannelTimeline(f"{label} pulse 1")),
            _NesChannel("pulse", 1, ChannelTimeline(f"{label} pulse 2")),
            _NesChannel("triangle", 2, ChannelTimeline(f"{label} triangle")),
            _NesChannel("noise", 3, ChannelTimeline(f"{label} noise")),
        ]

    def _advance(self, sample: int) -> None:
        for ch in self.channels:
            if not ch.held:
                continue
            limit = ch.pending_end if ch.pending_end is not None else sample
            if ch.env is not None and not ch.const_volume:
                for t, level in ch.env.points_until(min(sample, limit)):
                    ch.timeline.gain(t, _env_gain(level))
            if ch.pending_end is not None and sample >= ch.pending_end:
                ch.timeline.note_off(ch.pending_end)
                ch.held = False
                ch.pending_end = None

    def _note_on(self, ch: _NesChannel, sample: int) -> None:
        ch.timeline.note_on(
            sample,
            ch.pitch_units(self.clock),
            SimpleVoice(ch.module(), ch.wave(), "NES"),
            ch.gain(),
            4.0,
        )
        ch.held = True
        ch.note_start = sample
        ch.pending_end = None
        if ch.kind != "triangle" and not ch.const_volume:
            ch.env = _NesEnvelope()
            ch.env.reset(sample, ch.env_period, ch.length_halt)
        else:
            ch.env = None
        length = 0 if ch.length_halt else _length_samples(ch.length_index)
        end = sample + length if length else None
        if ch.kind == "triangle":
            if ch.linear_control:
                lin = int(round(ch.linear_reload * NES_SAMPLE_RATE / 240.0))
                if lin:
                    end = sample + lin if end is None else min(end, sample + lin)
                elif ch.linear_control and ch.linear_reload == 0:
                    end = sample
        if end is not None:
            ch.pending_end = end

    def _update_gain(self, ch: _NesChannel, sample: int) -> None:
        if ch.held:
            if ch.const_volume:
                gain = _env_gain(ch.volume)
                ch.timeline.gain(sample, gain)
                if gain <= 0.0:
                    ch.timeline.note_off(sample)
                    ch.held = False
                    ch.pending_end = None
            elif ch.env is not None:
                ch.timeline.gain(sample, _env_gain(ch.env.level))
        elif ch.const_volume and ch.volume > 0 and ch.length_index:
            # The length counter may still be running; a later volume write
            # makes the still-keyed channel audible again.
            self._note_on(ch, sample)

    def write(self, sample: int, reg: int, val: int) -> None:
        self._advance(sample)
        val &= 0xFF
        reg &= 0x7F
        if reg == 0x00:
            ch = self.channels[0]
            ch.duty = (val >> 6) & 3
            ch.length_halt = bool(val & 0x20)
            ch.const_volume = bool(val & 0x10)
            ch.volume = val & 0x0F
            ch.env_period = val & 0x0F
            self._update_gain(ch, sample)
        elif reg == 0x01:
            pass  # sweep: ignored
        elif reg == 0x02:
            ch = self.channels[0]
            ch.timer = (ch.timer & 0x700) | val
            if ch.held:
                ch.timeline.pitch(sample, ch.pitch_units(self.clock))
        elif reg == 0x03:
            ch = self.channels[0]
            ch.timer = (ch.timer & 0xFF) | ((val & 7) << 8)
            ch.length_index = (val >> 3) & 0x1F
            if ch.length_index:
                self._note_on(ch, sample)
            elif ch.held:
                ch.timeline.pitch(sample, ch.pitch_units(self.clock))
        elif reg == 0x04:
            ch = self.channels[1]
            ch.duty = (val >> 6) & 3
            ch.length_halt = bool(val & 0x20)
            ch.const_volume = bool(val & 0x10)
            ch.volume = val & 0x0F
            ch.env_period = val & 0x0F
            self._update_gain(ch, sample)
        elif reg == 0x05:
            pass
        elif reg == 0x06:
            ch = self.channels[1]
            ch.timer = (ch.timer & 0x700) | val
            if ch.held:
                ch.timeline.pitch(sample, ch.pitch_units(self.clock))
        elif reg == 0x07:
            ch = self.channels[1]
            ch.timer = (ch.timer & 0xFF) | ((val & 7) << 8)
            ch.length_index = (val >> 3) & 0x1F
            if ch.length_index:
                self._note_on(ch, sample)
            elif ch.held:
                ch.timeline.pitch(sample, ch.pitch_units(self.clock))
        elif reg == 0x08:
            ch = self.channels[2]
            ch.linear_control = bool(val & 0x80)
            ch.length_halt = bool(val & 0x80)
            ch.linear_reload = val & 0x7F
        elif reg == 0x0A:
            ch = self.channels[2]
            ch.timer = (ch.timer & 0x700) | val
            if ch.held:
                ch.timeline.pitch(sample, ch.pitch_units(self.clock))
        elif reg == 0x0B:
            ch = self.channels[2]
            ch.timer = (ch.timer & 0xFF) | ((val & 7) << 8)
            ch.length_index = (val >> 3) & 0x1F
            if ch.length_index:
                self._note_on(ch, sample)
            elif ch.held:
                ch.timeline.pitch(sample, ch.pitch_units(self.clock))
        elif reg == 0x0C:
            ch = self.channels[3]
            ch.length_halt = bool(val & 0x20)
            ch.const_volume = bool(val & 0x10)
            ch.volume = val & 0x0F
            ch.env_period = val & 0x0F
            self._update_gain(ch, sample)
        elif reg == 0x0E:
            ch = self.channels[3]
            ch.noise_period = val & 0x0F
            ch.short_noise = bool(val & 0x80)
            if ch.held:
                ch.timeline.pitch(sample, ch.pitch_units(self.clock))
        elif reg == 0x0F:
            ch = self.channels[3]
            ch.length_index = (val >> 3) & 0x1F
            if ch.length_index:
                self._note_on(ch, sample)

    def finish(self, total_samples: int) -> None:
        self._advance(total_samples)
        for ch in self.channels:
            ch.timeline.close_all(total_samples)

    def tracks(self) -> list[Track]:
        out: list[Track] = []
        order = self.order_base
        for ch in self.channels:
            if ch.timeline.notes:
                out.append(
                    Track(
                        label=ch.timeline.label,
                        module=ch.module(),
                        kind="nes",
                        timeline=ch.timeline,
                        default_voice=SimpleVoice(ch.module(), ch.wave(), "NES"),
                        order=order,
                    )
                )
                order += 1
        return out


def extract_nes(vgm: VgmFile, voices, *, instance: int = 0, order_base: int = 0) -> ExtractResult:
    result = ExtractResult()
    writes = sorted_writes(vgm, "nes", instance)
    if not writes:
        return result
    clock = vgm.clock("nes", instance) or 1_789_773
    label = "NES APU" if instance == 0 else f"NES APU #{instance + 1}"
    apu = _NesApu(clock, label, order_base)
    # Length-register writes key the channel and also carry the timer high
    # bits.  Drivers commonly write the timer low byte, period or volume in the
    # same sample; apply those non-key writes first so the key-on snapshot sees
    # the intended pitch and level.
    grouped: list[list] = []
    for w in writes:
        if grouped and grouped[-1][0].sample == w.sample:
            grouped[-1].append(w)
        else:
            grouped.append([w])
    keys = (0x03, 0x07, 0x0B, 0x0F)
    for group in grouped:
        for w in group:
            if w.reg not in keys:
                apu.write(w.sample, w.reg, w.val)
        for w in group:
            if w.reg in keys:
                apu.write(w.sample, w.reg, w.val)
    apu.finish(vgm.total_samples)
    result.tracks.extend(apu.tracks())
    result.stats["notes"] = sum(len(ch.timeline.notes) for ch in apu.channels)
    result.stats["writes"] = len(writes)
    if any(w.chip == "nes_dmc" for w in vgm.writes):
        result.warnings.append("NES DPCM writes ignored (PCM is out of scope)")
    return result
