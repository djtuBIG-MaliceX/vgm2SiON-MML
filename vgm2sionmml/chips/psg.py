"""PSG decoders: AY-3-8910/YM2149 and SN76489/T6W28.

AY/SSG channels feed SiON's ``%0`` PSG module (``@0`` square, ``@1`` pulse
noise).  A channel that enables both tone and noise gets two SiON tracks so
the mixed hardware output is represented.  The AY envelope generator is
simulated at its own step rate and written as ``x`` expression points.

SN76489 serial bytes are decoded into three tone channels and one noise
channel.  Game Gear stereo writes (VGM 0x4F/0x3F) become per-channel pan.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model import ChannelTimeline, SimpleVoice, Track
from ..pitch import (
    ay_freq,
    ay_noise_note,
    freq_to_midi,
    midi_to_units,
    sn_freq,
    sn_noise_note,
)
from ..vgm import VgmFile
from .common import ExtractResult, db_gain, sorted_writes


def _ay_gain(level: int) -> float:
    level = max(0, min(15, int(level)))
    if level <= 0:
        return 0.0
    # The real chip sums its three channels into one mono output.  VGM
    # players normalize that mix by 1/3; without the same headroom three
    # full-scale SiON tracks make the PSG drown the FM channels.
    return db_gain((15 - level) * 1.5) / 3.0


def _sn_gain(atten: int) -> float:
    atten = max(0, min(15, int(atten)))
    if atten >= 15:
        return 0.0
    return db_gain(atten * 2.0)


def _ay_voice_noise() -> SimpleVoice:
    return SimpleVoice(0, 1, "AY noise")


def _ay_voice_tone() -> SimpleVoice:
    return SimpleVoice(0, 0, "AY square")


@dataclass
class _AyEnv:
    clock: int
    period: int = 0
    shape: int = 0
    step_samples: int = 1
    next_sample: int = 0
    level: int = 15
    attack: bool = False
    active: bool = False

    def reset(self, sample: int, period: int, shape: int) -> None:
        self.period = max(1, period & 0xFFFF)
        self.shape = shape & 0x0F
        self.step_samples = max(1, int(round(self.clock / (16.0 * self.period))))
        self.attack = bool(self.shape & 4)
        self.level = 0 if self.attack else 15
        self.next_sample = sample + self.step_samples
        self.active = True

    def levels_until(self, end_sample: int, limit: int = 4096) -> list[tuple[int, int]]:
        """Advance the envelope to ``end_sample``; return (sample, level) steps."""
        out: list[tuple[int, int]] = []
        if not self.active:
            return out
        count = 0
        while self.next_sample <= end_sample and count < limit:
            sample = self.next_sample
            count += 1
            if self.attack:
                self.level += 1
            else:
                self.level -= 1
            if self.level > 15 or self.level < 0:
                if self.shape & 1:  # hold
                    self.level = 15 if self.level > 15 else 0
                    self.active = False
                elif self.shape & 2:  # alternate
                    self.attack = not self.attack
                    self.level = 15 if self.level > 15 else 0
                elif self.shape & 8:  # continue
                    self.level = 0 if self.attack else 15
                else:
                    self.level = 15 if self.level > 15 else 0
                    self.active = False
            out.append((sample, self.level))
            self.next_sample = sample + self.step_samples
        if count >= limit:
            self.active = False
        return out


class _AyChip:
    def __init__(self, clock: int, label: str, order_base: int) -> None:
        self.clock = clock
        self.label = label
        self.order_base = order_base
        self.regs = bytearray(16)
        self.regs[7] = 0xFF
        self.tone = [
            ChannelTimeline(f"{label} ch {i + 1} tone") for i in range(3)
        ]
        self.noise = [
            ChannelTimeline(f"{label} ch {i + 1} noise") for i in range(3)
        ]
        self.tone_active = [False] * 3
        self.noise_active = [False] * 3
        self.env = _AyEnv(clock)
        self.env_mode = [False] * 3
        self.divider_dirty = False
        self.divider_dirty_sample = 0

    def tone_divider(self, ch: int) -> int:
        low = self.regs[ch * 2]
        high = self.regs[ch * 2 + 1] & 0x0F
        return max(1, low | (high << 8))

    def noise_period(self) -> int:
        return max(1, self.regs[6] & 0x1F)

    def volume(self, ch: int) -> int:
        return self.regs[8 + ch] & 0x0F

    def envelope_enabled(self, ch: int) -> bool:
        return bool(self.regs[8 + ch] & 0x10)

    def tone_enabled(self, ch: int) -> bool:
        return not (self.regs[7] & (1 << ch))

    def noise_enabled(self, ch: int) -> bool:
        return not (self.regs[7] & (1 << (ch + 3)))

    def volume_gain(self, ch: int) -> float:
        if self.envelope_enabled(ch):
            return _ay_gain(self.env.level)
        return _ay_gain(self.volume(ch))

    def has_signal(self, ch: int) -> bool:
        if self.envelope_enabled(ch):
            return True
        return self.volume(ch) > 0

    def pitch(self, ch: int) -> int:
        return midi_to_units(freq_to_midi(ay_freq(self.tone_divider(ch), self.clock)))

    def noise_pitch(self) -> int:
        return ay_noise_note(self.noise_period()) * 64

    def refresh(self, sample: int, *, add_points: bool = True) -> None:
        for ch in range(3):
            tone_gate = self.tone_enabled(ch) and self.has_signal(ch)
            noise_gate = self.noise_enabled(ch) and self.has_signal(ch)
            gain = self.volume_gain(ch)
            tone_scale = 1.0
            if tone_gate and noise_gate:
                # Two SiON tracks represent one hardware channel; split the
                # level so the mix is not twice as loud.
                tone_scale = 0.7
            tone_gain = gain * tone_scale
            noise_gain = gain * (0.7 if (tone_gate and noise_gate) else 1.0)
            tone_pitch = self.pitch(ch)
            if tone_gate and not self.tone_active[ch]:
                self.tone[ch].note_on(
                    sample, tone_pitch, _ay_voice_tone(), tone_gain, 4.0
                )
                self.tone_active[ch] = True
            elif not tone_gate and self.tone_active[ch]:
                self.tone[ch].note_off(sample)
                self.tone_active[ch] = False
            elif tone_gate:
                open_note = self.tone[ch].open
                if (
                    open_note is not None
                    and abs(tone_pitch - open_note.pitch) >= 48
                    and sample - open_note.start >= 1470
                ):
                    self.tone[ch].note_on(
                        sample, tone_pitch, _ay_voice_tone(), tone_gain, 4.0
                    )
                elif add_points:
                    self.tone[ch].pitch(sample, tone_pitch)
                    self.tone[ch].gain(sample, tone_gain)
            noise_pitch = self.noise_pitch()
            if noise_gate and not self.noise_active[ch]:
                self.noise[ch].note_on(
                    sample, noise_pitch, _ay_voice_noise(), noise_gain, 4.0
                )
                self.noise_active[ch] = True
            elif not noise_gate and self.noise_active[ch]:
                self.noise[ch].note_off(sample)
                self.noise_active[ch] = False
            elif noise_gate:
                # Noise period changes alter the LFSR rate, not the musical
                # note.  Keep them as pitch points so a period sweep does not
                # become a burst of zero-length retriggers.
                if add_points:
                    self.noise[ch].pitch(sample, noise_pitch)
                    self.noise[ch].gain(sample, noise_gain)
    def advance_envelope(self, sample: int) -> None:
        for ch in range(3):
            if not self.env_mode[ch]:
                continue
            for t, level in self.env.levels_until(sample):
                gain = _ay_gain(level)
                if self.tone_active[ch]:
                    self.tone[ch].gain(t, gain * (0.7 if self.noise_active[ch] else 1.0))
                if self.noise_active[ch]:
                    self.noise[ch].gain(t, gain * (0.7 if self.tone_active[ch] else 1.0))

    def apply_write(self, sample: int, reg: int, val: int) -> None:
        self.advance_envelope(sample)
        old_env = [self.env_mode[i] for i in range(3)]
        reg &= 0x0F
        self.regs[reg] = val & 0xFF
        divider_write = reg in (0, 1, 2, 3, 4, 5)
        if divider_write and self.divider_dirty and sample - self.divider_dirty_sample > 16:
            self.refresh(self.divider_dirty_sample)
            self.divider_dirty = False
        if reg == 13:
            self.env.reset(sample, self.envelope_period(), val)
        elif reg in (11, 12):
            if any(self.env_mode):
                self.env.reset(sample, self.envelope_period(), self.regs[13])
        elif 8 <= reg <= 10:
            ch = reg - 8
            if self.envelope_enabled(ch) and not old_env[ch]:
                self.env.reset(sample, self.envelope_period(), self.regs[13])
            self.env_mode[ch] = self.envelope_enabled(ch)
        if divider_write:
            self.divider_dirty = True
            self.divider_dirty_sample = sample
            return
        if self.divider_dirty:
            self.refresh(sample)
            self.divider_dirty = False
        self.refresh(sample)

    def envelope_period(self) -> int:
        return max(1, self.regs[11] | ((self.regs[12] & 0xFF) << 8))

    def finish(self, total_samples: int) -> None:
        self.advance_envelope(total_samples)
        if self.divider_dirty:
            self.refresh(total_samples)
            self.divider_dirty = False
        for ch in range(3):
            self.tone[ch].close_all(total_samples)
            self.noise[ch].close_all(total_samples)

    def tracks(self) -> list[Track]:
        out: list[Track] = []
        order = self.order_base
        for ch in range(3):
            if self.tone[ch].notes:
                out.append(
                    Track(
                        label=self.tone[ch].label,
                        module=0,
                        kind="psg",
                        timeline=self.tone[ch],
                        default_voice=_ay_voice_tone(),
                        order=order,
                    )
                )
                order += 1
            if self.noise[ch].notes:
                out.append(
                    Track(
                        label=self.noise[ch].label,
                        module=0,
                        kind="psg",
                        timeline=self.noise[ch],
                        default_voice=_ay_voice_noise(),
                        order=order,
                    )
                )
                order += 1
        return out


def extract_ay8910(
    vgm: VgmFile,
    voices,
    *,
    chip: str = "ay8910",
    instance: int = 0,
    order_base: int = 0,
    label: str | None = None,
) -> ExtractResult:
    """Decode one AY-compatible SSG core."""
    result = ExtractResult()
    writes = sorted_writes(vgm, chip, instance)
    if not writes:
        return result
    clock = vgm.clock(chip, instance)
    if not clock:
        clock = 1_789_772
    name = label or ("AY-3-8910" if chip == "ay8910" else chip.upper())
    chip_label = name if instance == 0 else f"{name} #{instance + 1}"
    ay = _AyChip(clock, chip_label, order_base)
    for w in writes:
        ay.apply_write(w.sample, w.reg, w.val)
    ay.finish(vgm.total_samples)
    result.tracks.extend(ay.tracks())
    result.stats["notes"] = sum(len(tl.notes) for tl in ay.tone + ay.noise)
    result.stats["writes"] = len(writes)
    return result


# --------------------------------------------------------------------- SN76489


@dataclass
class _SnChannel:
    divider: int = 1
    noise: int = 0
    attenuation: int = 15
    noise_mode: int = 0


class _SnChip:
    def __init__(self, clock: int, label: str, order_base: int) -> None:
        self.clock = clock
        self.label = label
        self.order_base = order_base
        self.channels = [_SnChannel() for _ in range(4)]
        self.latched_channel = 0
        self.latched_volume = False
        self.stereo = [0xFF] * 4  # every channel on both sides unless GG says otherwise
        self.timelines = [
            ChannelTimeline(f"{label} ch {i + 1}") for i in range(3)
        ]
        self.noise_tl = ChannelTimeline(f"{label} noise")
        self.active = [False] * 3
        self.noise_active = False
        self.divider_dirty = False
        self.divider_dirty_sample = 0

    def pan_gain(self, ch: int) -> tuple[float, float]:
        mask = self.stereo[ch]
        left = bool(mask & (1 << ch))
        right = bool(mask & (1 << (ch + 4)))
        if left and right:
            return 4.0, 1.0
        if left:
            return 0.0, 1.0
        if right:
            return 8.0, 1.0
        return 4.0, 0.0

    def tone_gain(self, ch: int) -> tuple[float, float]:
        att = self.channels[ch].attenuation
        gain = _sn_gain(att) if att < 15 else 0.0
        pan, stereo_gain = self.pan_gain(ch)
        return pan, gain * stereo_gain

    def noise_gain(self) -> float:
        att = self.channels[3].attenuation
        if att >= 15:
            return 0.0
        mask = self.stereo[3]
        left = bool(mask & (1 << 3))
        right = bool(mask & (1 << 7))
        if not left and not right:
            return 0.0
        # Noise is a single mono source on the real chip; use the average of
        # the two enable bits as the gain, and pan by whichever side is on.
        return _sn_gain(att) * (1.0 if (left and right) else 0.75)

    def refresh(self, sample: int) -> None:
        for ch in range(3):
            pan, gain = self.tone_gain(ch)
            audible = gain > 0.0
            pitch = midi_to_units(freq_to_midi(sn_freq(self.channels[ch].divider, self.clock)))
            if audible and not self.active[ch]:
                self.timelines[ch].note_on(
                    sample, pitch, SimpleVoice(0, 0, "SN square"), gain, pan
                )
                self.active[ch] = True
            elif not audible and self.active[ch]:
                self.timelines[ch].note_off(sample)
                self.active[ch] = False
            elif audible:
                note = self.timelines[ch].open
                if (
                    note is not None
                    and abs(pitch - note.pitch) >= 48
                    and sample - note.start >= 735
                ):
                    self.timelines[ch].note_on(
                        sample, pitch, SimpleVoice(0, 0, "SN square"), gain, pan
                    )
                else:
                    self.timelines[ch].pitch(sample, pitch)
                    self.timelines[ch].pan(sample, pan)
                    self.timelines[ch].gain(sample, gain)
        gain = self.noise_gain()
        mask = self.stereo[3]
        left = bool(mask & 0x08)
        right = bool(mask & 0x80)
        pan = 4.0 if (left and right) else (0.0 if left else (8.0 if right else 4.0))
        voice = SimpleVoice(0, 2 if (self.channels[3].noise_mode & 4) == 0 else 1,
                            "SN noise")
        pitch = sn_noise_note(self.channels[3].noise & 7) * 64
        if gain > 0.0 and not self.noise_active:
            self.noise_tl.note_on(sample, pitch, voice, gain, pan)
            self.noise_active = True
        elif gain <= 0.0 and self.noise_active:
            self.noise_tl.note_off(sample)
            self.noise_active = False
        elif gain > 0.0:
            note = self.noise_tl.open
            if note is not None and note.voice != voice:
                self.noise_tl.note_on(sample, pitch, voice, gain, pan)
            elif (
                note is not None
                and abs(pitch - note.pitch) >= 48
                and sample - note.start >= 735
            ):
                self.noise_tl.note_on(sample, pitch, voice, gain, pan)
            else:
                self.noise_tl.pitch(sample, pitch)
                self.noise_tl.pan(sample, pan)
                self.noise_tl.gain(sample, gain)

    def write_byte(self, sample: int, value: int) -> None:
        value &= 0xFF
        if self.divider_dirty and sample - self.divider_dirty_sample > 16:
            self.refresh(self.divider_dirty_sample)
            self.divider_dirty = False
        if value & 0x80:
            ch = (value >> 5) & 3
            self.latched_channel = ch
            self.latched_volume = bool(value & 0x10)
            data = value & 0x0F
            if self.latched_volume:
                self.channels[ch].attenuation = data
                self.refresh(sample)
            else:
                if ch == 3:
                    self.channels[ch].noise = data & 0x07
                    self.channels[ch].noise_mode = data
                    self.refresh(sample)
                else:
                    self.channels[ch].divider = (self.channels[ch].divider & 0x3F0) | data
                    self.divider_dirty = True
                    self.divider_dirty_sample = sample
            return
        else:
            data = value & 0x3F
            ch = self.latched_channel
            if self.latched_volume:
                # Some drivers emit a second byte for a 4-bit volume anyway;
                # the low nibble is the authoritative value.
                self.channels[ch].attenuation = data & 0x0F
            else:
                if ch == 3:
                    self.channels[ch].noise = data & 0x07
                    self.channels[ch].noise_mode = data
                else:
                    self.channels[ch].divider = (self.channels[ch].divider & 0x0F) | (data << 4)
        self.divider_dirty = False
        self.refresh(sample)

    def set_stereo(self, mask: int) -> None:
        for ch in range(4):
            self.stereo[ch] = mask

    def finish(self, total_samples: int) -> None:
        if self.divider_dirty:
            self.refresh(total_samples)
            self.divider_dirty = False
        for tl in self.timelines:
            tl.close_all(total_samples)
        self.noise_tl.close_all(total_samples)

    def tracks(self) -> list[Track]:
        out: list[Track] = []
        order = self.order_base
        for i, tl in enumerate(self.timelines):
            if tl.notes:
                out.append(
                    Track(
                        label=tl.label,
                        module=0,
                        kind="psg",
                        timeline=tl,
                        default_voice=SimpleVoice(0, 0, "SN square"),
                        order=order,
                    )
                )
                order += 1
        if self.noise_tl.notes:
            out.append(
                Track(
                    label=self.noise_tl.label,
                    module=0,
                    kind="psg",
                    timeline=self.noise_tl,
                    default_voice=SimpleVoice(0, 2, "SN noise"),
                    order=order,
                )
            )
        return out


def extract_sn76489(
    vgm: VgmFile, voices, *, instance: int = 0, order_base: int = 0
) -> ExtractResult:
    result = ExtractResult()
    writes = sorted_writes(vgm, "sn76489", instance)
    if not writes:
        return result
    clock = vgm.clock("sn76489", instance) or 3_579_545
    label = "SN76489" if instance == 0 else f"SN76489 #{instance + 1}"
    chip = _SnChip(clock, label, order_base)
    for w in writes:
        if w.reg == 1:
            chip.set_stereo(w.val & 0xFF)
            chip.refresh(w.sample)
        else:
            chip.write_byte(w.sample, w.val)
    chip.finish(vgm.total_samples)
    result.tracks.extend(chip.tracks())
    result.stats["notes"] = sum(len(tl.notes) for tl in chip.timelines + [chip.noise_tl])
    result.stats["writes"] = len(writes)
    return result
