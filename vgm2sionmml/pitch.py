"""Chip pitch decoding and SiON note conversion.

SiON's note number is a MIDI note (``o5c`` is 60) and its ``k`` command is a
1/64-semitone detune.  The converters keep pitch in that 1/64-semitone unit
internally so fine detune survives.
"""

from __future__ import annotations

import math

# Official YM2151 KC note nibble -> semitone from C of this octave.
# C#=0, D=1, D#=2, E=4, F=5, F#=6, G=8, G#=9, A=10, A#=12, B=13,
# C(next octave)=14.  3/7/11/15 are unused aliases.
KC_SEMI = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    12: 10,
    13: 11,
    14: 12,
    15: 12,
}

OPM_REFERENCE_CLOCK = 3_579_545
E5_CENTER = 0x80


def clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def freq_to_midi(freq: float, reference: float = 440.0) -> float:
    if freq <= 0:
        return 0.0
    return 69.0 + 12.0 * math.log2(freq / reference)


def midi_to_units(pitch: float) -> int:
    """MIDI note (float) -> SiON 1/64-semitone units."""
    return int(round(pitch * 64.0))


def units_to_note_k(units: int) -> tuple[int, int]:
    """Return (MML note number, k detune) for an internal pitch unit.

    SiON's octave command stops at 9, so B9 (note 119) is the highest
    representable note.  Higher source pitches are clamped there.
    """
    units = max(0, min(119 * 64 + 63, int(units)))
    note = units // 64
    detune = units - note * 64
    if note > 119:
        note, detune = 119, 0
    # k is a signed detune in the MML; a small positive fraction is fine.
    return note, detune


def note_name(note: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[note % 12]}{note // 12 - 1}"


def opm_kc_units(kc: int, kf: int, clock: int = OPM_REFERENCE_CLOCK) -> int:
    """YM2151 KC/KF -> 1/64-semitone units.

    KC 0x00 is C#4 in the OPM's own numbering (the KC octave field is one
    below the musical octave for C), so the base note is
    ``12 * (octave + 1) + semitone``.  The KF fraction is the top six bits,
    one unit per ``k`` step.  A non-default clock transposes the whole log
    exactly, because the OPM's pitch is a clock ratio.
    """
    octave = (kc >> 4) & 7
    semi = KC_SEMI[kc & 0x0F]
    midi_units = (12 * (octave + 1) + semi) * 64 + ((kf >> 2) & 0x3F)
    if clock and clock != OPM_REFERENCE_CLOCK:
        midi_units += int(round(768.0 * math.log2(clock / OPM_REFERENCE_CLOCK)))
    return midi_units


def opm_clock_offset(clock: int) -> int:
    if not clock or clock == OPM_REFERENCE_CLOCK:
        return 0
    return int(round(768.0 * math.log2(clock / OPM_REFERENCE_CLOCK)))


def opn_freq(block: int, fnum: int, clock: int) -> float:
    """OPN (YM2203/2612/2608/2610) block/f-number -> Hz."""
    block = max(0, min(7, int(block)))
    fnum = max(1, min(0x7FF, int(fnum)))
    return fnum * clock / (144.0 * (2.0 ** (20 - block)))


def opl_freq(block: int, fnum: int, clock: int, opl3: bool = False) -> float:
    """OPL/OPL2/OPL3 block/f-number -> Hz.

    The OPL3 runs at four times the OPL2 prescaler; its usual 14.318 MHz clock
    therefore produces the same pitch as an OPL2 at 3.58 MHz for the same
    register value.
    """
    block = max(0, min(7, int(block)))
    fnum = max(1, min(0x3FF, int(fnum)))
    divisor = 288.0 if opl3 else 72.0
    return fnum * clock / (divisor * (2.0 ** (20 - block)))


def opn_units(block: int, fnum: int, clock: int) -> int:
    return midi_to_units(freq_to_midi(opn_freq(block, fnum, clock)))


def opl_units(block: int, fnum: int, clock: int, opl3: bool = False) -> int:
    return midi_to_units(freq_to_midi(opl_freq(block, fnum, clock, opl3)))


def ay_freq(divider: int, clock: int) -> float:
    divider = max(1, min(0xFFF, int(divider)))
    return clock / (16.0 * divider)


def ay_noise_freq(period: int, clock: int) -> float:
    period = max(1, min(0x1F, int(period)))
    return clock / (16.0 * period)


def sn_freq(divider: int, clock: int) -> float:
    divider = max(1, min(0x3FF, int(divider)))
    return clock / (32.0 * divider)


def sn_noise_freq(rate: int, clock: int) -> float:
    # The SN76489 noise source shifts its 15-bit LFSR at clock/16 divided by
    # the noise register's power-of-two period.
    rate = max(0, min(7, int(rate)))
    return clock / (16.0 * (16 << rate))


def nes_pulse_freq(timer: int, clock: int) -> float:
    timer = max(0, min(0x7FF, int(timer)))
    return clock / (16.0 * (timer + 1))


def nes_triangle_freq(timer: int, clock: int) -> float:
    timer = max(0, min(0x7FF, int(timer)))
    return clock / (32.0 * (timer + 1))


def ntsc_noise_note(period: int) -> int:
    """NES noise period (0-15) as the SiON o0c..o1d+ note."""
    return 12 + max(0, min(15, int(period)))


def ay_noise_note(period: int) -> int:
    """AY noise period (0-31) as the SiON o0c..o2g+ note."""
    return 12 + max(0, min(31, int(period)))


def sn_noise_note(rate: int) -> int:
    return 12 + max(0, min(7, int(rate))) * 4
