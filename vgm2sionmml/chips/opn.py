"""OPN-family FM decoder: YM2203, YM2612, YM2608, YM2610.

The FM half is exposed as SiON ``#OPN@`` voices.  The AY-compatible SSG half
is decoded by :mod:`vgm2sionmml.chips.psg` under a separate ``*_ssg`` chip
stream.  PCM/ADPCM sections are ignored by design.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from ..model import ChannelTimeline, FmVoice, LfoState, Track, VoiceLibrary
from ..pitch import opn_freq, freq_to_midi, midi_to_units
from ..vgm import VgmFile
from .common import ExtractResult, db_gain, normalize_carrier_levels, sorted_writes

# Carrier sets in *register* order (S1, S3, S2, S4).
OPN_CARRIERS = {
    0: (3,),
    1: (3,),
    2: (3,),
    3: (3,),
    4: (2, 3),
    5: (1, 2, 3),
    6: (1, 2, 3),
    7: (0, 1, 2, 3),
}

# Register slot -> logical datasheet operator (S1, S2, S3, S4).
SLOT_TO_LOGICAL = (0, 2, 1, 3)
BURST_SAMPLES = 64
# SiON operator order (S1, S2, S3, S4) carrier sets.
SIOPM_CARRIERS = {
    0: (3,),
    1: (3,),
    2: (3,),
    3: (3,),
    4: (1, 3),
    5: (1, 2, 3),
    6: (1, 2, 3),
    7: (0, 1, 2, 3),
}
LFO_RATES = (3.98, 5.56, 6.66, 7.37, 8.00, 10.0, 13.3, 16.0)

# SiON sums one track per hardware voice.  Genesis OPN logs commonly sit
# around 9 dB below OPM logs in carrier TL, and VGM players hide that by
# normalizing every enabled device (including an otherwise unused PSG half).
# Apply a fixed family-level reference instead; this is not a per-song gain
# and deliberately ignores chip count / source playback normalization.
OPN_OUTPUT_TRIM = 3.0


def _opn_gain(gain: float) -> float:
    return max(0.0, min(1.0, gain * OPN_OUTPUT_TRIM))


@dataclass
class OpnOp:
    dt_mul: int = 0x01
    tl: int = 0x7F
    ks_ar: int = 0x1F
    am_d1r: int = 0x00
    d2r: int = 0x00
    sl_rr: int = 0xF7
    ssg_eg: int = 0x00


@dataclass
class OpnState:
    """Register image of one OPN instance.

    ``ops[ch][slot]`` uses the chip's register slot order, which is
    S1, S3, S2, S4.  The voice builder re-orders it to S1..S4.
    """

    channels: int
    ops: list[list[OpnOp]]
    fnum: list[int]
    block: list[int]
    fb_alg_pan: list[int]
    pms_ams: list[int]
    pan_reg: list[int]
    keyed: list[bool]
    lfo_enable: bool = False
    lfo_rate: int = 0
    special_ch3: bool = False
    dac_enable: bool = False
    dac_data: int = 0
    freq_divisor: float = 144.0

    @classmethod
    def new(cls, channels: int, freq_divisor: float = 144.0) -> "OpnState":
        return cls(
            channels=channels,
            freq_divisor=freq_divisor,
            ops=[[OpnOp() for _ in range(4)] for _ in range(channels)],
            fnum=[0] * channels,
            block=[0] * channels,
            fb_alg_pan=[0] * channels,
            pms_ams=[0] * channels,
            pan_reg=[0] * channels,
            keyed=[False] * channels,
        )

    def write(self, reg: int, val: int) -> None:
        val &= 0xFF
        bank = (reg >> 8) & 1
        r = reg & 0xFF
        # Global registers exist only in bank 0.
        if bank == 0 and r <= 0x2F:
            if r == 0x22:
                self.lfo_enable = bool(val & 0x08)
                self.lfo_rate = val & 0x07
            elif r == 0x27:
                self.special_ch3 = bool(val & 0x40)
            elif r == 0x2B:
                self.dac_enable = bool(val & 0x80)
            elif r == 0x2A:
                self.dac_data = val
            return
        if 0x30 <= r <= 0x9F:
            slot_off = r & 0x0F
            ch_local = slot_off & 3
            slot = (r >> 2) & 3
            ch = ch_local + (3 if bank else 0)
            if ch >= self.channels:
                return
            op = self.ops[ch][slot]
            page = r & 0xF0
            if page == 0x30:
                op.dt_mul = val
            elif page == 0x40:
                op.tl = val & 0x7F
            elif page == 0x50:
                op.ks_ar = val
            elif page == 0x60:
                op.am_d1r = val
            elif page == 0x70:
                op.d2r = val & 0x1F
            elif page == 0x80:
                op.sl_rr = val
            elif page == 0x90:
                op.ssg_eg = val & 0x0F
            return
        if 0xA4 <= r <= 0xA6:
            ch_local = r & 3
            ch = ch_local + (3 if bank else 0)
            if ch >= self.channels:
                return
            # OPN high register: bits 0-2 are FNum[10:8], bits 3-5 are Block.
            self.fnum[ch] = (self.fnum[ch] & 0xFF) | ((val & 0x07) << 8)
            self.block[ch] = (val >> 3) & 7
            return
        if 0xA0 <= r <= 0xA2:
            ch_local = r & 3
            ch = ch_local + (3 if bank else 0)
            if ch >= self.channels:
                return
            self.fnum[ch] = (self.fnum[ch] & 0x700) | val
            return
        if 0xB0 <= r <= 0xB2:
            ch_local = r & 3
            ch = ch_local + (3 if bank else 0)
            if ch >= self.channels:
                return
            self.fb_alg_pan[ch] = val
            return
        if 0xB4 <= r <= 0xB6:
            # Per-channel LFO sensitivity (PMS/AMS) and, on the stereo OPN
            # members, the L/R output bits in the top two bits.
            ch_local = r & 3
            ch = ch_local + (3 if bank else 0)
            if ch >= self.channels:
                return
            self.pms_ams[ch] = val & 0x3F
            self.pan_reg[ch] = (val >> 6) & 3
            return

    def pitch_units(self, ch: int, clock: int) -> int:
        return midi_to_units(
            freq_to_midi(
                opn_freq(self.block[ch], self.fnum[ch], clock, self.freq_divisor)
            )
        )

    def carrier_tl(self, ch: int) -> int:
        alg = self.fb_alg_pan[ch] & 7
        tls = [self.ops[ch][i].tl for i in OPN_CARRIERS.get(alg, (3,))]
        return min(tls) if tls else 0x7F

    def pan_gain(self, ch: int) -> tuple[float, float]:
        # YM2203 is mono; its 0xB4 register has no output-enable field.
        if self.channels == 3:
            return 4.0, 1.0
        field = self.pan_reg[ch]
        # OPN $B4 is the opposite bit order from the OPM's $20: 1 = right,
        # 2 = left, 3 = both, 0 = muted.
        if field == 3:
            return 4.0, 1.0
        if field == 2:
            return 0.0, 1.0
        if field == 1:
            return 8.0, 1.0
        return 4.0, 0.0

    def patch(self, ch: int) -> tuple[int, int, tuple[tuple[int, ...], ...]]:
        b = self.fb_alg_pan[ch]
        alg = b & 7
        fb = (b >> 3) & 7
        ams = (self.pms_ams[ch] >> 4) & 3 if self.lfo_enable else 0
        ops: list[tuple[int, ...]] = []
        for logical in range(4):
            slot = SLOT_TO_LOGICAL[logical]
            op = self.ops[ch][slot]
            ar = op.ks_ar & 0x1F
            ks = (op.ks_ar >> 6) & 3
            dr = op.am_d1r & 0x1F
            sr = op.d2r & 0x1F
            rr = op.sl_rr & 0x0F
            sl = (op.sl_rr >> 4) & 0x0F
            tl = op.tl & 0x7F
            mul = op.dt_mul & 0x0F
            dt1 = (op.dt_mul >> 4) & 0x07
            # SiON's #OPN@ has 10 operator fields; AMS is the last one.
            ops.append((ar, dr, sr, rr, sl, tl, ks, mul, dt1, ams))
        return alg, fb, tuple(ops)

    def lfo(self, ch: int) -> LfoState | None:
        if not self.lfo_enable:
            return None
        pms = self.pms_ams[ch] & 7
        ams = (self.pms_ams[ch] >> 4) & 3
        if not pms and not ams:
            return None
        rate = LFO_RATES[self.lfo_rate & 7]
        period = max(1, min(255, int(round(60.0 / rate))))
        pitch = pms * 8
        amp = ams * 8
        return LfoState(period=period, wave=2, pitch=pitch, amp=amp)


def _apply_burst(
    state: OpnState, writes: list, start: int, ch: int, bank: int
) -> OpnState:
    snap = copy.deepcopy(state)
    sample0 = writes[start].sample
    for w in writes[start + 1 :]:
        if w.sample - sample0 > BURST_SAMPLES:
            break
        reg = w.reg
        if (reg & 0x100) != (bank << 8):
            continue
        if (reg & 3) != (ch & 3):
            continue
        r = reg & 0xFF
        if 0x30 <= r <= 0x9F or 0xA0 <= r <= 0xB6:
            snap.write(reg, w.val)
    return snap


def _channels_for(kind: str, flags: dict[str, int]) -> int:
    if kind == "ym2203":
        return 3
    if kind == "ym2610":
        return 6 if flags.get("ym2610_b") else 4
    return 6


def extract_opn(
    vgm: VgmFile,
    voices: VoiceLibrary,
    kind: str,
    order_base: int = 0,
    instance: int | None = None,
) -> ExtractResult:
    """Decode one OPN chip family from the write stream."""
    if kind not in ("ym2203", "ym2612", "ym2608", "ym2610"):
        raise ValueError(f"not an OPN kind: {kind}")
    result = ExtractResult()
    instances = (
        [instance]
        if instance is not None
        else sorted({w.chip_id for w in vgm.writes if w.chip == kind}) or [0]
    )
    channels = _channels_for(kind, vgm.chip_flags)
    stats = {"notes": 0, "keyons": 0, "tl_moves": 0, "special_skips": 0, "dac_skips": 0}
    skipped_ssg_eg = 0
    for instance in instances:
        writes = sorted_writes(vgm, kind, instance)
        if not writes:
            continue
        clock = vgm.clock(kind, instance)
        state = OpnState.new(channels, 72.0 if kind == "ym2203" else 144.0)
        timelines = [
            ChannelTimeline(f"{kind.upper()} {instance + 1} ch {ch + 1}")
            for ch in range(channels)
        ]
        tracks: list[Track] = []
        for ch, tl in enumerate(timelines):
            tracks.append(
                Track(
                    label=tl.label,
                    module=6,
                    kind="fm",
                    timeline=tl,
                    order=order_base + instance * 8 + ch,
                )
            )
        sounding = [False] * channels
        op_mask = [0] * channels
        base_tl = [0x7F] * channels
        mute = [False] * channels
        open_window = [-1] * channels
        last_lfo: list[LfoState | None] = [None] * channels

        for i, w in enumerate(writes):
            reg = w.reg
            bank = (reg >> 8) & 1
            r = reg & 0xFF
            val = w.val & 0xFF
            sample = w.sample
            if bank == 0 and r == 0x28:
                raw_ch = val & 7
                if raw_ch in (3, 7):
                    continue
                # 0-2 map directly; the second group continues at 3 after the
                # reserved slot 3.
                ch = raw_ch if raw_ch < 3 else raw_ch - 1
                mask = (val >> 4) & 0x0F
                if ch >= channels:
                    continue
                newly_on = mask & ~op_mask[ch]
                op_mask[ch] = mask
                if mask and newly_on:
                    if ch == 5 and state.dac_enable:
                        state.keyed[ch] = True
                        stats["dac_skips"] += 1
                        continue
                    if ch == 2 and state.special_ch3:
                        state.keyed[ch] = True
                        stats["special_skips"] += 1
                        continue
                    snap = _apply_burst(state, writes, i, ch, 1 if ch >= 3 else 0)
                    alg, fb, ops = snap.patch(ch)
                    if any(op.ssg_eg for op in snap.ops[ch]):
                        skipped_ssg_eg += 1
                    ops, tl0 = normalize_carrier_levels(
                        ops, SIOPM_CARRIERS.get(alg, (3,)), 5
                    )
                    voice = FmVoice("opn", alg, fb, ops)
                    pitch = snap.pitch_units(ch, clock)
                    pan, pan_gain = snap.pan_gain(ch)
                    timelines[ch].note_on(
                        sample,
                        pitch,
                        voice,
                        gain=_opn_gain(pan_gain * db_gain(tl0 * 0.75)),
                        pan=pan,
                    )
                    base_tl[ch] = tl0
                    mute[ch] = pan_gain <= 0.0
                    sounding[ch] = True
                    state.keyed[ch] = True
                    open_window[ch] = sample + BURST_SAMPLES
                    lfo = snap.lfo(ch)
                    if lfo is not None:
                        tracks[ch].lfo_points.append((sample, lfo))
                        last_lfo[ch] = lfo
                    else:
                        last_lfo[ch] = None
                    stats["notes"] += 1
                    stats["keyons"] += 1
                elif not mask and sounding[ch]:
                    timelines[ch].note_off(sample)
                    sounding[ch] = False
                    state.keyed[ch] = False
                continue

            state.write(reg, val)
            # A keyed channel can be cut by DAC mode or special mode.
            if state.dac_enable and sounding[5]:
                timelines[5].note_off(sample)
                sounding[5] = False
            if state.special_ch3 and sounding[2]:
                timelines[2].note_off(sample)
                sounding[2] = False

            if 0xA0 <= r <= 0xAF or 0xB0 <= r <= 0xB6:
                ch = (r & 3) + (3 if bank else 0)
                if ch >= channels:
                    continue
                if 0xA0 <= r <= 0xAF:
                    if sounding[ch]:
                        pitch = state.pitch_units(ch, clock)
                        note = timelines[ch].open
                        if note is not None and sample <= open_window[ch]:
                            note.pitch = pitch
                        else:
                            timelines[ch].pitch(sample, pitch)
                else:
                    pan, pan_gain = state.pan_gain(ch)
                    mute[ch] = pan_gain <= 0.0
                    timelines[ch].pan(sample, pan)
                    if sounding[ch]:
                        cur = state.carrier_tl(ch)
                        gain = 0.0 if mute[ch] else _opn_gain(
                            db_gain(cur * 0.75)
                        )
                        timelines[ch].gain(sample, gain)
                    if r == 0xB4 and sounding[ch]:
                        lfo = state.lfo(ch)
                        if lfo is not None and (
                            last_lfo[ch] is None or lfo.key != last_lfo[ch].key
                        ):
                            tracks[ch].lfo_points.append((sample, lfo))
                            last_lfo[ch] = lfo
                continue
            if 0x40 <= r <= 0x8F:
                ch = (r & 3) + (3 if bank else 0)
                if ch >= channels:
                    continue
                if 0x40 <= r <= 0x4F and sounding[ch]:
                    cur = state.carrier_tl(ch)
                    gain = 0.0 if mute[ch] else _opn_gain(
                        db_gain(cur * 0.75)
                    )
                    timelines[ch].gain(sample, gain)
                    stats["tl_moves"] += 1

        for ch in range(channels):
            timelines[ch].close_all(vgm.total_samples)
        result.tracks.extend(tracks)

    if skipped_ssg_eg:
        result.warnings.append(
            f"{skipped_ssg_eg} keyed OPN voice(s) used SSG-EG; the shape is not "
            "represented in SiON's #OPN@ voice"
        )
    if stats["dac_skips"]:
        result.warnings.append(
            f"{stats['dac_skips']} YM2612 DAC-mode key-on(s) skipped (PCM is out of scope)"
        )
    if stats["special_skips"]:
        result.warnings.append(
            f"{stats['special_skips']} channel-3 special-mode key-on(s) skipped"
        )
    result.stats.update(stats)
    return result
