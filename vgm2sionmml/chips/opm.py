"""YM2151 (OPM) register decoder.

Eight four-operator FM channels.  The patch is encoded exactly as SiON's
``#OPM@`` parameters expect: AR, DR, SR, RR, SL, TL, KS, MUL, DT1, DT2, AMS
per operator, plus ALG/FB.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from ..model import ChannelTimeline, FmVoice, LfoState, Track, VoiceLibrary
from ..pitch import opm_kc_units
from ..vgm import VgmFile
from .common import ExtractResult, db_gain, sorted_writes

# Carrier operator sets in register order.
CARRIERS = {
    0: (3,),
    1: (3,),
    2: (3,),
    3: (3,),
    4: (2, 3),
    5: (1, 2, 3),
    6: (1, 2, 3),
    7: (0, 1, 2, 3),
}

BURST_SAMPLES = 64


@dataclass
class OpRegs:
    dt_mul: int = 0x01
    tl: int = 0x7F
    ks_ar: int = 0x1F
    am_d1r: int = 0x00
    dt2_d2r: int = 0x00
    d1l_rr: int = 0xF7


@dataclass
class OpmState:
    rl_fb_con: list[int]
    kc: list[int]
    kf: list[int]
    pms_ams: list[int]
    ops: list[list[OpRegs]]
    lfrq: int = 0
    pmd: int = 0
    amd: int = 0
    wave: int = 0
    test: int = 0

    @classmethod
    def new(cls) -> "OpmState":
        return cls(
            rl_fb_con=[0xC0] * 8,
            kc=[0] * 8,
            kf=[0] * 8,
            pms_ams=[0] * 8,
            ops=[[OpRegs() for _ in range(4)] for _ in range(8)],
        )

    def write(self, reg: int, val: int) -> None:
        reg &= 0xFF
        val &= 0xFF
        if 0x20 <= reg <= 0x27:
            self.rl_fb_con[reg - 0x20] = val
        elif 0x28 <= reg <= 0x2F:
            self.kc[reg - 0x28] = val
        elif 0x30 <= reg <= 0x37:
            self.kf[reg - 0x30] = val
        elif 0x38 <= reg <= 0x3F:
            self.pms_ams[reg - 0x38] = val
        elif 0x40 <= reg <= 0xFF:
            ch = reg & 7
            if ch >= 8:
                return
            op = (reg >> 3) & 3
            slot = self.ops[ch][op]
            page = reg & 0xE0
            if page == 0x40:
                slot.dt_mul = val
            elif page == 0x60:
                slot.tl = val
            elif page == 0x80:
                slot.ks_ar = val
            elif page == 0xA0:
                slot.am_d1r = val
            elif page == 0xC0:
                slot.dt2_d2r = val
            elif page == 0xE0:
                slot.d1l_rr = val
        elif reg == 0x18:
            self.lfrq = val
        elif reg == 0x19:
            if val & 0x80:
                self.pmd = val & 0x7F
            else:
                self.amd = val & 0x7F
        elif reg == 0x1B:
            self.wave = val & 3
        elif reg == 0x01:
            self.test = val

    def patch(self, ch: int) -> tuple[int, int, tuple[tuple[int, ...], ...]]:
        con = self.rl_fb_con[ch]
        alg = con & 7
        fb = (con >> 3) & 7
        chan_ams = self.pms_ams[ch] & 3
        ops: list[tuple[int, ...]] = []
        for i in range(4):
            slot = self.ops[ch][i]
            ar = slot.ks_ar & 0x1F
            ks = (slot.ks_ar >> 6) & 3
            dr = slot.am_d1r & 0x1F
            ams = chan_ams if (slot.am_d1r & 0x80) else 0
            sr = slot.dt2_d2r & 0x1F
            dt2 = (slot.dt2_d2r >> 6) & 3
            rr = slot.d1l_rr & 0x0F
            sl = (slot.d1l_rr >> 4) & 0x0F
            tl = slot.tl & 0x7F
            mul = slot.dt_mul & 0x0F
            dt1 = (slot.dt_mul >> 4) & 0x07
            ops.append((ar, dr, sr, rr, sl, tl, ks, mul, dt1, dt2, ams))
        return alg, fb, tuple(ops)

    def carrier_tl(self, ch: int) -> int:
        alg = self.rl_fb_con[ch] & 7
        tls = [self.ops[ch][i].tl & 0x7F for i in CARRIERS.get(alg, (3,))]
        return min(tls) if tls else 0x7F

    def pan_gain(self, ch: int) -> tuple[float, float]:
        rl = (self.rl_fb_con[ch] >> 6) & 3
        # 1 = left, 2 = right, 3 = both, 0 = muted on the OPM.
        if rl == 1:
            return 0.0, 1.0
        if rl == 2:
            return 8.0, 1.0
        if rl == 3:
            return 4.0, 1.0
        return 4.0, 0.0

    def lfo(self, ch: int, clock: int) -> LfoState | None:
        pms = (self.pms_ams[ch] >> 4) & 7
        ams = self.pms_ams[ch] & 3
        if not self.pmd and not self.amd:
            return None
        # OPM LFO frequency: clock / (64 * 2^(LFRQ[7:4]) * (16 + LFRQ[3:0])).
        # SiON's @lfo argument is 60 frames per cycle.
        coarse = (self.lfrq >> 4) & 7
        fine = self.lfrq & 0x0F
        hz = clock / (64.0 * (2.0 ** coarse) * (16.0 + fine))
        period = max(1, min(255, int(round(60.0 / hz)))) if hz > 0 else 20
        pitch = int(round((self.pmd / 127.0) * (pms / 7.0) * 128.0)) if pms else 0
        amp = int(round((self.amd / 127.0) * (ams / 3.0) * 48.0)) if ams else 0
        if not pitch and not amp:
            return None
        return LfoState(period=period, wave=self.wave & 3, pitch=pitch, amp=amp)


def _apply_burst(state: OpmState, writes: list, start: int, ch: int) -> OpmState:
    """Copy state and fold a short post-key-on voice/pitch burst into it."""
    snap = copy.deepcopy(state)
    sample0 = writes[start].sample
    for w in writes[start + 1 :]:
        if w.sample - sample0 > BURST_SAMPLES:
            break
        reg = w.reg & 0xFF
        if reg == 0x08:
            continue
        if (reg & 7) != ch:
            continue
        if 0x40 <= reg <= 0xDF or 0x60 <= reg <= 0x7F:
            snap.write(reg, w.val)
        elif 0x28 <= reg <= 0x37:
            snap.write(reg, w.val)
    return snap


def extract_ym2151(
    vgm: VgmFile, voices: VoiceLibrary, order_base: int = 0
) -> ExtractResult:
    result = ExtractResult()
    instances = sorted({w.chip_id for w in vgm.writes if w.chip == "ym2151"}) or [0]
    stats = {"notes": 0, "keyons": 0, "tl_moves": 0, "lfo_notes": 0, "mute_moves": 0}
    for instance in instances:
        writes = sorted_writes(vgm, "ym2151", instance)
        if not writes:
            continue
        clock = vgm.clock("ym2151", instance)
        state = OpmState.new()
        timelines = [ChannelTimeline(f"YM2151 {instance + 1} ch {ch + 1}") for ch in range(8)]
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
        sounding = [False] * 8
        op_mask = [0] * 8
        base_tl = [0x7F] * 8
        mute = [False] * 8
        open_window = [-1] * 8
        last_lfo: list[LfoState | None] = [None] * 8

        for i, w in enumerate(writes):
            reg = w.reg & 0xFF
            val = w.val & 0xFF
            sample = w.sample
            if reg == 0x08:
                ch = val & 7
                mask = (val >> 3) & 0x0F
                if ch >= 8:
                    continue
                newly_on = mask & ~op_mask[ch]
                op_mask[ch] = mask
                if mask and newly_on:
                    snap = _apply_burst(state, writes, i, ch)
                    alg, fb, ops = snap.patch(ch)
                    voice = FmVoice("opm", alg, fb, ops)
                    pitch = opm_kc_units(snap.kc[ch], snap.kf[ch], clock)
                    pan, pan_gain = snap.pan_gain(ch)
                    tl0 = snap.carrier_tl(ch)
                    note = timelines[ch].note_on(
                        sample, pitch, voice, gain=pan_gain, pan=pan
                    )
                    base_tl[ch] = tl0
                    mute[ch] = pan_gain <= 0.0
                    sounding[ch] = True
                    open_window[ch] = sample + BURST_SAMPLES
                    lfo = snap.lfo(ch, clock)
                    if lfo is not None:
                        tracks[ch].lfo_points.append((sample, lfo))
                        last_lfo[ch] = lfo
                        stats["lfo_notes"] += 1
                    else:
                        tracks[ch].lfo_points.append(
                            (sample, LfoState(period=20, wave=2, pitch=0, amp=0))
                        )
                        last_lfo[ch] = None
                    stats["notes"] += 1
                    stats["keyons"] += 1
                elif not mask and sounding[ch]:
                    timelines[ch].note_off(sample)
                    sounding[ch] = False
                continue

            state.write(reg, val)
            ch = reg & 7
            if ch >= 8:
                continue

            if 0x28 <= reg <= 0x37 and sounding[ch]:
                pitch = opm_kc_units(state.kc[ch], state.kf[ch], clock)
                note = timelines[ch].open
                if note is not None and sample <= open_window[ch]:
                    note.pitch = pitch
                else:
                    timelines[ch].pitch(sample, pitch)
            elif 0x60 <= reg <= 0x7F and sounding[ch]:
                cur = state.carrier_tl(ch)
                delta_db = (cur - base_tl[ch]) * 0.375
                gain = 0.0 if mute[ch] else max(0.0, min(1.0, db_gain(delta_db)))
                if abs(gain - (timelines[ch].open.gain if timelines[ch].open else gain)) > 0.004:
                    timelines[ch].gain(sample, gain)
                    stats["tl_moves"] += 1
            elif 0x20 <= reg <= 0x27:
                pan, pan_gain = state.pan_gain(ch)
                if mute[ch] != (pan_gain <= 0.0):
                    stats["mute_moves"] += 1
                mute[ch] = pan_gain <= 0.0
                timelines[ch].pan(sample, pan)
                if sounding[ch]:
                    cur = state.carrier_tl(ch)
                    delta_db = (cur - base_tl[ch]) * 0.375
                    gain = 0.0 if mute[ch] else max(0.0, min(1.0, db_gain(delta_db)))
                    timelines[ch].gain(sample, gain)
            elif 0x38 <= reg <= 0x3F or reg in (0x18, 0x19, 0x1B):
                if sounding[ch]:
                    lfo = state.lfo(ch, clock)
                    if lfo is not None:
                        if last_lfo[ch] is None or lfo.key != last_lfo[ch].key:
                            tracks[ch].lfo_points.append((sample, lfo))
                            last_lfo[ch] = lfo
                    elif last_lfo[ch] is not None:
                        tracks[ch].lfo_points.append(
                            (sample, LfoState(period=20, wave=2, pitch=0, amp=0))
                        )
                        last_lfo[ch] = None

        for ch in range(8):
            timelines[ch].close_all(vgm.total_samples)
        result.tracks.extend(tracks)

    result.stats.update(stats)
    return result
