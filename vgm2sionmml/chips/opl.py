"""OPL-family decoder: YM3526 (OPL1), YM3812 (OPL2), YMF262 (OPL3).

Two-operator channels map to ``#OPL@`` voices with two operators.  OPL3's
four-operator pairs (0,3), (1,4), (2,5) per bank map to a four-operator voice
in the instrument order used by libSiON: [head op1, second op1, head op2,
second op2].  Rhythm mode and the OPL4 wave half are out of scope.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from ..model import ChannelTimeline, FmVoice, Track, VoiceLibrary
from ..pitch import freq_to_midi, midi_to_units, opl_freq
from ..vgm import VgmFile
from .common import ExtractResult, db_gain, sorted_writes

BURST_SAMPLES = 64
# OPL operator slot addresses per channel (0-8).
CHANNEL_SLOTS = (
    (0x00, 0x03),
    (0x01, 0x04),
    (0x02, 0x05),
    (0x08, 0x0B),
    (0x09, 0x0C),
    (0x0A, 0x0D),
    (0x10, 0x13),
    (0x11, 0x14),
    (0x12, 0x15),
)
SLOT_CHANNEL = {
    0x00: 0,
    0x03: 0,
    0x01: 1,
    0x04: 1,
    0x02: 2,
    0x05: 2,
    0x08: 3,
    0x0B: 3,
    0x09: 4,
    0x0C: 4,
    0x0A: 5,
    0x0D: 5,
    0x10: 6,
    0x13: 6,
    0x11: 7,
    0x14: 7,
    0x12: 8,
    0x15: 8,
}
# Carrier operator indices for the 4-op instrument order [1,2,3,4].
# OPL3 4-op chain order is [head op1, head op2, second op1, second op2].
# Algorithm 0 is FM-FM, 1 AM-FM (op1+op4), 2 FM-AM (op2+op4), 3 AM-AM
# (op1+op3+op4).  This matches libvgm's adlibemu_opl_inc.c topology.
FOUR_CARRIERS = {0: (3,), 1: (0, 3), 2: (1, 3), 3: (0, 2, 3)}
TWO_CARRIERS = {0: (1,), 1: (0, 1)}
# 0xBD trigger bit -> (name, channel, single operator slot or None for BD).
DRUMS = {
    0x10: ("BD", 6, None, 17640),
    0x08: ("SD", 7, 0x14, 8820),
    0x04: ("TOM", 8, 0x15, 8820),
    0x02: ("TC", 8, 0x12, 8820),
    0x01: ("HH", 7, 0x11, 4410),
}


@dataclass
class OplOp:
    am_vib_eg_ksr_mul: int = 0x00
    ksl_tl: int = 0x00
    ar_dr: int = 0x00
    sl_rr: int = 0x00
    wave: int = 0x00


@dataclass
class OplState:
    opl3: bool
    opl2: bool
    regs: dict[int, int]
    ops: dict[int, OplOp]
    four_op_mask: int = 0
    rhythm: bool = False

    @classmethod
    def new(cls, opl3: bool, opl2: bool) -> "OplState":
        return cls(opl3=opl3, opl2=opl2, regs={}, ops={})

    @property
    def four_op(self) -> bool:
        return bool(self.four_op_mask)

    def op(self, addr: int) -> OplOp:
        return self.ops.setdefault(addr, OplOp())

    def write(self, addr: int, val: int) -> None:
        addr &= 0x1FF
        val &= 0xFF
        self.regs[addr] = val
        r = addr & 0xFF
        bank = (addr >> 8) & 1
        if not self.opl3:
            bank = 0
            addr &= 0xFF
        if bank == 1 and r == 0x04:
            # Six per-pair enable bits: 0-3, 1-4, 2-5, 9-C, A-D, B-E.
            self.four_op_mask = val & 0x3F
            return
        if bank == 1 and r == 0x05:
            return
        if bank == 0 and r == 0xBD:
            self.rhythm = bool(val & 0x20)
            return
        # Operator pages share low five bits: base + slot, where slot is
        # 0x00-0x05 / 0x08-0x0D / 0x10-0x15.
        low = r & 0x1F
        page = r - low
        if page in (0x20, 0x40, 0x60, 0x80, 0xE0) and low <= 0x15:
            op_addr = (bank << 8) | low
            op = self.op(op_addr)
            if page == 0x20:
                op.am_vib_eg_ksr_mul = val
            elif page == 0x40:
                op.ksl_tl = val
            elif page == 0x60:
                op.ar_dr = val
            elif page == 0x80:
                op.sl_rr = val
            elif page == 0xE0:
                op.wave = val
            return

    def channel_base(self, ch: int) -> int:
        bank = 1 if ch >= 9 else 0
        local = ch % 9
        return (bank << 8) | local

    def pair_index(self, ch: int) -> int | None:
        bank = 1 if ch >= 9 else 0
        local = ch % 9
        if local in (0, 1, 2):
            return bank * 3 + local
        if local in (3, 4, 5):
            return bank * 3 + (local - 3)
        return None

    def pair_enabled(self, ch: int) -> bool:
        pair = self.pair_index(ch)
        return pair is not None and bool(self.four_op_mask & (1 << pair))

    def four_op_head(self, ch: int) -> int | None:
        """Return the head channel when ``ch`` is an enabled 4-op head."""
        local = ch % 9
        if local in (0, 1, 2) and self.pair_enabled(ch):
            return ch
        return None

    def four_op_second(self, head: int) -> int:
        bank = 1 if head >= 9 else 0
        local = head % 9
        return bank * 9 + local + 3

    def is_four_op_channel(self, ch: int) -> bool:
        return self.four_op_head(ch) is not None

    def is_four_op_second(self, ch: int) -> bool:
        local = ch % 9
        return local in (3, 4, 5) and self.pair_enabled(ch)

    def pitch_units(self, ch: int, clock: int) -> int:
        rb = self.channel_base(ch)
        a0 = self.regs.get(rb | 0xA0, 0)
        b0 = self.regs.get(rb | 0xB0, 0)
        fnum = ((b0 & 3) << 8) | a0
        block = (b0 >> 2) & 7
        return midi_to_units(freq_to_midi(opl_freq(block, fnum, clock, self.opl3)))

    def pan(self, ch: int) -> tuple[float, float]:
        if not self.opl3:
            return 4.0, 1.0
        rb = self.channel_base(ch)
        c0 = self.regs.get(rb | 0xC0, 0)
        left = bool(c0 & 0x10)
        right = bool(c0 & 0x20)
        if left and right:
            return 4.0, 1.0
        if left:
            return 0.0, 1.0
        if right:
            return 8.0, 1.0
        return 4.0, 0.0

    def _op_params(self, slot_addr: int) -> tuple[int, ...]:
        op = self.ops.get(slot_addr, OplOp())
        r20 = op.am_vib_eg_ksr_mul
        r40 = op.ksl_tl
        r60 = op.ar_dr
        r80 = op.sl_rr
        wave = op.wave & 7 if self.opl2 else 0
        ar = (r60 >> 4) & 0x0F
        dr = r60 & 0x0F
        rr = r80 & 0x0F
        egt = 1 if (r20 & 0x20) else 0
        sl = (r80 >> 4) & 0x0F
        tl = r40 & 0x3F
        ksr = (r20 >> 4) & 1
        ksl = (r40 >> 6) & 3
        mul = r20 & 0x0F
        # SiON's #OPL@ parser maps these three OPL multiplier table entries
        # back to the generic OPM table.  Keep the patch equivalent.
        if mul in (11, 13):
            mul -= 1
        elif mul == 14:
            mul += 1
        ams = 1 if (r20 & 0x80) else 0
        return (wave, ar, dr, rr, egt, sl, tl, ksr, ksl, mul, ams)

    def patch(self, ch: int) -> tuple[int, int, tuple[tuple[int, ...], ...]] | None:
        head = self.four_op_head(ch)
        if head is not None:
            second = self.four_op_second(head)
            hb = self.channel_base(head)
            sb = self.channel_base(second)
            bank_hi = hb & 0x100
            hc0 = self.regs.get(hb | 0xC0, 0)
            sc0 = self.regs.get(sb | 0xC0, 0)
            alg = (hc0 & 1) | ((sc0 & 1) << 1)
            fb = (hc0 >> 1) & 7
            h_slots = CHANNEL_SLOTS[head % 9]
            s_slots = CHANNEL_SLOTS[second % 9]
            ops = (
                self._op_params(bank_hi | h_slots[0]),
                self._op_params(bank_hi | h_slots[1]),
                self._op_params(bank_hi | s_slots[0]),
                self._op_params(bank_hi | s_slots[1]),
            )
            return alg, fb, ops
        if self.is_four_op_channel(ch):
            return None
        if self.is_four_op_second(ch):
            # The second channel of a four-operator pair shares the head
            # voice; it never gets a note of its own.
            return None
        rb = self.channel_base(ch)
        bank_hi = rb & 0x100
        c0 = self.regs.get(rb | 0xC0, 0)
        alg = c0 & 1
        fb = (c0 >> 1) & 7
        slots = CHANNEL_SLOTS[ch % 9]
        ops = (
            self._op_params(bank_hi | slots[0]),
            self._op_params(bank_hi | slots[1]),
        )
        return alg, fb, ops

    def carrier_tl(self, ch: int) -> int:
        head = self.four_op_head(ch)
        if head is not None:
            p = self.patch(ch)
            if p is None:
                return 0x3F
            alg = p[0]
            ops = p[2]
            return min(ops[i][6] for i in FOUR_CARRIERS.get(alg, (3,)))
        rb = self.channel_base(ch)
        bank_hi = rb & 0x100
        c0 = self.regs.get(rb | 0xC0, 0)
        alg = c0 & 1
        slots = CHANNEL_SLOTS[ch % 9]
        ops = (
            self._op_params(bank_hi | slots[0]),
            self._op_params(bank_hi | slots[1]),
        )
        return min(ops[i][6] for i in TWO_CARRIERS.get(alg, (1,)))


def _slot_operator(ch: int, low: int, op_count: int) -> int | None:
    """Map a physical OPL operator slot to SiON's operator index."""
    slots = CHANNEL_SLOTS[ch % 9]
    if low == slots[0]:
        return 0
    if low == slots[1]:
        return 1
    # Four-operator pairs use the second channel of the pair.
    return None


def _opl_voice(alg: int, fb: int, ops) -> tuple[FmVoice, int, tuple[tuple[int, int], ...]]:
    """Build a TL-normalized OPL voice and its per-note operator overrides.

    The carrier floor becomes the note's ``x`` level, as before.  Modulator
    TL is moved out of the voice too: SiON has 256 FM slots, while a driver
    that rides the FM depth writes a different modulator level on many notes.
    The renderer re-applies the logged modulator TL with ``iN@tl``, so one
    slot can serve every note of that timbre without changing its sound.
    """
    carriers = set(
        FOUR_CARRIERS.get(alg, (3,)) if len(ops) == 4
        else TWO_CARRIERS.get(alg, (1,))
    )
    floor = min((ops[i][6] for i in carriers), default=0)
    out = []
    overrides: list[tuple[int, int]] = []
    for i, op in enumerate(ops):
        values = list(op)
        if i in carriers:
            values[6] = max(0, values[6] - floor)
        else:
            overrides.append((i, values[6]))
            values[6] = 0
        out.append(tuple(values))
    return FmVoice("opl", alg, fb, tuple(out)), floor, tuple(overrides)


def _apply_burst(state: OplState, writes: list, start: int, ch: int) -> OplState:
    snap = copy.deepcopy(state)
    sample0 = writes[start].sample
    head = snap.four_op_head(ch)
    if head is not None:
        second = snap.four_op_second(head)
        locals_ = {head % 9, second % 9}
        banks = {1 if head >= 9 else 0}
    else:
        locals_ = {ch % 9}
        banks = {1 if ch >= 9 else 0}
    slots = set()
    for local in locals_:
        slots.update(CHANNEL_SLOTS[local])
    for w in writes[start + 1 :]:
        if w.sample - sample0 > BURST_SAMPLES:
            break
        addr = w.reg & 0x1FF
        r = addr & 0xFF
        bank = (addr >> 8) & 1 if state.opl3 else 0
        if bank not in banks:
            continue
        if 0xB0 <= r <= 0xB8 and bool(w.val & 0x20):
            continue
        low5 = r & 0x1F
        if low5 in slots:
            snap.write(addr, w.val)
            continue
        if (0xA0 <= r <= 0xA8 or 0xB0 <= r <= 0xB8 or 0xC0 <= r <= 0xC8) and (r & 0x0F) in locals_:
            snap.write(addr, w.val)
    return snap


def _opl_kind(kind: str) -> tuple[bool, bool]:
    if kind == "ymf262":
        return True, True
    if kind == "ym3812":
        return False, True
    return False, False


def extract_opl(
    vgm: VgmFile, voices: VoiceLibrary, kind: str, order_base: int = 0
) -> ExtractResult:
    result = ExtractResult()
    opl3, opl2 = _opl_kind(kind)
    instances = sorted({w.chip_id for w in vgm.writes if w.chip == kind}) or [0]
    total_channels = 18 if opl3 else 9
    stats = {"notes": 0, "keyons": 0, "tl_moves": 0, "four_op_notes": 0, "rhythm_skips": 0}
    rhythm_seen = False
    for instance in instances:
        writes = sorted_writes(vgm, kind, instance)
        if not writes:
            continue
        clock = vgm.clock(kind, instance) or (14_318_180 if opl3 else 3_579_545)
        state = OplState.new(opl3, opl2)
        timelines: list[ChannelTimeline | None] = []
        for ch in range(total_channels):
            timelines.append(ChannelTimeline(f"{kind.upper()} {instance + 1} ch {ch + 1}"))
        drum_timelines = {
            bit: ChannelTimeline(f"{kind.upper()} {instance + 1} {name}")
            for bit, (name, _ch, _slot, _dur) in DRUMS.items()
        }
        last_drum_sample = {bit: -1 for bit in DRUMS}
        sounding = [False] * total_channels
        base_tl = [0x3F] * total_channels
        mute = [False] * total_channels
        open_window = [-1] * total_channels
        voice_kind = [False] * total_channels

        # Group same-sample writes so a patch byte written next to the key-on
        # is already loaded when the note is snapshotted.
        grouped: list[list] = []
        for w in writes:
            if grouped and grouped[-1][-1].sample == w.sample:
                grouped[-1].append(w)
            else:
                grouped.append([w])
        ordered_writes: list = []
        for group in grouped:
            ordered_writes.extend(w for w in group if not _is_key_write(w))
            ordered_writes.extend(w for w in group if _is_key_write(w))

        for i, w in enumerate(ordered_writes):
            addr = w.reg & 0x1FF
            r = addr & 0xFF
            bank = (addr >> 8) & 1 if opl3 else 0
            val = w.val & 0xFF
            sample = w.sample
            if r == 0xBD and bank == 0:
                old_bd = state.regs.get(addr, 0)
                state.write(addr, val)
                if state.rhythm:
                    rhythm_seen = True
                    # Rhythm mode takes channels 6-8 away from the melodic
                    # section.  Convert its five one-shot drums instead of
                    # dropping the percussion.
                    for ch6 in (6, 7, 8):
                        if sounding[ch6]:
                            timelines[ch6].note_off(sample)
                            sounding[ch6] = False
                    trig = val & 0x1F
                    old_trig = (old_bd & 0x1F) if (old_bd & 0x20) else 0
                    if trig:
                        for bit, (name, ch6, slot, duration) in DRUMS.items():
                            if not (trig & bit) or (old_trig & bit):
                                continue
                            if last_drum_sample[bit] == sample:
                                continue
                            last_drum_sample[bit] = sample
                            hb = state.channel_base(ch6)
                            bank_hi = hb & 0x100
                            if slot is None:
                                c0 = state.regs.get(hb | 0xC0, 0)
                                fb = (c0 >> 1) & 7
                                slots = CHANNEL_SLOTS[ch6 % 9]
                                ops = (
                                    state._op_params(bank_hi | slots[0]),
                                    state._op_params(bank_hi | slots[1]),
                                )
                                voice = FmVoice("opl", 0, fb, ops)
                            else:
                                voice = FmVoice(
                                    "opl", 0, 0, (state._op_params(bank_hi | slot),)
                                )
                            pitch = state.pitch_units(ch6, clock)
                            pan, pan_gain = state.pan(ch6)
                            note = drum_timelines[bit].note_on(
                                sample, pitch, voice, gain=pan_gain, pan=pan
                            )
                            note.end = sample + duration
                            stats["drum_notes"] = stats.get("drum_notes", 0) + 1
                continue
            if opl3 and bank == 1 and r == 0x04:
                old_mask = state.four_op_mask
                state.write(addr, val)
                if state.four_op_mask != old_mask:
                    # A pair changed between 2-op and 4-op; a voice already
                    # sounding cannot be reinterpreted in place.
                    for ch in range(total_channels):
                        if sounding[ch]:
                            timelines[ch].close_all(sample)
                            sounding[ch] = False
                continue
            if 0xB0 <= r <= 0xB8:
                ch = bank * 9 + (r - 0xB0)
                old = state.regs.get(addr, 0)
                is_on = bool(val & 0x20)
                was_on = bool(old & 0x20)
                if state.is_four_op_second(ch):
                    # The second channel's key bit does not key the pair.
                    state.write(addr, val)
                    continue
                state.write(addr, val)
                if is_on and not was_on:
                    if state.rhythm and bank == 0 and (ch % 9) >= 6:
                        stats["rhythm_skips"] += 1
                        continue
                    snap = _apply_burst(state, ordered_writes, i, ch)
                    patch = snap.patch(ch)
                    if patch is None:
                        continue
                    alg, fb, ops = patch
                    voice, tl0, op_tl = _opl_voice(alg, fb, ops)
                    pitch = snap.pitch_units(ch, clock)
                    pan, pan_gain = snap.pan(ch)
                    note = timelines[ch].note_on(
                        sample,
                        pitch,
                        voice,
                        gain=pan_gain * db_gain(tl0 * 0.75),
                        pan=pan,
                    )
                    note.op_tl = op_tl
                    base_tl[ch] = tl0
                    mute[ch] = pan_gain <= 0.0
                    sounding[ch] = True
                    voice_kind[ch] = len(ops) == 4
                    open_window[ch] = sample + BURST_SAMPLES
                    stats["notes"] += 1
                    stats["keyons"] += 1
                    if len(ops) == 4:
                        stats["four_op_notes"] += 1
                elif was_on and not is_on and sounding[ch]:
                    timelines[ch].note_off(sample)
                    sounding[ch] = False
                if sounding[ch]:
                    pitch = state.pitch_units(ch, clock)
                    note = timelines[ch].open
                    if note is not None and sample <= open_window[ch]:
                        note.pitch = pitch
                    else:
                        timelines[ch].pitch(sample, pitch)
                continue

            state.write(addr, val)
            if 0xA0 <= r <= 0xA8:
                ch = bank * 9 + (r - 0xA0)
                if sounding[ch]:
                    pitch = state.pitch_units(ch, clock)
                    note = timelines[ch].open
                    if note is not None and sample <= open_window[ch]:
                        note.pitch = pitch
                    else:
                        timelines[ch].pitch(sample, pitch)
                continue
            if 0xC0 <= r <= 0xC8:
                ch = bank * 9 + (r - 0xC0)
                if sounding[ch]:
                    pan, pan_gain = state.pan(ch)
                    mute[ch] = pan_gain <= 0.0
                    timelines[ch].pan(sample, pan)
                    cur = state.carrier_tl(ch)
                    gain = 0.0 if mute[ch] else max(
                        0.0, min(1.0, db_gain(cur * 0.75))
                    )
                    timelines[ch].gain(sample, gain)
                continue
            low = r & 0x1F
            page = r - low
            if page == 0x40 and low <= 0x15:
                # TL pages 0x40-0x55.  The slot belongs to this channel, or
                # to the second channel of a four-op pair headed by this
                # channel.
                owner = SLOT_CHANNEL.get(low)
                if owner is None:
                    continue
                if owner in (3, 4, 5) and state.pair_enabled(bank * 9 + owner):
                    owner -= 3
                if state.four_op and owner in (0, 1, 2):
                    # A 4-op head owns its pair's slots; a 2-op channel 0-2
                    # in 4-op mode is not present.
                    ch = bank * 9 + owner
                elif state.four_op and owner >= 6:
                    ch = bank * 9 + owner
                else:
                    ch = bank * 9 + owner
                if 0 <= ch < total_channels and sounding[ch]:
                    alg = state.regs.get(state.channel_base(ch) | 0xC0, 0) & 1
                    ops = len(state.patch(ch)[2]) if state.patch(ch) else 2
                    carriers = set(
                        FOUR_CARRIERS.get(alg, (3,)) if ops == 4
                        else TWO_CARRIERS.get(alg, (1,))
                    )
                    op_index = _slot_operator(ch, low, ops)
                    if op_index is not None and op_index not in carriers:
                        timelines[ch].op_tl(
                            sample, op_index, val & 0x3F
                        )
                    else:
                        cur = state.carrier_tl(ch)
                        gain = 0.0 if mute[ch] else max(
                            0.0, min(1.0, db_gain(cur * 0.75))
                        )
                        timelines[ch].gain(sample, gain)
                    stats["tl_moves"] += 1

        for ch in range(total_channels):
            timelines[ch].close_all(vgm.total_samples)
        for tl in drum_timelines.values():
            tl.close_all(vgm.total_samples)
        for ch, tl in enumerate(timelines):
            if tl.notes:
                result.tracks.append(
                    Track(
                        label=tl.label,
                        module=6,
                        kind="fm",
                        timeline=tl,
                        order=order_base + instance * 18 + ch,
                    )
                )
        for bit, tl in drum_timelines.items():
            if tl.notes:
                result.tracks.append(
                    Track(
                        label=tl.label,
                        module=6,
                        kind="fm",
                        timeline=tl,
                        order=order_base + instance * total_channels + total_channels + bit,
                    )
                )

    if rhythm_seen:
        result.warnings.append(
            "OPL rhythm mode converted as five one-shot drum tracks; "
            "drum pitch and decay are approximations"
        )
    result.stats.update(stats)
    return result


def _is_key_write(w) -> bool:
    r = w.reg & 0xFF
    if 0xB0 <= r <= 0xB8:
        return bool(w.val & 0x20)
    if r == 0xBD:
        return bool(w.val & 0x20) or bool(w.val & 0x1F)
    return False
