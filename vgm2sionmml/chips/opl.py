"""OPL-family decoder: YM3526 (OPL1), YM3812 (OPL2), YMF262 (OPL3).

Two-operator channels map to ``#OPL@`` voices with two operators.  OPL3's
four-operator pairs (0,3), (1,4), (2,5) per bank map to a four-operator voice
in the instrument order used by libSiON: [head op1, second op1, head op2,
second op2].  Rhythm mode is converted as one-shot drums.  For SD, HH and
TC, the operators involved at note-on are captured and rebuilt as a per-hit
two-operator OPL ring source, which drives SiON's @r cross-channel modulation.
A shared square-wave ring layer supplies the pitched body.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from ..model import ChannelTimeline, FmVoice, SimpleVoice, Track, VoiceLibrary
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
# The snare and top-cymbal envelopes live on channel 7/8's second operator;
# the tom and hi-hat use their first operator.
DRUMS = {
    0x10: ("BD", 6, None, 17640),
    0x08: ("SD", 7, 0x14, 8820),
    0x04: ("TOM", 8, 0x12, 8820),
    0x02: ("TC", 8, 0x15, 8820),
    0x01: ("HH", 7, 0x11, 4410),
}
# SiON's @r ring modulation is a cross-channel effect.  At each rhythm
# trigger the involved OPL operators are captured and rebuilt as a per-hit
# two-operator instrument; that instrument is the ring source for the drum's
# one-operator envelope carrier.
RHYTHM_RING_BITS = (0x08, 0x02, 0x01)
RHYTHM_RING_LEVEL = 2
RHYTHM_RING_GAIN = {0x08: 0.25, 0x02: 0.18, 0x01: 0.16}
RHYTHM_RING_FEEDBACK = 7
# An additional square-wave ring layer supplies the pitched body that the
# noise-like phase interaction alone lacks.  It is intentionally shared: the
# drum gates, not the source, decide when it is heard.
RHYTHM_SQUARE_LEVEL = 2
RHYTHM_SQUARE_GAIN = {0x08: 0.18, 0x02: 0.09, 0x01: 0.08}
# Drum -> (phase source slot, carrier slot, modulator TL).  The source and
# carrier operators are captured at note-on and rebuilt as a two-operator OPL
# instrument; the TL values are the rhythmic modulation-depth interpretation
# because the phase source has no audible level of its own.
RHYTHM_RING_SPECS = {
    0x08: (0x11, 0x14, 8),   # snare
    0x02: (0x11, 0x15, 13),  # top cymbal
    0x01: (0x15, 0x11, 16),  # hi-hat
}


def _ring_operator(op: tuple[int, ...], tl: int) -> tuple[int, ...]:
    """Keep an operator's oscillator fields, but normalize its amplitude."""
    return (op[0], 15, 0, 0, 1, 0, tl, op[7], op[8], op[9], 0)


def _ring_voice(
    source_op: tuple[int, ...], carrier_op: tuple[int, ...], source_tl: int
) -> FmVoice:
    """A per-hit two-operator instrument interpreting the phase interaction."""
    return FmVoice(
        "opl",
        0,
        RHYTHM_RING_FEEDBACK,
        (_ring_operator(source_op, source_tl), _ring_operator(carrier_op, 0)),
    )


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


# OPL envelope timing tables from libvgm fmopl.c.  The rate index is
# ``16 + 4*raw_rate`` for non-zero rates; ksr adds the key-scaling offset.
_EG_SHIFT = [0] * 16
for _rate in range(13):
    _EG_SHIFT.extend([12 - _rate] * 4)
_EG_SHIFT.extend([0] * 12)  # rates 13, 14, 15
_EG_SHIFT.extend([0] * 16)

_EG_SELECT = [14 * 8] * 16
for _rate in range(13):
    _EG_SELECT.extend([0, 8, 16, 24])
_EG_SELECT.extend([32, 40, 48, 56])   # rate 13
_EG_SELECT.extend([64, 72, 80, 88])   # rate 14
_EG_SELECT.extend([96, 96, 96, 96])   # rate 15
_EG_SELECT.extend([96] * 16)          # dummy rates

_EG_INC = [
    0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 1, 1, 0, 1,
    0, 1, 1, 1, 0, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 2, 1, 1, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 2, 2, 1, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 4, 2, 2, 2, 4,
    2, 4, 2, 4, 2, 4, 2, 4,
    2, 4, 4, 4, 2, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4,
    8, 8, 8, 8, 8, 8, 8, 8,
    0, 0, 0, 0, 0, 0, 0, 0,
]
_SL_VOLUME = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 496]


def _opl_rate_index(raw: int, ksr: int) -> int:
    if raw == 0:
        return 0
    return 16 + (raw << 2) + (0 if ksr else 2)


def _opl_phase_samples(delta: int, raw_rate: int, ksr: int, clock: int) -> float | None:
    if delta <= 0:
        return 0.0
    index = _opl_rate_index(raw_rate, ksr)
    if index <= 0 or index >= len(_EG_SHIFT):
        return None
    shift = _EG_SHIFT[index]
    select = _EG_SELECT[index]
    inc = sum(_EG_INC[select : select + 8]) / 8.0
    if inc <= 0 or not clock:
        return None
    updates = (delta + inc - 1.0) / inc
    ticks = updates * (1 << shift)
    # The EG counter advances at clock/72; one sample is one output frame.
    return ticks * (72.0 * 44100.0 / clock)


def _opl_auto_release_samples(alg: int, ops, clock: int) -> int | None:
    """When ET=0 the envelope releases at the end of the S phase.

    SiON's ET flag alone does not shorten a held note in every player path,
    so return the key-on-to-silence distance for the carrier set.  This is
    the chip's attack + decay-to-SL + RR release, expressed in output samples.
    """
    carriers = FOUR_CARRIERS.get(alg, (3,)) if len(ops) == 4 else TWO_CARRIERS.get(alg, (1,))
    if not carriers:
        return None
    ends: list[float] = []
    for i in carriers:
        if i >= len(ops) or ops[i][4] != 0:
            return None
        op = ops[i]
        ar, dr, rr, sl = op[1], op[2], op[3], op[5]
        ksr = op[7]
        attack = _opl_phase_samples(511, ar, ksr, clock)
        decay = _opl_phase_samples(_SL_VOLUME[sl], dr, ksr, clock)
        release = _opl_phase_samples(511 - _SL_VOLUME[sl], rr, ksr, clock)
        if attack is None or decay is None or release is None:
            return None
        ends.append(attack + decay + release)
    if not ends:
        return None
    return max(1, int(round(max(ends))))


def _apply_opl_auto_release(note, alg: int, ops, clock: int, start: int) -> None:
    frames = _opl_auto_release_samples(alg, ops, clock)
    if frames is None:
        return
    end = start + frames
    if note.end is None or end < note.end:
        note.end = end


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
    vgm: VgmFile,
    voices: VoiceLibrary,
    kind: str,
    order_base: int = 0,
    instance: int | None = None,
) -> ExtractResult:
    result = ExtractResult()
    opl3, opl2 = _opl_kind(kind)
    instances = (
        [instance]
        if instance is not None
        else sorted({w.chip_id for w in vgm.writes if w.chip == kind}) or [0]
    )
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
        drum_tracks: dict[int, Track] = {}
        ring_timelines = {
            bit: ChannelTimeline(f"{kind.upper()} {instance + 1} {DRUMS[bit][0]} ring")
            for bit in RHYTHM_RING_BITS
        }
        ring_voice_cache: dict[FmVoice, FmVoice] = {}
        last_drum_sample = {bit: -1 for bit in DRUMS}
        # Rhythm drum notes share pitch registers with their linked FM
        # channels.  Keep them here so A0/B0 writes after the trigger retune
        # a sounding hit.
        active_drums: dict[int, list] = {6: [], 7: [], 8: []}
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

        def update_drum_pitch(channel: int, sample: int) -> None:
            if channel not in active_drums:
                return
            pitch = state.pitch_units(channel, clock)
            live = []
            for note in active_drums[channel]:
                end = note.end
                if end is not None and sample >= end:
                    continue
                if note.start <= sample:
                    note.add_pitch(sample, pitch)
                live.append(note)
            active_drums[channel] = live

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
                            if bit == 0x10:
                                # Bass drum is the normal FM channel 6
                                # (one-based channel 7), keyed by the rhythm
                                # trigger.  Keep it on the normal timeline so
                                # its pitch and live operator writes work.
                                patch = state.patch(ch6)
                                if patch is None:
                                    continue
                                alg, fb, ops = patch
                                voice, tl0, op_tl = _opl_voice(alg, fb, ops)
                                pitch = state.pitch_units(ch6, clock)
                                pan, pan_gain = state.pan(ch6)
                                note = timelines[ch6].note_on(
                                    sample, pitch, voice,
                                    gain=pan_gain * db_gain(tl0 * 0.75),
                                    pan=pan,
                                )
                                note.op_tl = op_tl
                                _apply_opl_auto_release(
                                    note, alg, ops, clock, sample
                                )
                                base_tl[ch6] = tl0
                                mute[ch6] = pan_gain <= 0.0
                                open_window[ch6] = sample + BURST_SAMPLES
                                active_drums[ch6].append(note)
                            else:
                                voice = FmVoice(
                                    "opl", 0, 0,
                                    (state._op_params(bank_hi | slot),),
                                )
                                pitch = state.pitch_units(ch6, clock)
                                pan, pan_gain = state.pan(ch6)
                                note = drum_timelines[bit].note_on(
                                    sample,
                                    pitch,
                                    voice,
                                    gain=pan_gain
                                    * RHYTHM_RING_GAIN.get(bit, 1.0),
                                    pan=pan,
                                )
                                active_drums[ch6].append(note)
                            note.end = sample + duration
                            _apply_opl_auto_release(
                                note, 0, voice.ops, clock, sample
                            )
                            if bit in RHYTHM_RING_SPECS:
                                source_slot, carrier_slot, source_tl = (
                                    RHYTHM_RING_SPECS[bit]
                                )
                                source_voice = _ring_voice(
                                    state._op_params(bank_hi | source_slot),
                                    state._op_params(bank_hi | carrier_slot),
                                    source_tl,
                                )
                                source_voice = ring_voice_cache.setdefault(
                                    source_voice, source_voice
                                )
                                source_ch = 7 if source_slot == 0x11 else 8
                                source_note = ring_timelines[bit].note_on(
                                    sample,
                                    state.pitch_units(source_ch, clock),
                                    source_voice,
                                    gain=1.0,
                                    pan=4.0,
                                )
                                source_note.end = note.end
                            stats["drum_notes"] = stats.get("drum_notes", 0) + 1
                            if bit in RHYTHM_RING_BITS:
                                stats["ring_mod_notes"] = (
                                    stats.get("ring_mod_notes", 0) + 1
                                )
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
                    _apply_opl_auto_release(
                        note, alg, ops, clock, sample
                    )
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
                if state.rhythm and bank == 0 and ch in active_drums:
                    update_drum_pitch(ch, sample)
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
                if state.rhythm and bank == 0 and ch in active_drums:
                    update_drum_pitch(ch, sample)
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
        for tl in ring_timelines.values():
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
                track = Track(
                    label=tl.label,
                    module=6,
                    kind="fm",
                    timeline=tl,
                    order=order_base + instance * total_channels + total_channels + bit,
                )
                drum_tracks[bit] = track
                result.tracks.append(track)
        for index, bit in enumerate(RHYTHM_RING_BITS):
            source_timeline = ring_timelines[bit]
            if not source_timeline.notes:
                continue
            pipe = (instance * len(RHYTHM_RING_BITS) + index) % 4
            if bit in drum_tracks:
                drum_tracks[bit].ring_mod = (RHYTHM_RING_LEVEL, pipe)
            result.tracks.append(
                Track(
                    label=source_timeline.label,
                    module=6,
                    kind="fm",
                    timeline=source_timeline,
                    output_pipe=pipe,
                    order=order_base - 1 - index,
                )
            )
            stats["ring_sources"] = stats.get("ring_sources", 0) + 1

        square_tracks = [
            drum_tracks[bit]
            for bit in RHYTHM_RING_BITS
            if bit in drum_tracks and drum_tracks[bit].timeline.notes
        ]
        if square_tracks:
            square_voice = SimpleVoice(5, 0, "square ring")
            square_timeline = ChannelTimeline(
                f"{kind.upper()} {instance + 1} square ring"
            )
            loop_sample = vgm.loop_sample
            if loop_sample is not None and 0 < loop_sample < vgm.total_samples:
                first = square_timeline.note_on(
                    0, midi_to_units(60), square_voice, gain=1.0, pan=4.0
                )
                first.end = loop_sample
                second = square_timeline.note_on(
                    loop_sample, midi_to_units(60), square_voice, gain=1.0, pan=4.0
                )
                second.end = vgm.total_samples
            else:
                square_note = square_timeline.note_on(
                    0, midi_to_units(60), square_voice, gain=1.0, pan=4.0
                )
                square_note.end = vgm.total_samples
            result.tracks.append(
                Track(
                    label=square_timeline.label,
                    module=5,
                    kind="psg",
                    timeline=square_timeline,
                    default_voice=square_voice,
                    output_pipe=3,
                    order=order_base - 10,
                )
            )
            stats["square_ring_sources"] = stats.get("square_ring_sources", 0) + 1
            for bit in RHYTHM_RING_BITS:
                original = drum_tracks.get(bit)
                if original is None or not original.timeline.notes:
                    continue
                square = copy.deepcopy(original)
                square.label = original.label + " square"
                square.order = original.order + 0.5
                square.ring_mod = (RHYTHM_SQUARE_LEVEL, 3)
                gain_ratio = (
                    RHYTHM_SQUARE_GAIN[bit] / RHYTHM_RING_GAIN[bit]
                )
                for note in square.timeline.notes:
                    note.gain *= gain_ratio
                result.tracks.append(square)

    if rhythm_seen:
        result.warnings.append(
            "OPL rhythm mode converted with normal FM channel 6 for the "
            "bass drum, linked-channel pitch tracking for drums, and @r "
            "ring modulation for snare/hi-hat/top cymbal and a square body layer; decay is approximate"
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
