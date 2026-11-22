"""Yamaha YM2413 (OPLL) and Konami VRC7 decoder."""

from __future__ import annotations

from dataclasses import dataclass

from ..model import ChannelTimeline, FmVoice, Track, VoiceLibrary
from ..pitch import freq_to_midi, midi_to_units, opll_freq
from ..vgm import VgmFile
from .common import ExtractResult, db_gain, sorted_writes
from .opll_banks import OPLL_PRESETS, VRC7_PRESETS

OPLL_CHANNELS = 9
VRC7_CHANNELS = 6
OPLL_REFIT = 150
OPLL_DRUM_DURATION = 17_640
OPLL_DRUM_HAT_DURATION = 8_820

# hw channel, label, volume register, nibble shift, trigger bit.
OPLL_DRUMS = (
    (6, "BD", 0x36, 0, 0x10, 16, (0, 1)),
    (7, "SD", 0x37, 0, 0x08, 17, (1,)),
    (8, "TM", 0x38, 4, 0x04, 18, (0,)),
    (8, "TP", 0x38, 0, 0x02, 18, (1,)),
    (7, "HH", 0x37, 4, 0x01, 17, (0,)),
)


def _dump_op(dump: bytes, index: int, tl: int) -> tuple[int, ...]:
    amvib = dump[index]
    ksl_tl = dump[2 + index]
    ardr = dump[4 + index]
    slrr = dump[6 + index]
    ws = 1 if (dump[3] & (0x08 << index)) else 0
    return (
        ws,
        (ardr >> 4) & 0x0F,
        ardr & 0x0F,
        slrr & 0x0F,
        1 if (amvib & 0x20) else 0,
        (slrr >> 4) & 0x0F,
        tl & 0x3F,
        1 if (amvib & 0x10) else 0,
        (ksl_tl >> 6) & 3,
        amvib & 0x0F,
        1 if (amvib & 0x80) else 0,
    )


def _patch_from_dump(dump: bytes, preset: int = 0) -> FmVoice:
    fb = dump[3] & 7
    mod = _dump_op(dump, 0, dump[2] & 0x3F)
    # OPLL's carrier level is the $3x low nibble, not a patch byte.
    car = _dump_op(dump, 1, 0)
    return FmVoice("opl", 0, fb, (mod, car))


def _custom_dump(regs: list[int]) -> bytes:
    return bytes(regs[0:8])


def _preset_dump(bank: tuple[int, ...], preset: int) -> bytes | None:
    if preset < 0 or preset * 8 + 8 > len(bank):
        return None
    off = preset * 8
    return bytes(bank[off : off + 8])


@dataclass
class _Voice:
    index: int
    fnum: int = 0
    block: int = 0
    keyed: bool = False
    preset: int = 0
    volume: int = 0
    note: object = None
    refit_until: int = 0


def extract_opll(
    vgm: VgmFile,
    voices: VoiceLibrary,
    *,
    instance: int = 0,
    order_base: int = 0,
    label: str | None = None,
) -> ExtractResult:
    result = ExtractResult()
    writes = sorted_writes(vgm, "ym2413", instance)
    if not writes:
        return result
    vrc7 = bool(vgm.chip_flags.get("ym2413_vrc7"))
    channels = VRC7_CHANNELS if vrc7 else OPLL_CHANNELS
    clock = vgm.clock("ym2413", instance) or 3_579_545
    bank = VRC7_PRESETS if vrc7 else OPLL_PRESETS
    name = label or ("VRC7" if vrc7 else "YM2413")
    timelines = [
        ChannelTimeline(f"{name} {instance + 1} ch {ch + 1}")
        for ch in range(channels)
    ]
    drum_timelines = {
        entry[1]: ChannelTimeline(f"{name} {instance + 1} {entry[1]}")
        for entry in OPLL_DRUMS
    }
    states = [_Voice(ch) for ch in range(channels)]
    custom_regs = [0] * 8
    rhythm = False
    rhythm_trig = 0
    active_drums: dict[int, list] = {}
    last_drum_sample = {entry[4]: -1 for entry in OPLL_DRUMS}
    stats = {
        "notes": 0,
        "drums": 0,
        "preset_notes": 0,
        "custom_notes": 0,
        "rhythm_skips": 0,
    }

    def patch_for(voice: _Voice) -> FmVoice:
        if voice.preset != 0:
            dump = _preset_dump(bank, voice.preset)
            if dump is not None:
                return _patch_from_dump(dump, voice.preset)
        return _patch_from_dump(_custom_dump(custom_regs), 0)

    def pitch(voice: _Voice) -> int:
        return midi_to_units(
            freq_to_midi(
                opll_freq(voice.block, voice.fnum, clock)
            )
        )

    def close(voice: _Voice, sample: int) -> None:
        note = voice.note
        if note is None:
            return
        note.close(sample)
        voice.note = None

    def start(voice: _Voice, sample: int) -> None:
        if voice.note is not None:
            close(voice, sample)
        note = timelines[voice.index].note_on(
            sample,
            pitch(voice),
            patch_for(voice),
            gain=db_gain(voice.volume * 3.0),
            pan=4.0,
            tag=f"{name} {voice.index + 1}",
        )
        voice.note = note
        voice.refit_until = sample + OPLL_REFIT
        stats["notes"] += 1
        if voice.preset:
            stats["preset_notes"] += 1
        else:
            stats["custom_notes"] += 1

    def update_pitch(voice: _Voice, sample: int) -> None:
        if voice.note is None:
            return
        value = pitch(voice)
        if sample <= voice.refit_until:
            voice.note.pitch = value
        else:
            voice.note.add_pitch(sample, value)

    def update_volume(voice: _Voice, sample: int) -> None:
        if voice.note is None:
            return
        gain = db_gain(voice.volume * 3.0)
        if sample <= voice.refit_until:
            voice.note.gain = gain
        else:
            voice.note.add_gain(sample, gain)

    def refresh_custom(sample: int) -> None:
        """Apply a custom-patch write to an already held custom voice."""
        for voice in states:
            if voice.note is None or voice.preset != 0:
                continue
            voice.note.voice = patch_for(voice)

    def update_drum_pitch(hw: int, sample: int) -> None:
        value = midi_to_units(
            freq_to_midi(opll_freq(states[hw].block, states[hw].fnum, clock))
        )
        live = []
        for note in active_drums.get(hw, ()):
            if note.end is not None and sample >= note.end:
                continue
            # Drum triggers often arrive before the linked channel's f-num
            # write; settle the base pitch instead of leaving the note at the
            # uninitialised (near-zero) register value.
            if sample <= note.start + OPLL_REFIT:
                note.pitch = value
            else:
                note.add_pitch(sample, value)
            live.append(note)
        active_drums[hw] = live

    def drum_note(entry, sample: int) -> None:
        hw, label_, vreg, shift, bit, preset, ops = entry
        voice = states[hw]
        dump = _preset_dump(bank, preset)
        if dump is None:
            return
        full = _patch_from_dump(dump, preset)
        chosen = tuple(full.ops[i] for i in ops if i < len(full.ops))
        drum_voice = FmVoice("opl", 0, full.fb, chosen)
        # Drum levels live in $36-$38; use the same 3 dB nibble scale.
        level = (regvals[vreg] >> shift) & 0x0F
        note = drum_timelines[label_].note_on(
            sample,
            midi_to_units(
                freq_to_midi(opll_freq(voice.block, voice.fnum, clock))
            ),
            drum_voice,
            gain=db_gain(level * 3.0),
            pan=4.0,
            tag=f"{name} {label_}",
        )
        note.end = sample + (
            OPLL_DRUM_HAT_DURATION if label_ == "HH" else OPLL_DRUM_DURATION
        )
        active_drums.setdefault(hw, []).append(note)
        stats["drums"] += 1

    regvals = [0] * 0x40

    for w in writes:
        reg = w.reg & 0x3F
        val = w.val & 0xFF
        sample = w.sample
        regvals[reg] = val
        if reg < 0x08:
            custom_regs[reg] = val
            refresh_custom(sample)
            continue
        if reg == 0x0E:
            if vrc7:
                continue
            was = rhythm
            rhythm = bool(val & 0x20)
            trig = (val & 0x1F) if rhythm else 0
            if was and not rhythm:
                for _hw, _label, *_rest in OPLL_DRUMS:
                    for note in active_drums.get(_hw, ()):
                        note.close(sample)
                    active_drums[_hw] = []
            if rhythm and not was:
                # Channels 6-8 switch from melodic voices to the fixed kit.
                for voice in states[6:]:
                    close(voice, sample)
            if rhythm:
                for entry in OPLL_DRUMS:
                    bit = entry[4]
                    if (trig & bit) and not (rhythm_trig & bit):
                        if last_drum_sample[bit] != sample:
                            last_drum_sample[bit] = sample
                            drum_note(entry, sample)
            rhythm_trig = trig
            continue

        if 0x10 <= reg <= 0x18:
            ch = reg - 0x10
            if ch < channels:
                voice = states[ch]
                voice.fnum = (voice.fnum & 0x100) | val
                if rhythm and ch >= 6:
                    update_drum_pitch(ch, sample)
                else:
                    update_pitch(voice, sample)
            continue

        if 0x20 <= reg <= 0x28:
            ch = reg - 0x20
            if ch >= channels:
                continue
            voice = states[ch]
            voice.fnum = (voice.fnum & 0xFF) | ((val & 1) << 8)
            voice.block = (val >> 1) & 7
            keyed = bool(val & 0x10)
            if rhythm and ch >= 6:
                # The kit owns these channels; their key bit is not a note.
                if keyed and not voice.keyed:
                    stats["rhythm_skips"] += 1
                voice.keyed = keyed
                update_drum_pitch(ch, sample)
                continue
            if keyed and not voice.keyed:
                start(voice, sample)
            elif not keyed and voice.keyed:
                close(voice, sample)
            voice.keyed = keyed
            update_pitch(voice, sample)
            continue

        if 0x30 <= reg <= 0x38:
            ch = reg - 0x30
            if ch >= channels:
                continue
            voice = states[ch]
            new_preset = (val >> 4) & 0x0F
            new_volume = val & 0x0F
            changed_preset = new_preset != voice.preset
            voice.preset = new_preset
            voice.volume = new_volume
            if rhythm and ch >= 6:
                # Drum levels use the same register; the trigger path reads
                # the current nibble.
                if voice.note is not None:
                    update_volume(voice, sample)
                continue
            if voice.note is not None:
                if changed_preset and sample > voice.refit_until:
                    close(voice, sample)
                    start(voice, sample)
                else:
                    update_volume(voice, sample)
            continue

    for voice in states:
        close(voice, vgm.total_samples)
    for timeline in drum_timelines.values():
        timeline.close_all(vgm.total_samples)
    for ch, voice in enumerate(states):
        if timelines[ch].notes:
            result.tracks.append(
                Track(
                    label=timelines[ch].label,
                    module=6,
                    kind="fm",
                    timeline=timelines[ch],
                    order=order_base + ch,
                )
            )
    for i, entry in enumerate(OPLL_DRUMS):
        timeline = drum_timelines[entry[1]]
        if timeline.notes:
            result.tracks.append(
                Track(
                    label=timeline.label,
                    module=6,
                    kind="fm",
                    timeline=timeline,
                    order=order_base + channels + i,
                )
            )
    result.stats.update(stats)
    if rhythm:
        result.warnings.append(
            "OPLL rhythm mode converted as five fixed drum tracks; "
            "drum pitch/level follow the linked registers"
        )
    if vrc7:
        result.warnings.append("YM2413 clock flag selects the VRC7 preset bank")
    return result
