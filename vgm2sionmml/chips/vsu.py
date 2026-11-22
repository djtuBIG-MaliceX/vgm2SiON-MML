"""Nintendo Virtual Boy VSU (VSU-VUE) wavetable decoder."""

from __future__ import annotations

from dataclasses import dataclass

from ..model import ChannelTimeline, SimpleVoice, Track, WaveTableLibrary
from ..pitch import freq_to_midi, midi_to_units, vsu_freq
from ..vgm import VGM_RATE, VgmFile
from .common import ExtractResult, sorted_writes

VSU_CHANNELS = 6
VSU_WAVE_CHANNELS = 5
VSU_WAVE_LEN = 32
VSU_REFIT = 735
VSU_MIN_NOTE = 100
WAVE_BURST = 64
VSU_INTERVAL_CLOCKS = 76_800
VSU_ENVELOPE_CLOCKS = 76_800


def vsu_clock(vgm: VgmFile, instance: int) -> int:
    return vgm.clock("vsu", instance) or 5_000_000


def _wave_samples(raw) -> tuple[int, ...]:
    # VGMPlay subtracts 0x20 from the 6-bit wave byte, so this preserves
    # both the shape and the hardware's sign.
    return tuple(max(-128, min(127, ((v & 0x3F) - 0x20) * 4)) for v in raw)


@dataclass
class _Voice:
    index: int
    wave_ram: list[bytearray]
    period: int = 0
    vol: int = 0
    lrv: int = 0
    ev0_low: int = 0
    ev1: int = 0
    ram: int = 0
    intl: int = 0
    keyed: bool = False
    note: object = None
    refit_until: int = 0
    auto_end: int = 0
    env_value: int = 0
    env_reload: int = 0
    env_counter: int = 1
    env_next: int = 0


def _levels(lrv: int, env: int) -> tuple[float, float]:
    left = (lrv >> 4) & 0x0F
    right = lrv & 0x0F
    peak = max(left, right)
    if not peak or not env:
        return 0.0, 4.0
    pan = 4.0 + 4.0 * (right - left) / peak
    return (peak / 15.0) * (env / 15.0), pan


def extract_vsu(
    vgm: VgmFile,
    waves: WaveTableLibrary,
    *,
    instance: int = 0,
    order_base: int = 0,
    label: str | None = None,
) -> ExtractResult:
    result = ExtractResult()
    writes = sorted_writes(vgm, "vsu", instance)
    if not writes:
        return result
    clock = vsu_clock(vgm, instance)
    env_tick = max(1.0, VSU_ENVELOPE_CLOCKS * VGM_RATE / clock)
    interval_tick = max(1.0, VSU_INTERVAL_CLOCKS * VGM_RATE / clock)
    ram = [bytearray(VSU_WAVE_LEN) for _ in range(VSU_WAVE_CHANNELS)]
    dirty = [False] * VSU_WAVE_CHANNELS
    dirty_start = [0] * VSU_WAVE_CHANNELS
    voices = [_Voice(i, ram) for i in range(VSU_CHANNELS)]
    timelines = [
        ChannelTimeline(f"{label or 'VSU'} {instance + 1} ch {v + 1}")
        for v in range(VSU_CHANNELS)
    ]
    stats = {"notes": 0, "noise_notes": 0, "wave_changes": 0, "short_gates": 0}

    def settle(table: int, sample: int) -> None:
        if not dirty[table]:
            return
        started = dirty_start[table]
        dirty[table] = False
        image = _wave_samples(ram[table])
        idx = waves.add(image)
        for voice in voices:
            note = voice.note
            if voice.index >= VSU_WAVE_CHANNELS:
                continue
            if note is None or voice.ram != table:
                continue
            if started <= voice.refit_until:
                note.voice = SimpleVoice(4, idx, "VSU wave")
            elif not note.wave_points or note.wave_points[-1][1] != idx:
                note.add_wave(started, idx)
                stats["wave_changes"] += 1

    def settle_all(sample: int) -> None:
        for table in range(VSU_WAVE_CHANNELS):
            settle(table, sample)

    def env_gain(voice: _Voice, env: int) -> tuple[float, float]:
        return _levels(voice.lrv, env if voice.ev1 & 0x01 else voice.vol)

    def record_level(voice: _Voice, sample: int) -> None:
        note = voice.note
        if note is None:
            return
        gain, pan = env_gain(voice, voice.env_value)
        if sample <= voice.refit_until:
            note.gain = gain
            note.pan = pan
        else:
            note.add_gain(sample, gain)
            note.add_pan(sample, pan)

    def advance(voice: _Voice, sample: int) -> None:
        note = voice.note
        if note is None:
            return
        if voice.auto_end and voice.auto_end <= sample:
            note.close(voice.auto_end)
            voice.note = None
            voice.keyed = False
            voice.auto_end = 0
            return
        if not (voice.ev1 & 0x01):
            voice.env_next = max(voice.env_next, sample + env_tick)
            return
        guard = 0
        while voice.env_next <= sample and guard < 65536:
            guard += 1
            tick = int(round(voice.env_next))
            voice.env_next += env_tick
            voice.env_counter -= 1
            if voice.env_counter > 0:
                continue
            voice.env_counter = (voice.ev0_low & 0x07) + 1
            value = voice.env_value
            up = bool(voice.ev0_low & 0x08)
            if value < 15 and up:
                value += 1
            elif value > 0 and not up:
                value -= 1
            elif voice.ev1 & 0x02:
                value = voice.env_reload
            else:
                continue
            if value != voice.env_value:
                voice.env_value = value
                note.add_gain(tick, env_gain(voice, value)[0])

    def close(voice: _Voice, sample: int) -> None:
        note = voice.note
        if note is None:
            return
        if sample - note.start < VSU_MIN_NOTE:
            stats["short_gates"] += 1
        note.close(sample)
        voice.note = None
        voice.keyed = False
        voice.auto_end = 0

    def start(voice: _Voice, sample: int) -> None:
        settle_all(sample)
        gain, pan = env_gain(voice, voice.vol)
        period = (2048 - (voice.period & 0x7FF)) & 0x7FF
        if voice.index >= VSU_WAVE_CHANNELS:
            # Channel 5 is the LFSR noise voice.  SiON has a periodic-noise
            # module, so keep it as a real track instead of dropping it.
            note = timelines[voice.index].note_on(
                sample,
                60 * 64,
                SimpleVoice(2, 2, "VSU noise"),
                gain=gain,
                pan=pan,
                tag=f"VSU noise {voice.index + 1}",
            )
            voice.note = note
            voice.refit_until = sample + VSU_REFIT
            voice.env_value = voice.vol
            voice.env_reload = voice.vol
            voice.env_counter = (voice.ev0_low & 0x07) + 1
            voice.env_next = sample + env_tick
            if voice.intl & 0x20:
                voice.auto_end = sample + int(
                    round(((voice.intl & 0x1F) + 1) * interval_tick)
                )
            stats["noise_notes"] += 1
            return
        if voice.ram > 4:
            return
        image = _wave_samples(ram[voice.ram])
        idx = waves.add(image)
        pitch = midi_to_units(freq_to_midi(vsu_freq(period, clock)))
        note = timelines[voice.index].note_on(
            sample, pitch, SimpleVoice(4, idx, "VSU wave"),
            gain=gain, pan=pan, tag=f"VSU {voice.index + 1}",
        )
        voice.note = note
        voice.refit_until = sample + VSU_REFIT
        voice.env_value = voice.vol
        voice.env_reload = voice.vol
        voice.env_counter = (voice.ev0_low & 0x07) + 1
        voice.env_next = sample + env_tick
        if voice.intl & 0x20:
            voice.auto_end = sample + int(
                round(((voice.intl & 0x1F) + 1) * interval_tick)
            )
        stats["notes"] += 1

    def key_on(voice: _Voice, sample: int, val: int) -> None:
        old = voice.keyed
        voice.intl = val
        now = bool(val & 0x80)
        if now:
            if voice.note is not None:
                close(voice, sample)
            voice.keyed = True
            start(voice, sample)
        elif old:
            close(voice, sample)

    for w in writes:
        reg = w.reg & 0x7FFF
        val = w.val & 0xFF
        sample = w.sample
        if reg < 0x0A0:
            table = reg >> 5
            if table < VSU_WAVE_CHANNELS:
                if not dirty[table]:
                    dirty[table] = True
                    dirty_start[table] = sample
                ram[table][reg & 0x1F] = val & 0x3F
            continue
        if reg < 0x100:
            continue  # modulation table; not representable in SiON %4
        settle_all(sample)
        if reg == 0x160:
            if val & 1:
                for voice in voices:
                    close(voice, sample)
            continue
        if not 0x100 <= reg < 0x160:
            continue
        ch = (reg - 0x100) >> 4
        r = (reg - 0x100) & 0x0F
        if ch >= VSU_CHANNELS:
            continue
        voice = voices[ch]
        advance(voice, sample)

        if r == 0x00:
            key_on(voice, sample, val)
        elif r == 0x01:
            voice.lrv = val
            record_level(voice, sample)
        elif r == 0x02:
            voice.period = (voice.period & 0x700) | val
            if voice.note is not None:
                period = (2048 - (voice.period & 0x7FF)) & 0x7FF
                pitch = midi_to_units(freq_to_midi(vsu_freq(period, clock)))
                if sample <= voice.refit_until:
                    voice.note.pitch = pitch
                else:
                    voice.note.add_pitch(sample, pitch)
        elif r == 0x03:
            voice.period = (voice.period & 0x0FF) | ((val & 0x07) << 8)
            if voice.note is not None:
                period = (2048 - (voice.period & 0x7FF)) & 0x7FF
                pitch = midi_to_units(freq_to_midi(vsu_freq(period, clock)))
                if sample <= voice.refit_until:
                    voice.note.pitch = pitch
                else:
                    voice.note.add_pitch(sample, pitch)
        elif r == 0x04:
            old_vol = voice.vol
            new_vol = (val >> 4) & 0x0F
            voice.vol = new_vol
            voice.ev0_low = val & 0x0F
            voice.env_value = new_vol
            voice.env_reload = new_vol
            voice.env_counter = (voice.ev0_low & 0x07) + 1
            voice.env_next = sample + env_tick
            if (
                voice.note is not None
                and sample > voice.refit_until
                and new_vol > old_vol
            ):
                # Jack Bros and similar drivers articulate by raising EV0's
                # volume while the gate stays set.  That is a new hit.
                close(voice, sample)
                start(voice, sample)
            else:
                record_level(voice, sample)
        elif r == 0x05:
            was = bool(voice.ev1 & 0x01)
            voice.ev1 = val
            if not was and (val & 0x01):
                voice.env_value = voice.vol
                voice.env_counter = (voice.ev0_low & 0x07) + 1
                voice.env_next = sample + env_tick
            record_level(voice, sample)
        elif r == 0x06:
            old_ram = voice.ram
            voice.ram = val & 0x0F
            if voice.index >= VSU_WAVE_CHANNELS:
                continue
            if voice.note is not None and voice.ram != old_ram:
                if voice.ram > 4:
                    close(voice, sample)
                else:
                    idx = waves.add(_wave_samples(ram[voice.ram]))
                    if sample <= voice.refit_until:
                        voice.note.voice = SimpleVoice(4, idx, "VSU wave")
                    else:
                        voice.note.add_wave(sample, idx)
                        stats["wave_changes"] += 1

    end = vgm.total_samples
    settle_all(end)
    for voice in voices:
        advance(voice, end)
        close(voice, end)
    for v in range(VSU_CHANNELS):
        tl = timelines[v]
        if not tl.notes:
            continue
        if v >= VSU_WAVE_CHANNELS:
            result.tracks.append(
                Track(
                    label=tl.label,
                    module=2,
                    kind="psg",
                    timeline=tl,
                    default_voice=SimpleVoice(2, 2, "VSU noise"),
                    order=order_base + v,
                )
            )
        else:
            result.tracks.append(
                Track(
                    label=tl.label,
                    module=4,
                    kind="wavetable",
                    timeline=tl,
                    default_voice=SimpleVoice(4, 0, "VSU wave"),
                    order=order_base + v,
                )
            )
    result.stats.update(stats)
    return result
