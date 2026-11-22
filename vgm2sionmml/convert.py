"""Conversion orchestration: VGM write stream -> chip decoders -> SiON MML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .chips import (
    extract_ay8910,
    extract_gb,
    extract_huc6280,
    extract_k005289,
    extract_nes,
    extract_opl,
    extract_opll,
    extract_opn,
    extract_scc,
    extract_sn76489,
    extract_vsu,
    extract_ym2151,
)
from .chips.common import ExtractResult
from .mml import MmlRenderer
from .optimize import detect_bpm
from .envelopes import assign_envelopes
from .model import EnvelopeTableLibrary, Song, VoiceLibrary, WaveTableLibrary
from .vgm import VgmFile, load_vgm


class ConversionError(RuntimeError):
    pass


@dataclass
class ConversionResult:
    text: str
    source: Path
    title: str
    tracks: int
    notes: int
    voices: int
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)


OPN_KINDS = ("ym2203", "ym2612", "ym2608", "ym2610")
OPL_KINDS = ("ym3526", "ym3812", "ymf262")
HANDLED = {
    "ym2151",
    *OPN_KINDS,
    "ym2203_ssg",
    "ym2608_ssg",
    "ym2610_ssg",
    "ay8910",
    *OPL_KINDS,
    "ym2413",
    "sn76489",
    "nes",
    "k051649",
    "huc6280",
    "vsu",
    "gb",
    "k005289",
}


def _max_order(result: ExtractResult, fallback: int) -> int:
    if not result.tracks:
        return fallback
    return max(t.order for t in result.tracks) + 1


def convert_vgm(
    vgm: VgmFile,
    *,
    bpm: int | None = None,
    loop: bool = True,
    title: str | None = None,
    max_voices: int = 256,
    gain: float = 1.0,
    optimize: bool = True,
) -> ConversionResult:
    if bpm is not None and not 1 <= bpm <= 511:
        raise ConversionError("bpm must be between 1 and 511")
    if not 0.0 < gain <= 4.0:
        raise ConversionError("gain must be greater than 0 and at most 4")
    if not 1 <= max_voices <= 256:
        raise ConversionError("max_voices must be between 1 and 256")
    voices = VoiceLibrary(limit=max_voices)
    waves = WaveTableLibrary(limit=256)
    warnings: list[str] = list(vgm.warnings)
    stats: dict[str, object] = {"source_writes": len(vgm.writes)}
    tracks = []
    order = 0

    def absorb(result: ExtractResult, label: str) -> None:
        nonlocal order
        tracks.extend(result.tracks)
        warnings.extend(result.warnings)
        order = _max_order(result, order)
        if result.stats:
            stats[label] = dict(result.stats)

    # YM2151 (OPM)
    for instance in sorted({w.chip_id for w in vgm.writes if w.chip == "ym2151"}):
        absorb(
            extract_ym2151(vgm, voices, order_base=order),
            f"ym2151_{instance}",
        )

    # OPN family, then its AY-compatible SSG half.
    for kind in OPN_KINDS:
        for instance in sorted({w.chip_id for w in vgm.writes if w.chip == kind}):
            fm = extract_opn(vgm, voices, kind, order_base=order)
            absorb(fm, f"{kind}_{instance}")
            ssg_name = f"{kind}_ssg"
            if any(w.chip == ssg_name and w.chip_id == instance for w in vgm.writes):
                absorb(
                    extract_ay8910(
                        vgm,
                        voices,
                        chip=ssg_name,
                        instance=instance,
                        order_base=order,
                        label=f"{kind.upper()} SSG",
                    ),
                    f"{kind}_ssg_{instance}",
                )

    # Standalone AY-3-8910 / YM2149.
    for instance in sorted({w.chip_id for w in vgm.writes if w.chip == "ay8910"}):
        absorb(
            extract_ay8910(
                vgm,
                voices,
                chip="ay8910",
                instance=instance,
                order_base=order,
            ),
            f"ay8910_{instance}",
        )

    # OPL family.
    for kind in OPL_KINDS:
        for instance in sorted({w.chip_id for w in vgm.writes if w.chip == kind}):
            absorb(
                extract_opl(vgm, voices, kind, order_base=order),
                f"{kind}_{instance}",
            )

    # YM2413 / VRC7.
    for instance in sorted({w.chip_id for w in vgm.writes if w.chip == "ym2413"}):
        absorb(
            extract_opll(vgm, voices, instance=instance, order_base=order),
            f"ym2413_{instance}",
        )

    # SN76489 / T6W28.
    for instance in sorted({w.chip_id for w in vgm.writes if w.chip == "sn76489"}):
        absorb(
            extract_sn76489(vgm, voices, instance=instance, order_base=order),
            f"sn76489_{instance}",
        )

    # NES APU (DPCM ignored).
    for instance in sorted({w.chip_id for w in vgm.writes if w.chip == "nes"}):
        absorb(
            extract_nes(vgm, voices, instance=instance, order_base=order),
            f"nes_{instance}",
        )

    # Wavetable chips share SiON's %4 SCC module and its 256 #WAVB slots.
    for instance in sorted({w.chip_id for w in vgm.writes if w.chip == "k051649"}):
        result = extract_scc(
            vgm, waves, instance=instance, order_base=order, label="SCC"
        )
        absorb(result, f"k051649_{instance}")
    for instance in sorted({w.chip_id for w in vgm.writes if w.chip == "huc6280"}):
        result = extract_huc6280(
            vgm, waves, instance=instance, order_base=order, label="HuC6280"
        )
        absorb(result, f"huc6280_{instance}")
    for instance in sorted({w.chip_id for w in vgm.writes if w.chip == "vsu"}):
        result = extract_vsu(
            vgm, waves, instance=instance, order_base=order, label="VSU"
        )
        absorb(result, f"vsu_{instance}")
    for instance in sorted({w.chip_id for w in vgm.writes if w.chip == "gb"}):
        absorb(
            extract_gb(
                vgm, waves, instance=instance, order_base=order, label="Game Boy"
            ),
            f"gb_{instance}",
        )
    for instance in sorted({w.chip_id for w in vgm.writes if w.chip == "k005289"}):
        absorb(
            extract_k005289(
                vgm, waves, instance=instance, order_base=order,
                label="K005289",
            ),
            f"k005289_{instance}",
        )

    used = {w.chip for w in vgm.writes}
    if any(w.chip == "ym2608" and 0x10 <= (w.reg & 0xFF) <= 0x1F for w in vgm.writes):
        warnings.append("YM2608 rhythm/ADPCM writes ignored (PCM is out of scope)")
    if any(w.chip == "ym2610" and (w.reg & 0x100) and (w.reg & 0xFF) < 0x30 for w in vgm.writes):
        warnings.append("YM2610 ADPCM writes ignored (PCM is out of scope)")
    if any(w.chip == "nes_dmc" for w in vgm.writes):
        warnings.append("NES DPCM writes ignored (PCM is out of scope)")
    unsupported = sorted(used - HANDLED - {"nes_dmc"})
    if unsupported:
        warnings.append("unsupported chips ignored: " + ", ".join(unsupported))

    if gain != 1.0:
        for track in tracks:
            for note in track.timeline.notes:
                note.gain = max(0.0, min(1.0, note.gain * gain))
                note.gain_points = [
                    (t, max(0.0, min(1.0, v * gain)))
                    for t, v in note.gain_points
                ]

    tracks = [
        t for t in tracks
        if any(n.end is None or n.end > n.start for n in t.timeline.notes)
    ]
    tables = None
    envelope_stats: dict[str, int] = {}
    if optimize:
        tables, envelope_stats = assign_envelopes(tracks, vgm.total_samples)
    if len(tracks) > 128:
        warnings.append(
            f"{len(tracks)} tracks emitted; SiON's default maximum is 128"
        )
    if not tracks:
        raise ConversionError(
            "no convertible notes found (supported chips: YM2151, YM2203, "
            "YM2612, YM2608, YM2610, YM3526, YM3812, YMF262, YM2413, "
            "AY-3-8910, "
            "SN76489, NES APU, SCC, HuC6280, Virtual Boy VSU, Game Boy, "
            "K005289)"
        )

    auto_tempo = bpm is None
    tempo_score = 0.0
    if bpm is None:
        bpm, tempo_score = detect_bpm(tracks, vgm.total_samples, vgm.rate)

    title_text = title or vgm.gd3.get("title") or vgm.path.stem
    if vgm.gd3.get("game") and not title:
        title_text = f"{title_text} ({vgm.gd3['game']})"
    song = Song(
        title=title_text,
        tracks=tracks,
        voices=voices,
        waves=waves,
        tables=tables if tables is not None else EnvelopeTableLibrary(),
        total_samples=vgm.total_samples,
        loop_sample=vgm.loop_sample if loop else None,
        bpm=bpm,
        warnings=warnings,
    )
    renderer = MmlRenderer(song, optimize=optimize)
    text = renderer.render()
    notes = sum(len(t.timeline.notes) for t in tracks)
    stats["tracks"] = len(tracks)
    stats["notes"] = notes
    stats["voices"] = len(voices.voices)
    stats["waves"] = len(waves.tables)
    stats["envelope_tables"] = len(song.tables.tables)
    stats["envelope_stats"] = envelope_stats
    stats["bpm"] = bpm
    stats["gain"] = gain
    stats["duration_seconds"] = round(vgm.duration_seconds, 3)
    stats["loop_sample"] = song.loop_sample
    stats["render"] = {
        "notes": renderer.stats.notes,
        "tied_segments": renderer.stats.tied_segments,
        "rests": renderer.stats.rests,
        "loop_split": renderer.stats.loop_split,
        "clamped": renderer.stats.clamped,
        "original_chars": renderer.stats.original_chars,
        "optimized_chars": renderer.stats.optimized_chars,
        "macros": renderer.stats.macros,
        "loops": renderer.stats.loops,
    }
    stats["optimize"] = optimize
    stats["auto_tempo"] = auto_tempo
    stats["tempo_score"] = round(tempo_score, 6)
    if renderer.stats.clamped:
        warnings.append(
            f"{renderer.stats.clamped} note segment(s) above B9 were clamped"
        )
    if voices.overflow:
        warnings.append(
            f"{voices.overflow} note(s) reused a nearby FM voice after the "
            f"{max_voices}-voice SiON limit"
        )
    if waves.overflow:
        warnings.append(
            f"{waves.overflow} note(s) reused a nearby wavetable after the "
            "256-slot SiON limit"
        )
    # Keep the report readable when the same condition is found by two paths.
    seen: set[str] = set()
    warnings = [w for w in warnings if not (w in seen or seen.add(w))]
    return ConversionResult(
        text=text,
        source=vgm.path,
        title=title_text,
        tracks=len(tracks),
        notes=notes,
        voices=len(voices.voices),
        warnings=warnings,
        stats=stats,
    )


def convert_file(
    path: str | Path,
    *,
    bpm: int | None = None,
    loop: bool = True,
    title: str | None = None,
    max_voices: int = 256,
    gain: float = 1.0,
    optimize: bool = True,
) -> ConversionResult:
    vgm = load_vgm(path)
    return convert_vgm(
        vgm,
        bpm=bpm,
        loop=loop,
        title=title,
        max_voices=max_voices,
        gain=gain,
        optimize=optimize,
    )
