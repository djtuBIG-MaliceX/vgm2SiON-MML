"""VGM 1.00-1.71 / VGZ reader.

Only the parts needed by the converter are modelled: the header clocks, the
GD3 title, the flat chip-write stream, timing, and the loop point.  PCM sample
banks are deliberately skipped; command sizes are still honoured so a log that
contains them stays in sync.

The register encodings follow libvgm and vgm2151fur:

* YM2612/YM2608/YM2610 port 1 registers are stored as ``0x100 | reg``.
* OPL3 ports are stored as ``port << 8 | reg``.
* OPN SSG writes are emitted under a separate ``*_ssg`` chip name because the
  SSG is an AY-3-8910-compatible core clocked at half the FM master clock.
* SN76489 writes are serial bytes and are stored in ``reg=0``.
"""

from __future__ import annotations

import gzip
import zlib
from dataclasses import dataclass, field
from pathlib import Path

VGM_RATE = 44100
CLOCK_MIN = 100_000
CLOCK_MAX = 80_000_000

# Main-header clock fields.  The high bits are flags, so the clock itself is
# masked with 0x3FFFFFFF unless noted otherwise.
HDR_SN76489 = 0x0C
HDR_YM2413 = 0x10
HDR_YM2612 = 0x2C
HDR_YM2151 = 0x30
HDR_YM2203 = 0x44
HDR_YM2608 = 0x48
HDR_YM2610 = 0x4C
HDR_YM3812 = 0x50
HDR_YM3526 = 0x54
HDR_OPL3 = 0x5C
HDR_AY8910 = 0x74
HDR_GB = 0x80
HDR_NES = 0x84
HDR_SCC = 0x9C
HDR_HUC6280 = 0xA4
HDR_VSU = 0xC4
HDR_K005289 = 0xEC

DEFAULT_CLOCKS = {
    "ym2151": 3_579_545,
    "ym2612": 7_670_453,
    "ym2203": 3_579_545,
    "ym2608": 7_987_200,
    "ym2610": 8_000_000,
    "ym3812": 3_579_545,
    "ym3526": 3_579_545,
    "ym2413": 3_579_545,
    "ymf262": 14_318_180,
    "ay8910": 1_789_772,
    "sn76489": 3_579_545,
    "nes": 1_789_773,
    "gb": 4_194_304,
    "k051649": 1_789_772,
    "huc6280": 3_579_545,
    "vsu": 5_000_000,
    "k005289": 3_579_545,
}

# Extra-header chip ids (libvgm device indices).
EXTRA_CHIP_NAMES = {
    0x00: "sn76489",
    0x01: "ym2413",
    0x02: "ym2612",
    0x03: "ym2151",
    0x05: "rf5c68",
    0x06: "ym2203",
    0x07: "ym2608",
    0x08: "ym2610",
    0x09: "ym3812",
    0x0A: "ym3526",
    0x0B: "y8950",
    0x0C: "ymf262",
    0x0D: "ymf278b",
    0x0E: "ymf271",
    0x0F: "ymz280b",
    0x10: "rf5c164",
    0x11: "pwm",
    0x12: "ay8910",
    0x13: "gb",
    0x14: "nes",
    0x15: "multipcm",
    0x16: "upd7759",
    0x17: "msm6258",
    0x18: "msm6295",
    0x19: "k051649",
    0x1A: "k054539",
    0x1B: "huc6280",
    0x1C: "c140",
    0x1D: "k053260",
    0x1E: "pokey",
    0x1F: "qsound",
    0x20: "scsp",
    0x21: "wonderswan",
    0x22: "vsu",
    0x23: "saa1099",
    0x24: "es5503",
    0x25: "es5506",
    0x26: "x1_010",
    0x27: "c352",
    0x28: "ga20",
    0x29: "mikey",
    0x2A: "k007232",
}


@dataclass
class ChipWrite:
    """One register write, ordered by sample time.

    ``chip`` is a canonical lower-case name (``ym2151``, ``ym2612``,
    ``ym2608_ssg`` ...).  ``chip_id`` selects a second instance when the VGM
    has one.  ``reg`` is flat as described in the module docstring.
    """

    sample: int
    chip: str
    chip_id: int
    reg: int
    val: int


@dataclass
class DataBlock:
    block_type: int
    start: int
    data: bytes
    sample: int


@dataclass
class VgmFile:
    path: Path
    version: int
    total_samples: int
    loop_samples: int
    loop_sample: int | None
    gd3: dict[str, str] = field(default_factory=dict)
    writes: list[ChipWrite] = field(default_factory=list)
    clocks: dict[tuple[str, int], int] = field(default_factory=dict)
    chip_flags: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    rate: int = 60
    header_clocks: dict[str, int] = field(default_factory=dict)
    unsupported_writes: dict[str, int] = field(default_factory=dict)
    data_blocks: list[DataBlock] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return self.total_samples / VGM_RATE

    def clock(self, chip: str, instance: int = 0) -> int:
        """Best clock for a chip, falling back to the hardware default."""
        got = self.clocks.get((chip, instance))
        if got:
            return got
        # A few names share the same silicon family.
        aliases = {
            "ym2203_ssg": ("ay8910",),
            "ym2608_ssg": ("ay8910",),
            "ym2610_ssg": ("ay8910",),
        }
        for alias in aliases.get(chip, ()):
            got = self.clocks.get((alias, instance))
            if got:
                return got
        return DEFAULT_CLOCKS.get(chip, 0)

    def chips_used(self) -> list[str]:
        seen: list[str] = []
        for w in self.writes:
            if w.chip not in seen:
                seen.append(w.chip)
        return seen

    def write_count(self, chip: str) -> int:
        return sum(1 for w in self.writes if w.chip == chip)


def _u32(buf: bytes, off: int) -> int:
    if off < 0 or off + 4 > len(buf):
        return 0
    return int.from_bytes(buf[off : off + 4], "little")


def _u16(buf: bytes, off: int) -> int:
    if off < 0 or off + 2 > len(buf):
        return 0
    return int.from_bytes(buf[off : off + 2], "little")


def _sane_clock(raw: int) -> int:
    value = raw & 0x3FFFFFFF
    return value if CLOCK_MIN <= value <= CLOCK_MAX else 0


def read_vgm_bytes(path: Path | str) -> bytes:
    data = Path(path).read_bytes()
    if data[:2] == b"\x1f\x8b":
        try:
            return gzip.decompress(data)
        except OSError as exc:
            raise ValueError(f"{path}: damaged gzip stream: {exc}") from exc
    return data


def data_offset(buf: bytes) -> int:
    if len(buf) < 0x40 or buf[:4] != b"Vgm ":
        raise ValueError(f"not a VGM: {buf[:4]!r}")
    version = _u32(buf, 0x08)
    if version < 0x150:
        return 0x40
    rel = _u32(buf, 0x34)
    off = 0x34 + rel if rel else 0x40
    if off < 0x40 or off > len(buf):
        return 0x40
    return off


def _parse_clock_list(buf: bytes, off: int) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    if off < 0 or off + 4 > len(buf):
        return out
    count = _u32(buf, off)
    if count > 1024:
        return out
    pos = off + 4
    for _ in range(count):
        if pos + 5 > len(buf):
            break
        chip_id = buf[pos]
        clock = _u32(buf, pos + 1) & 0x3FFFFFFF
        if CLOCK_MIN <= clock <= CLOCK_MAX:
            out.setdefault(chip_id, []).append(clock)
        pos += 5
    return out


def _parse_gd3(buf: bytes, off: int) -> dict[str, str]:
    if not off:
        return {}
    # A few older rips point a few bytes before the chunk (or use a data
    # block around it).  Find the actual tag when the header value is close.
    if off + 12 > len(buf) or buf[off : off + 4] != b"Gd3 ":
        lo = max(0, off - 64)
        hi = min(len(buf), off + 64)
        found = buf.find(b"Gd3 ", lo, hi)
        if found < 0:
            found = buf.find(b"Gd3 ")
        if found >= 0:
            off = found
    if off + 12 > len(buf) or buf[off : off + 4] != b"Gd3 ":
        return {}
    length = _u32(buf, off + 8)
    start = off + 12
    end = min(len(buf), start + length)
    body = buf[start:end]
    fields = [
        "title",
        "title_jp",
        "game",
        "game_jp",
        "system",
        "system_jp",
        "author",
        "author_jp",
        "date",
        "ripper",
        "notes",
    ]
    try:
        text = body.decode("utf-16-le", errors="replace")
    except Exception:
        return {}
    parts = text.split("\x00")
    out: dict[str, str] = {}
    for name, value in zip(fields, parts):
        value = value.strip("\x00").strip()
        if value:
            out[name] = value
    return out


def _extra_clocks(buf: bytes) -> tuple[dict[int, list[int]], dict[int, int]]:
    """Return (extra clocks by id, extra volume entries by id)."""
    clocks: dict[int, list[int]] = {}
    volumes: dict[int, int] = {}
    if len(buf) < 0xC0:
        return clocks, volumes
    rel = _u32(buf, 0xBC)
    if not rel:
        return clocks, volumes
    extra = 0xBC + rel
    if extra + 8 > len(buf):
        return clocks, volumes
    size = _u32(buf, extra)
    if size < 8:
        return clocks, volumes
    off1 = _u32(buf, extra + 4)
    if off1:
        clocks = _parse_clock_list(buf, extra + 4 + off1)
    if size >= 12 and extra + 12 <= len(buf):
        off2 = _u32(buf, extra + 8)
        if off2:
            pos = extra + 8 + off2
            if pos + 4 <= len(buf):
                count = _u32(buf, pos)
                pos += 4
                for _ in range(min(count, 1024)):
                    if pos + 2 > len(buf):
                        break
                    chip_id = buf[pos]
                    vol = buf[pos + 1]
                    volumes[chip_id] = vol
                    pos += 2
    return clocks, volumes


# VGM command sizes.  0x67 is variable and handled separately.
def cmd_size(cmd: int) -> int:
    if cmd in (0x62, 0x63, 0x66) or 0x70 <= cmd <= 0x8F:
        return 1
    if cmd in (0x4F, 0x50, 0x94) or 0x30 <= cmd <= 0x3F:
        return 2
    if (
        cmd in (0x2D, 0x61)
        or 0x40 <= cmd <= 0x4E
        or 0x51 <= cmd <= 0x5F
        or 0xA0 <= cmd <= 0xBF
    ):
        return 3
    if 0xC0 <= cmd <= 0xDF:
        return 4
    if cmd in (0x90, 0x91, 0x95) or 0xE0 <= cmd <= 0xFF:
        return 5
    if cmd == 0x92:
        return 6
    if cmd == 0x93:
        return 11
    if cmd == 0x68:
        return 12
    return 2


def _add_main_clock(out: dict[tuple[str, int], int], name: str, clock: int, instance: int = 0) -> None:
    if clock and CLOCK_MIN <= clock <= CLOCK_MAX:
        out[(name, instance)] = clock


def _header_flags(buf: bytes, off: int) -> int:
    return _u32(buf, off)


def load_vgm(path: Path | str) -> VgmFile:
    """Load a ``.vgm`` or ``.vgz`` into a flat, time-stamped write stream."""
    path = Path(path)
    raw = read_vgm_bytes(path)
    if raw[:4] != b"Vgm ":
        raise ValueError(f"{path}: not a VGM file")
    version = _u32(raw, 8)
    total_samples = _u32(raw, 0x18)
    loop_rel = _u32(raw, 0x1C)
    loop_samples = _u32(raw, 0x20)
    rate = _u16(raw, 0x24) or 60
    gd3 = _parse_gd3(raw, _u32(raw, 0x14))
    extra_clocks, _extra_volumes = _extra_clocks(raw)

    clocks: dict[tuple[str, int], int] = {}
    flags: dict[str, int] = {}

    # YM2151
    raw2151 = _header_flags(raw, HDR_YM2151)
    ym2151_dual = bool(raw2151 & 0x40000000)
    _add_main_clock(clocks, "ym2151", _sane_clock(raw2151))
    flags["ym2151_dual"] = int(ym2151_dual)

    # YM2612
    raw2612 = _header_flags(raw, HDR_YM2612)
    ym2612_dual = bool(raw2612 & 0x40000000)
    _add_main_clock(clocks, "ym2612", _sane_clock(raw2612))
    flags["ym2612_dual"] = int(ym2612_dual)

    # YM2203
    raw2203 = _header_flags(raw, HDR_YM2203)
    ym2203_dual = bool(raw2203 & 0x40000000)
    _add_main_clock(clocks, "ym2203", _sane_clock(raw2203))
    flags["ym2203_dual"] = int(ym2203_dual)

    # YM2608
    raw2608 = _header_flags(raw, HDR_YM2608)
    ym2608_dual = bool(raw2608 & 0x40000000)
    _add_main_clock(clocks, "ym2608", _sane_clock(raw2608))
    flags["ym2608_dual"] = int(ym2608_dual)

    # YM2610
    raw2610 = _header_flags(raw, HDR_YM2610)
    ym2610_b = bool(raw2610 & 0x80000000)
    ym2610_dual = bool(raw2610 & 0x40000000)
    _add_main_clock(clocks, "ym2610", _sane_clock(raw2610))
    flags["ym2610_b"] = int(ym2610_b)
    flags["ym2610_dual"] = int(ym2610_dual)

    # OPL family
    raw3812 = _header_flags(raw, HDR_YM3812)
    _add_main_clock(clocks, "ym3812", _sane_clock(raw3812))
    flags["ym3812_dual"] = int(bool(raw3812 & 0x40000000))
    _add_main_clock(clocks, "ym3526", _sane_clock(_header_flags(raw, HDR_YM3526)))
    _add_main_clock(clocks, "ymf262", _sane_clock(_header_flags(raw, HDR_OPL3)))
    raw2413 = _header_flags(raw, HDR_YM2413)
    _add_main_clock(clocks, "ym2413", _sane_clock(raw2413))
    flags["ym2413_dual"] = int(bool(raw2413 & 0x40000000))
    flags["ym2413_vrc7"] = int(bool(raw2413 & 0x80000000))

    # AY-3-8910 / YM2149
    raw_ay = _header_flags(raw, HDR_AY8910)
    _add_main_clock(clocks, "ay8910", _sane_clock(raw_ay))
    flags["ay8910_dual"] = int(bool(raw_ay & 0x40000000))
    flags["ay8910_type"] = raw[0x78] if len(raw) > 0x78 else 0

    # SN76489 / T6W28
    raw_sn = _header_flags(raw, HDR_SN76489)
    _add_main_clock(clocks, "sn76489", _sane_clock(raw_sn))
    flags["sn76489_dual"] = int(bool(raw_sn & 0x40000000))
    flags["t6w28"] = int(bool(raw_sn & 0x80000000))

    # NES APU
    raw_nes = _header_flags(raw, HDR_NES)
    _add_main_clock(clocks, "nes", _sane_clock(raw_nes))
    flags["nes_dual"] = int(bool(raw_nes & 0x40000000))
    flags["nes_fds"] = int(bool(raw_nes & 0x80000000))

    # Game Boy (recognized, not converted)
    _add_main_clock(clocks, "gb", _sane_clock(_header_flags(raw, HDR_GB)))

    # Konami SCC / Hudson HuC6280 / Virtual Boy VSU.  The high bits of
    # these clock words carry dual/plus flags instead of clock bits.
    raw_scc = _header_flags(raw, HDR_SCC)
    _add_main_clock(clocks, "k051649", _sane_clock(raw_scc))
    flags["k051649_dual"] = int(bool(raw_scc & 0x40000000))
    flags["k051649_plus"] = int(bool(raw_scc & 0x80000000))

    raw_huc = _header_flags(raw, HDR_HUC6280)
    _add_main_clock(clocks, "huc6280", _sane_clock(raw_huc))
    flags["huc6280_dual"] = int(bool(raw_huc & 0x40000000))

    raw_vsu = _header_flags(raw, HDR_VSU)
    _add_main_clock(clocks, "vsu", _sane_clock(raw_vsu))
    flags["vsu_dual"] = int(bool(raw_vsu & 0x40000000))

    # Konami K005289 Bubble System WSG (VGM 1.71).
    _add_main_clock(clocks, "k005289", _sane_clock(_header_flags(raw, HDR_K005289)))

    # Extra header clock list.  Repeated ids are instances in order; entries
    # already supplied by a main field fill instance 0 first.
    for chip_id, values in extra_clocks.items():
        name = EXTRA_CHIP_NAMES.get(chip_id)
        if not name:
            continue
        instance = 0
        for clock in values:
            while (name, instance) in clocks:
                instance += 1
            clocks[(name, instance)] = clock
            instance += 1

    # Derived SSG clocks.  The YM2203's AY-compatible core runs at master/2;
    # the YM2608 and YM2610 cores run at master/4 (verified with VGMPlay).
    for fm_name, ssg_name, divisor in (
        ("ym2203", "ym2203_ssg", 2),
        ("ym2608", "ym2608_ssg", 4),
        ("ym2610", "ym2610_ssg", 4),
    ):
        for instance in range(2):
            clock = clocks.get((fm_name, instance))
            if clock:
                clocks[(ssg_name, instance)] = clock // divisor

    # Fill defaults for every clock that was used but not declared.
    for name, default in DEFAULT_CLOCKS.items():
        if (name, 0) not in clocks:
            clocks[(name, 0)] = default

    writes: list[ChipWrite] = []
    unsupported: dict[str, int] = {}
    sample = 0
    pos = data_offset(raw)
    n = len(raw)
    saw_dac = False
    data_blocks: list[DataBlock] = []
    loop_offset = 0x1C + loop_rel if loop_rel else 0
    loop_sample_from_offset: int | None = None

    def note_unsupported(name: str, count: int = 1) -> None:
        unsupported[name] = unsupported.get(name, 0) + count

    def add(chip: str, cid: int, reg: int, val: int) -> None:
        writes.append(ChipWrite(sample, chip, cid, reg, val))

    while pos < n:
        if loop_offset and pos == loop_offset:
            loop_sample_from_offset = sample
        cmd = raw[pos]
        if cmd == 0x66:  # end of song
            break
        if 0x70 <= cmd <= 0x7F:
            sample += (cmd & 0x0F) + 1
            pos += 1
            continue
        if cmd == 0x61:
            if pos + 3 > n:
                break
            sample += _u16(raw, pos + 1)
            pos += 3
            continue
        if cmd == 0x62:
            sample += 735
            pos += 1
            continue
        if cmd == 0x63:
            sample += 882
            pos += 1
            continue

        if cmd == 0x67:  # data block: 0x66, type, 32-bit size, payload
            if pos + 7 > n:
                break
            block_type = raw[pos + 2]
            size = _u32(raw, pos + 3)
            if size > n - pos - 7:
                size = max(0, n - pos - 7)
            payload = raw[pos + 7 : pos + 7 + size]
            if block_type == 0xC3 and len(payload) >= 2:
                data_blocks.append(
                    DataBlock(
                        0xC3,
                        int.from_bytes(payload[:2], "little"),
                        bytes(payload[2:]),
                        sample,
                    )
                )
            pos += 7 + size
            continue
        if cmd == 0x68:  # PCM RAM write
            pos += 12
            continue
        if 0x80 <= cmd <= 0x8F:  # YM2612 DAC data
            # 0x8n writes one PCM byte from DAC stream 0, then waits n
            # samples.  The sample advance matters even though the PCM
            # output itself is out of scope.
            saw_dac = True
            sample += cmd & 0x0F
            pos += 1
            continue
        if cmd == 0x90:
            # DAC stream setup: remember that the YM2612 DAC is in use; the
            # PCM data itself is out of scope.
            if pos + 5 <= n and (raw[pos + 2] & 0x7F) == 0x02:
                saw_dac = True
            pos += 5
            continue
        if cmd in (0x91, 0x95, 0x94):
            pos += cmd_size(cmd)
            continue
        if cmd == 0x92:
            pos += 6
            continue
        if cmd == 0x93:
            pos += 11
            continue

        if cmd == 0xB3:  # Game Boy DMG APU register write
            if pos + 3 <= n:
                rr = raw[pos + 1]
                add("gb", (rr >> 7) & 1, rr & 0x7F, raw[pos + 2])
            pos += 3
            continue
        if cmd == 0x42:  # K005289 Bubble System WSG register write
            if pos + 3 <= n:
                off = (raw[pos + 1] >> 4) & 0x0F
                val = ((raw[pos + 1] & 0x0F) << 8) | raw[pos + 2]
                add("k005289", 0, off, val)
            pos += 3
            continue
        if cmd == 0xB9:  # HuC6280 PSG register write
            if pos + 3 <= n:
                rr = raw[pos + 1]
                add("huc6280", (rr >> 7) & 1, rr & 0x7F, raw[pos + 2])
            pos += 3
            continue
        if cmd == 0xC7:  # Virtual Boy VSU register word write
            if pos + 4 <= n:
                reg = int.from_bytes(raw[pos + 1 : pos + 3], "big")
                add(
                    "vsu",
                    1 if reg & 0x8000 else 0,
                    reg & 0x7FFF,
                    raw[pos + 3],
                )
            pos += 4
            continue
        if cmd == 0xD2:  # Konami SCC register write
            if pos + 4 <= n:
                port = raw[pos + 1]
                add(
                    "k051649",
                    (port >> 7) & 1,
                    ((port & 0x7F) << 8) | raw[pos + 2],
                    raw[pos + 3],
                )
            pos += 4
            continue

        if cmd == 0x30:
            if pos + 2 <= n:
                add("sn76489", 1, 0, raw[pos + 1])
            pos += 2
            continue
        if cmd == 0x3F:
            if pos + 2 <= n:
                add("sn76489", 1, 1, raw[pos + 1])
            pos += 2
            continue
        if cmd == 0x4F:
            if pos + 2 <= n:
                add("sn76489", 0, 1, raw[pos + 1])
            pos += 2
            continue
        if cmd == 0x50:
            if pos + 2 <= n:
                add("sn76489", 0, 0, raw[pos + 1])
            pos += 2
            continue
        if cmd in (0x51, 0xA1):
            if pos + 3 <= n:
                add("ym2413", 1 if cmd == 0xA1 else 0, raw[pos + 1], raw[pos + 2])
            pos += 3
            continue

        if cmd in (0x52, 0x53, 0xA2, 0xA3):
            if pos + 3 > n:
                break
            reg = raw[pos + 1]
            val = raw[pos + 2]
            cid = 1 if cmd in (0xA2, 0xA3) else 0
            bank = 0x100 if cmd in (0x53, 0xA3) else 0
            add("ym2612", cid, bank | reg, val)
            if reg == 0x2B and (val & 0x80):
                saw_dac = True
            pos += 3
            continue

        if cmd in (0x54, 0xA4):
            if pos + 3 > n:
                break
            add("ym2151", 1 if cmd == 0xA4 else 0, raw[pos + 1], raw[pos + 2])
            pos += 3
            continue

        if cmd in (0x55, 0xA5):
            if pos + 3 > n:
                break
            reg, val = raw[pos + 1], raw[pos + 2]
            cid = 1 if cmd == 0xA5 else 0
            if reg < 0x10:
                add("ym2203_ssg", cid, reg, val)
            else:
                add("ym2203", cid, reg, val)
            pos += 3
            continue

        if cmd in (0x56, 0x57, 0xA6, 0xA7):
            if pos + 3 > n:
                break
            reg, val = raw[pos + 1], raw[pos + 2]
            cid = 1 if cmd in (0xA6, 0xA7) else 0
            port1 = cmd in (0x57, 0xA7)
            if not port1 and reg < 0x10:
                add("ym2608_ssg", cid, reg, val)
            else:
                add("ym2608", cid, (0x100 if port1 else 0) | reg, val)
            pos += 3
            continue

        if cmd in (0x58, 0x59, 0xA8, 0xA9):
            if pos + 3 > n:
                break
            reg, val = raw[pos + 1], raw[pos + 2]
            cid = 1 if cmd in (0xA8, 0xA9) else 0
            port1 = cmd in (0x59, 0xA9)
            if not port1 and reg < 0x10:
                add("ym2610_ssg", cid, reg, val)
            else:
                add("ym2610", cid, (0x100 if port1 else 0) | reg, val)
            pos += 3
            continue

        if cmd in (0x5A, 0xAA):
            if pos + 3 > n:
                break
            add("ym3812", 1 if cmd == 0xAA else 0, raw[pos + 1], raw[pos + 2])
            pos += 3
            continue
        if cmd in (0x5B, 0xAB):
            if pos + 3 > n:
                break
            add("ym3526", 1 if cmd == 0xAB else 0, raw[pos + 1], raw[pos + 2])
            pos += 3
            continue
        if cmd in (0x5E, 0x5F, 0xAE, 0xAF):
            if pos + 3 > n:
                break
            cid = 1 if cmd in (0xAE, 0xAF) else 0
            port = 1 if cmd in (0x5F, 0xAF) else 0
            add("ymf262", cid, (port << 8) | raw[pos + 1], raw[pos + 2])
            pos += 3
            continue

        if cmd in (0x5C, 0xAC):
            # Y8950: the FM half is OPL1-shaped, the ADPCM half is out of
            # scope.  Keep FM registers for a clear diagnostic.
            if pos + 3 <= n:
                add("y8950", 1 if cmd == 0xAC else 0, raw[pos + 1], raw[pos + 2])
            pos += 3
            continue
        if cmd in (0x5D, 0xAD):
            note_unsupported("ymz280b")
            pos += 3
            continue

        if cmd == 0xA0:
            if pos + 3 > n:
                break
            rr = raw[pos + 1]
            cid = (rr >> 7) & 1
            add("ay8910", cid, rr & 0x7F, raw[pos + 2])
            pos += 3
            continue

        if cmd == 0xB3:
            note_unsupported("gameboy")
            pos += 3
            continue
        if cmd == 0xB4:
            if pos + 3 > n:
                break
            aa, val = raw[pos + 1], raw[pos + 2]
            if aa >= 0x4000:
                aa -= 0x4000
            cid = (aa >> 7) & 1
            reg = aa & 0x7F
            if reg >= 0x10:
                add("nes_dmc", cid, reg, val)
            else:
                add("nes", cid, reg, val)
            pos += 3
            continue

        # Remaining commands: sample-chip writes and seek commands.  Their
        # sizes are fixed and skipping them keeps the timeline intact.
        size = cmd_size(cmd)
        if size <= 1:
            size = 2
        pos += size

    if total_samples == 0:
        total_samples = sample
    # 0x20 is the *length* of the loop in samples, not its start.  The log is
    # written through the end of the loop, so the start is total - length.
    # Fall back to the sample position at the loop-offset command only when
    # the header carries no usable loop length.
    loop_sample: int | None = None
    if loop_rel:
        if 0 < loop_samples < total_samples:
            loop_sample = total_samples - loop_samples
        else:
            loop_sample = loop_sample_from_offset
    if loop_sample is not None and not 0 <= loop_sample < total_samples:
        loop_sample = None

    if saw_dac:
        warnings = ["YM2612 DAC/PCM stream present; PCM output is ignored"]
    else:
        warnings = []
    if unsupported:
        parts = ", ".join(f"{k} x{v}" for k, v in sorted(unsupported.items()))
        warnings.append(f"unsupported chips ignored: {parts}")
    return VgmFile(
        path=path,
        version=version,
        total_samples=total_samples,
        loop_samples=loop_samples,
        loop_sample=loop_sample,
        gd3=gd3,
        writes=writes,
        clocks=clocks,
        chip_flags=flags,
        warnings=warnings,
        rate=rate,
        header_clocks={k: v for (k, i), v in clocks.items() if i == 0},
        unsupported_writes=unsupported,
        data_blocks=data_blocks,
    )
