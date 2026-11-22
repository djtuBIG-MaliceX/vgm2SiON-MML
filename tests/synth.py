"""Synthetic VGM builders used by the unit tests."""

from __future__ import annotations

import os
import struct
import subprocess
import wave
from pathlib import Path

SION_PLAY = os.environ.get("SION_PLAY") or (
    r"D:/Projects/Rust/libSiON-rs/target/release/sion-play.exe"
)

# Standard sample-chip commands.
def wait(n: int) -> bytes:
    return bytes([0x61, n & 0xFF, (n >> 8) & 0xFF])


def opm(reg: int, val: int) -> bytes:
    return bytes([0x54, reg & 0xFF, val & 0xFF])


def ym2612(port: int, reg: int, val: int) -> bytes:
    return bytes([0x52 if port == 0 else 0x53, reg & 0xFF, val & 0xFF])


def ym2203(reg: int, val: int) -> bytes:
    return bytes([0x55, reg & 0xFF, val & 0xFF])


def ym2608(port: int, reg: int, val: int) -> bytes:
    return bytes([0x56 if port == 0 else 0x57, reg & 0xFF, val & 0xFF])


def ym2610(port: int, reg: int, val: int) -> bytes:
    return bytes([0x58 if port == 0 else 0x59, reg & 0xFF, val & 0xFF])


def ym3812(reg: int, val: int) -> bytes:
    return bytes([0x5A, reg & 0xFF, val & 0xFF])


def ym3526(reg: int, val: int) -> bytes:
    return bytes([0x5B, reg & 0xFF, val & 0xFF])


def ymf262(port: int, reg: int, val: int) -> bytes:
    return bytes([0x5E if port == 0 else 0x5F, reg & 0xFF, val & 0xFF])


def ay8910(reg: int, val: int) -> bytes:
    return bytes([0xA0, reg & 0xFF, val & 0xFF])


def scc(bank: int, reg: int, val: int) -> bytes:
    return bytes([0xD2, bank & 0x7F, reg & 0xFF, val & 0xFF])


def huc(reg: int, val: int) -> bytes:
    return bytes([0xB9, reg & 0xFF, val & 0xFF])


def vsu(reg: int, val: int) -> bytes:
    return bytes([0xC7, (reg >> 8) & 0xFF, reg & 0xFF, val & 0xFF])


def gb_apu(reg: int, val: int) -> bytes:
    return bytes([0xB3, reg & 0xFF, val & 0xFF])


def k005289(off: int, val: int) -> bytes:
    return bytes([
        0x42,
        ((off & 0x0F) << 4) | ((val >> 8) & 0x0F),
        val & 0xFF,
    ])


def data_block(block_type: int, start: int, payload: bytes) -> bytes:
    size = len(payload) + 2
    return (
        bytes([0x67, 0x66, block_type & 0xFF])
        + size.to_bytes(4, "little")
        + (start & 0xFFFF).to_bytes(2, "little")
        + payload
    )


def sn76489(value: int) -> bytes:
    return bytes([0x50, value & 0xFF])


def gg_stereo(mask: int) -> bytes:
    return bytes([0x4F, mask & 0xFF])


def nes(reg: int, val: int) -> bytes:
    return bytes([0xB4, reg & 0xFF, val & 0xFF])


def make_vgm(
    commands: bytes,
    total_samples: int,
    *,
    clocks: dict[int, int] | None = None,
    loop_sample: int | None = None,
    loop_offset: int = 0,
    title: str | None = None,
    rate: int = 60,
) -> bytes:
    """Build a minimal VGM 1.50 file with a 0x40-byte header."""
    header = bytearray(0x100)
    header[0:4] = b"Vgm "
    struct.pack_into("<I", header, 8, 0x150)
    struct.pack_into("<I", header, 0x18, total_samples)
    struct.pack_into("<I", header, 0x1C, loop_offset)
    struct.pack_into("<I", header, 0x20, loop_sample or 0)
    struct.pack_into("<H", header, 0x24, rate)
    struct.pack_into("<I", header, 0x34, 0x100 - 0x34)
    for offset, value in (clocks or {}).items():
        struct.pack_into("<I", header, offset, value & 0x3FFFFFFF)
    body = bytearray(commands)
    if title:
        body.extend(b"\x00")
    body.append(0x66)
    return bytes(header) + bytes(body)


def sion_play_available() -> bool:
    return Path(SION_PLAY).is_file()


def compile_mml(text: str, workdir: Path, name: str = "song") -> Path:
    """Compile an MML string with sion-play and return the WAV path."""
    if not sion_play_available():
        raise RuntimeError("sion-play is not installed")
    mml = workdir / f"{name}.sionmml"
    wav = workdir / f"{name}.wav"
    mml.write_text(text, encoding="utf-8")
    proc = subprocess.run(
        [SION_PLAY, "-f", str(mml), "-o", str(wav), "-t", "5"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"sion-play failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )
    if not wav.is_file() or wav.stat().st_size <= 44:
        raise AssertionError("sion-play produced no audio")
    return wav


def wav_has_signal(path: Path) -> bool:
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
        return any(byte != 0 for byte in frames)
