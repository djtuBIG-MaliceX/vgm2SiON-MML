"""Command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .convert import ConversionError, convert_file
from .vgm import load_vgm


def _iter_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files: list[Path] = []
        for suffix in ("*.vgm", "*.vgz", "*.VGM", "*.VGZ"):
            files.extend(path.glob(suffix))
        return sorted(set(files), key=lambda p: p.name.lower())
    return []


def _report(path: Path) -> int:
    vgm = load_vgm(path)
    print(f"{path}")
    print(f"  version 0x{vgm.version:X}")
    print(f"  duration {vgm.duration_seconds:.2f}s, loop {vgm.loop_sample}")
    if vgm.gd3:
        for key in ("title", "game", "system", "author"):
            if vgm.gd3.get(key):
                print(f"  {key}: {vgm.gd3[key]}")
    print("  chips:")
    for chip in vgm.chips_used():
        clocks = sorted(
            {v for (name, _i), v in vgm.clocks.items() if name == chip}
        )
        print(f"    {chip:12s} writes={vgm.write_count(chip):7d} clocks={clocks}")
    for warning in vgm.warnings:
        print(f"  warning: {warning}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vgm2sionmml",
        description="Convert VGM/VGZ chiptune logs to SiON MML text.",
    )
    sub = p.add_subparsers(dest="command")
    c = sub.add_parser("convert", help="convert a file or folder")
    c.add_argument("input", type=Path)
    c.add_argument("-o", "--output", type=Path, help="output .sionmml file or folder")
    c.add_argument("--bpm", type=int, default=120, help="SiON tempo (default 120)")
    c.add_argument("--no-loop", action="store_true", help="do not emit the $ loop point")
    c.add_argument("--title", help="override #TITLE")
    c.add_argument("--max-voices", type=int, default=256)
    c.add_argument("--gain", type=float, default=0.75, help="master gain multiplier (default 0.75)")
    c.add_argument("--quiet", action="store_true")
    r = sub.add_parser("report", help="inspect a VGM without converting")
    r.add_argument("input", type=Path)
    return p


def _convert_one(args, src: Path, dst: Path | None) -> int:
    try:
        result = convert_file(
            src,
            bpm=args.bpm,
            loop=not args.no_loop,
            title=args.title,
            max_voices=args.max_voices,
            gain=args.gain,
        )
    except (ConversionError, ValueError, OSError) as exc:
        print(f"ERROR {src}: {exc}", file=sys.stderr)
        return 1
    if dst is None:
        dst = src.with_suffix(".sionmml")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(result.text, encoding="utf-8")
    if not args.quiet:
        print(
            f"{src} -> {dst}  "
            f"({result.tracks} tracks, {result.notes} notes, {result.voices} voices)"
        )
        for warning in result.warnings:
            print(f"  warning: {warning}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    if not args.input.exists():
        print(f"ERROR: no such file or directory: {args.input}", file=sys.stderr)
        return 2
    if args.command == "report":
        if args.input.is_dir():
            for path in _iter_inputs(args.input):
                _report(path)
            return 0
        return _report(args.input)

    inputs = _iter_inputs(args.input)
    if not inputs:
        print(f"ERROR: no .vgm/.vgz files under {args.input}", file=sys.stderr)
        return 2
    if args.output is None:
        outputs = [None] * len(inputs)
    elif args.input.is_file() and args.output.suffix.lower() in (".sionmml", ".mml", ".txt"):
        outputs = [args.output]
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        outputs = [args.output / (p.stem + ".sionmml") for p in inputs]
    failures = 0
    for src, dst in zip(inputs, outputs):
        failures += _convert_one(args, src, dst)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
