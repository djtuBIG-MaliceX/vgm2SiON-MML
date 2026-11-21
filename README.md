# vgm2sionmml

Convert chiptune `.vgm` / `.vgz` logs into playable **SiON MML** text.

The converter reads the chip register stream, reconstructs notes, patches,
pitch, volume and stereo, then writes SiON's MML dialect with real FM voice
definitions (`#OPM@`, `#OPN@`, `#OPL@`) and PSG/NES tracks (`%0`, `%1`).
PCM/ADPCM sample playback is intentionally out of scope.

## Supported chips

| Family | Chips | SiON output |
| --- | --- | --- |
| FM / OPM | YM2151 | `#OPM@` four-operator voices, 8 tracks |
| FM / OPN | YM2203, YM2612, YM2608, YM2610/YM2610B | `#OPN@` four-operator voices, 3/4/6 tracks |
| SSG | AY-compatible SSG inside YM2203/2608/2610 | `%0` square/noise tracks |
| FM / OPL | YM3526 (OPL1), YM3812 (OPL2), YMF262 (OPL3) | `#OPL@` 2-op and 4-op voices |
| PSG | AY-3-8910 / YM2149 | `%0` square/envelope and noise tracks |
| PSG | SN76489, Game Gear stereo; T6W28 halves as two mono chips | `%0` tone/noise tracks |
| APU | NES 2A03 pulse 1/2, triangle, noise | `%1` tracks |

Recognized but not converted: YM2413/OPLL, Y8950 ADPCM, YMF278B/OPL4,
YMF271, YMZ280B, and every sample/ADPCM chip. The report and warnings name
what was ignored.

## Install / run

Python 3.10+ and no runtime dependencies.

```bash
python -m pip install -e .
python -m vgm2sionmml convert song.vgm -o song.sionmml
python -m vgm2sionmml report song.vgz
python -m vgm2sionmml convert "VGM folder" -o mml/
```

Options:

```text
--bpm N          SiON tempo, 1-511 (default 120)
--no-loop        do not emit the `$` loop point
--title TEXT     override #TITLE
--max-voices N   SiON FM voice slot cap (default 256)
--gain F         master gain multiplier (default 0.75; 1.0 keeps source level)
--quiet          only errors
```

Example output:

```mml
#TITLE{Scene 1: Flower Field};

#OPM@0{
4, 6;
31, 10, 0, 0, 2, 30, 1, 0, 1, 0, 1;
31, 24, 0, 8, 0, 0, 1, 1, 5, 0, 1;
31, 15, 0, 0, 3, 19, 1, 0, 0, 0, 1;
31, 10, 0, 8, 1, 0, 0, 1, 1, 0, 1;
};

// YM2151 1 ch 1
%6 t120 q8 v16 @0 o5 k1 a2 ...;
```

## How it works

1. `vgm.py` walks the VGM/VGZ command stream and produces a flat,
   time-stamped `ChipWrite` list (plus header clocks). Header 0x20 is the
   loop *length*, not its start, so the loop starts at
   `total_samples - loop_length` (with the 0x1C offset as a fallback).
   YM2612 PCM commands `0x8n` advance the sample clock by `n` even though
   their PCM output is out of scope.
2. `chips/` contains one register-state decoder per family. Each decoder
   reconstructs key-on/key-off, pitch, patch, TL/gain, pan and LFO state and
   emits a monophonic `ChannelTimeline`.
3. `model.py` keeps notes with control points for pitch, gain and pan.
4. `mml.py` quantizes event times to SiON's 1920-tick whole note, splits held
   notes at control points into slurred (`&`) segments, encodes note lengths
   as exact sums of SiON note values, deduplicates FM voices and writes the
   MML.
5. The loop point becomes a `$` in every track. Notes crossing it are split
   into an intro copy and a loop-body copy, and every loop body is padded to
   the same end time.

### Voice mapping

* **YM2151**: register pages are M1, M2, C1, C2; they are reordered to
  SiON's datasheet operator order M1, C1, M2, C2. The `#OPM@` voice is
  `ALG, FB, (AR, DR, SR, RR, SL, TL, KS, MUL, DT1, DT2, AMS) x4`.
* **OPN**: OPN's register slots are S1, S3, S2, S4; they are reordered to the
  S1..S4 order SiON's generic FM engine uses. TL/AR/DR/D2R/SL/RR/KS/MUL/DT1
  carry over; SSG-EG is not representable and is reported. FM pitch follows
  libvgm/VGMPlay, including the YM2203's /72 prescaler (the YM2612/YM2608/
  YM2610 use /144). SSG clocks are master/2 on the YM2203 and master/4 on the
  YM2608/YM2610.
* **OPL**: OPL1/OPL2/OPL3 two-operator voices map to two-operator `#OPL@`
  voices. OPL3's per-pair 0x104 enable bits drive 4-op voices in physical
  chain order `[head op1, head op2, second op1, second op2]`. OPL rhythm mode
  becomes five one-shot drum tracks using the chip's rhythm operators.
* Carrier-level floors are moved out of FM voices into `x` expression values;
  this keeps voice identities stable and matches the source's per-note level.
* **PSG**: tone channels use `%0@0`; noise uses `%0@1`/`%0@2`. AY envelope
  levels are simulated at the envelope step rate and written as `x`
  expression points. Game Gear stereo becomes `p` values.
* **NES**: pulse duty becomes `%1@1/@2/@4/@6`, triangle uses `%3@16` (MA3
  triangle wave, full pitch range) with low notes folded up by an octave, and
  noise uses `%1@9/@10`; the hardware length counter and envelope are used
  for note ends and volume points. DPCM is ignored.

Pitches are computed from each chip's real clock and quantized to SiON's
1/64-semitone `k` resolution. Notes above B9 are clamped because SiON's
octave command stops at 9.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite covers the VGM header/loop parser, exact duration arithmetic, each
chip's A4 pitch mapping, OPN stereo, OPM/OPN/OPL operator ordering, OPL 4-op
and rhythm voices, OPL TL automation, NES triangle/MA3 pitch, SN Game Gear
stereo, and end-to-end compilation with the reference `libSiON-rs` player.
Sample-song tests skip when their files are absent. The sample expectations
were cross-checked against libvgm/VGMPlay renders and the vgm2151fur register
analyses.

Set `SION_PLAY` to point at a `sion-play` executable if it is not at
`D:/Projects/Rust/libSiON-rs/target/release/sion-play.exe`.

## Known limitations

* **PCM/ADPCM is not converted**: YM2612 DAC, YM2608/2610 rhythm/ADPCM,
  NES DPCM, OPL4 waves, and all sample chips are ignored (with warnings).
* **OPN channel-3 special mode** is skipped because its four operators have
  independent pitches, which the `#OPN@` voice model cannot express.
* **OPL rhythm** pitch and decay are approximations; the five drum tracks use
  the chip's rhythm operators and one-shot note lengths.
* **NES sweep** is ignored, and the envelope/length simulation is an
  approximation rather than cycle-exact APU emulation.
* **AY/SN noise** is approximated with SiON's pulse/periodic noise voices.
* Mid-note timbre changes are not re-voiced; carrier TL changes are written
  as `x` expression points, and boosts above the key-on level are clamped.
* Long vibrato/pitch automation is written as stepped, slurred note segments;
  SiON LFO commands are emitted from OPM/OPN LFO registers when present.
* SiON has 256 FM voice slots and 128 tracks. Voice overflow reuses the
  nearest timbre and is warned about. The converter currently emits one track
  per hardware channel; logs with more than 128 simultaneous channels are not
  supported.
* Dense multi-chip logs can clip in the player. The CLI defaults to `--gain 0.75` for headroom; pass `--gain 1.0` to keep the source level or lower it further.

## Layout

```text
vgm2sionmml/
  vgm.py          VGM/VGZ header + command walker
  model.py        notes, control points, tracks, voice library
  pitch.py        chip frequency formulas and SiON pitch units
  mml.py          SiON MML renderer
  convert.py      orchestration and diagnostics
  cli.py          command line interface
  chips/
    opm.py        YM2151
    opn.py        YM2203/2612/2608/2610 FM
    opl.py        YM3526/YM3812/YMF262
    psg.py        AY-3-8910/YM2149 + SN76489
    nes.py        NES 2A03
tests/            synthetic VGM fixtures + sample-song checks
```

## References

* SiON MML syntax: <https://keim.github.io/SiON/mmlref/sion_mml_reference_e.html>
* libSiON-rs: `D:/Projects/Rust/libSiON-rs`
* libvgm VGM format: `D:/Projects/Forks/libvgm`
* vgm2151fur register/pitch reference: `D:/Projects/Forks/vgm2151fur`
* Example songs: `D:/Projects/Forks/SiONMML`
