# vgm 2 SiON MML

Convert chiptune `.vgm` / `.vgz` logs into playable **SiON MML** text.

The converter reads the chip register stream, reconstructs notes, patches,
pitch, volume, stereo and wavetables, then writes SiON's MML dialect with
real FM voice definitions (`#OPM@`, `#OPN@`, `#OPL@`), custom `#WAVB`
tables, and PSG/NES/Game Boy tracks (`%0`, `%1`, `%2`, `%4`, `%8`).
PCM/ADPCM sample playback is intentionally out of scope.

## Supported chips

| Family | Chips | SiON output |
| --- | --- | --- |
| FM / OPM | YM2151 | `#OPM@` four-operator voices, 8 tracks |
| FM / OPN | YM2203, YM2612, YM2608, YM2610/YM2610B | `#OPN@` four-operator voices, 3/4/6 tracks |
| SSG | AY-compatible SSG inside YM2203/2608/2610 | `%0` square/noise tracks |
| FM / OPL | YM3526 (OPL1), YM3812 (OPL2), YMF262 (OPL3) | `#OPL@` 2-op and 4-op voices |
| FM / OPLL | YM2413 (OPLL), Konami VRC7 | `#OPL@` 2-op voices, 9 or 6 channels |
| PSG | AY-3-8910 / YM2149 | `%0` square/envelope and noise tracks |
| PSG | SN76489, Game Gear stereo; T6W28 halves as two mono chips | `%0` tone/noise tracks |
| APU | NES 2A03 pulse 1/2, triangle, noise | `%1` tracks |
| Wavetable | Konami SCC / SCC+ (K051649/K052539) | `%4` + `#WAVB` tables |
| Wavetable | Konami K005289 Bubble System WSG | `%4` + `#WAVB` tables |
| Wavetable | Hudson HuC6280 (PC Engine) | `%4` + `#WAVB`, `%2` noise |
| Wavetable | Nintendo Virtual Boy VSU | `%4` + `#WAVB`, `%2` noise |
| APU | Game Boy DMG pulse 1/2, wave, noise | `%8`, `%4`, `%2` tracks |

Recognized but not converted: Y8950 ADPCM, YMF278B/OPL4, YMF271,
YMZ280B, and every sample/ADPCM chip. HuC6280, VSU and Game Boy noise use
SiON's periodic-noise module; the noise pitch is approximate. The report
and warnings name what was ignored.

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
--bpm N            explicit SiON tempo, 1-511 (default: auto-detect)
--no-auto-tempo     use 120 BPM instead of auto tempo detection
--no-loop           do not emit the `$` loop point
--no-optimize      disable macro/loop/default-length and table-envelope optimization
--title TEXT        override #TITLE
--max-voices N      SiON FM voice slot cap (default 256)
--gain F            master gain multiplier (default 0.75; 1.0 keeps the converted mix level)
--quiet             only errors
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
   A dual-chip main-header clock is copied to chip instance 1, so a 4 MHz
   dual YM2151 does not fall back to the 3.579545 MHz default on its second
   chip. YM2612 PCM commands `0x8n` advance the sample clock by `n` even
   though their PCM output is out of scope. For ROM/RAM data blocks, bit 31
   of the size field selects the chip instance and is masked before the
   payload is skipped; otherwise an OKIM6295 ROM block can look like a 2 GiB
   block and hide the YM2151 commands that follow it.
2. `chips/` contains one register-state decoder per family. Each decoder
   reconstructs key-on/key-off, pitch, patch, TL/gain, pan and LFO state,
   handles one selected chip instance at a time, and emits a monophonic
   `ChannelTimeline`.
3. `model.py` keeps notes with control points for pitch, gain, pan, waveform
   changes and OPL operator levels.
4. `mml.py` quantizes event times to SiON's 1920-tick whole note, splits held
   notes at control points into slurred (`&`) segments, encodes note lengths
   as exact sums of SiON note values, deduplicates FM voices and writes the
   MML.  `envelopes.py` can replace a repeated or long pitch/gain control
   curve with a `#TABLE` and `nt`/`na`; fractional `np` tables are not emitted
   because their frame cursor did not reproduce the source curves exactly.
5. The first pass renders token streams.  `optimize.py` then chooses a tempo
   grid, applies a common default note length, finds immediately repeated
   token blocks for `[ ... ]n` loops, and emits up to 26 shared `#A`..`#Z`
   macros.  A macro body must have at least four commands and at least three
   note commands, so a repeated single note is not turned into a macro.  The
   second pass writes the compact MML.  SiON tempo is track-local, so every
   track starts with its own `t<bpm>`; otherwise a long rest on one channel
   would drift against the others.
6. The loop point becomes a `$` in every track. Notes crossing it are split
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
  For percussive operators (ET=0), the extractor also computes the
  attack/decay/release frames from the OPL envelope-rate tables and caps the
  note at the point where the carrier envelope can reach silence; players
  that do not auto-release the S phase therefore do not hold the note too
  long.
* **OPLL/VRC7**: a YM2413 clock word with bit 31 set selects Konami's VRC7
  preset bank and six channels; otherwise the chip uses the standard 19-entry
  OPLL bank and nine channels. Preset 0 reads the custom `$00`-`$07` patch.
  The `$3x` low nibble becomes note gain (3 dB per step). Rhythm mode becomes
  five one-shot tracks (BD, SD, TOM, TOP, HH) built from preset operators
  16-18 and the `$36`-`$38` level nibbles.
* Carrier-level floors are moved out of FM voices into `x` expression values;
  this keeps voice identities stable and matches the source's per-note level.
  OPN carrier levels are brought to the same SiON family reference as
  OPM/OPL with a fixed `3x` trim.  This is not a per-song or chip-count
  normalization; it compensates Genesis patch levels that sit below OPM
  levels and avoids inheriting VGMPlay's whole-log volume normalization.
* **PSG**: tone channels use `%0@0`; noise uses `%0@1`/`%0@2`. AY envelope
  levels are simulated at the envelope step rate and written as `x`
  expression points. Volume 0 and SN attenuation 15 close the current note
  instead of leaving a silent held note. AY and SN76489 tracks keep their
  full per-channel level and source attenuation curves; no mix-headroom or
  chip-count attenuation is applied. Game Gear stereo becomes `p` values.
  AY and SN noise use the 32-entry PSG noise pitch range, not the old
  out-of-range note offsets.
* **NES**: pulse duty becomes `%1@1/@2/@4/@6`, triangle uses `%3@16` (MA3
  triangle wave, full pitch range) with low notes folded up by an octave, and
  noise uses `%1@9/@10`; the hardware length counter and envelope are used
  for note ends and volume points. DPCM is ignored. Dual-APU logs keep the
  two register banks separate via bit 7 of the `0xB4` register byte. Timer 0
  produces an above-B9/ultrasonic pulse; those gates are emitted as silence
  rather than clamped to an audible B9 beep.
* **HuC6280 gates**: control/balance changes inside the first 64 samples are
  folded into the note's attack level; later changes become expression or
  `na` table envelopes. Sub-frame gates below 100 samples are treated as
  driver housekeeping and dropped, preventing one-sample percussion clicks.
  Noise notes use an in-range pitch derived from `31 - (noise & 31)` and
  follow the `$07` period, so low noise settings no longer render in a
  detached high octave. WAVE/noise volume follows libvgm/Ootake's 1.0869 dB
  per-step table (indexed by control volume plus main and channel balance),
  rather than a linear `level / 31`; a channel-balance nibble of 0 is a hard
  mute. Each hardware gate-off also inserts a zero expression a short tick
  before the note end because SiON's `%4` channel applies a KS release tail
  that the HuC6280 does not have.
* **SCC, K005289, HuC6280, VSU, Game Boy wave**: 32-frame waveforms are
  normalized to signed 8-bit and deduplicated into `#WAVB` tables. `%4@n`
  selects a table; held-note waveform rewrites become `@n` changes between
  slurred segments. K005289 PROM data comes from its `0xC3` VGM data block.
* **Game Boy pulse/noise**: pulse duties map to `%8` widths; noise uses
  `%2@2` with a note range derived from NR43. NRx2 envelope steps are written
  as `x` gain points; a zero down-envelope, DAC-off write, or zero wave level
  closes the note. NR51 drives `p` per channel, including the initial centering
  write, rather than the old fixed left bias.
* **OPL operator levels**: carrier TL is normalized into `x`; non-carrier TL
  is written per note as `iN@tl`, which keeps 15k-note OPL3 logs inside the
  256-voice limit without changing the patch.  OPL multiplier table entries
  11/13/14 are translated to the equivalent SiON multipliers.

Pitches are computed from each chip's real clock and quantized to SiON's
1/64-semitone `k` resolution. Notes above B9 are clamped because SiON's
octave command stops at 9.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite covers the VGM header/loop parser, exact duration arithmetic, each
chip's A4 pitch mapping, OPN stereo, OPM/OPN/OPL operator ordering, OPL 4-op
and rhythm voices, OPLL/VRC7 and OPLL rhythm, OPL TL automation, NES triangle/MA3 pitch, SN Game Gear
stereo, SCC/K005289/HuC6280/VSU/Game Boy wavetable tracks, size optimization,
and end-to-end compilation with the reference `libSiON-rs` player.
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
* **OPL/OPLL rhythm** pitch and decay are approximations; the five drum tracks
  use the chip's rhythm operators and one-shot note lengths. OPLL rhythm
  channel registers are linked, so the drum pitch tracks the linked hardware
  channel rather than independent drum tuning.
* **NES sweep** is ignored, and the envelope/length simulation is an
  approximation rather than cycle-exact APU emulation.
* **AY/SN noise** is approximated with SiON's pulse/periodic noise voices.
* Mid-note timbre changes are not re-voiced; carrier TL changes are written
  as `x` expression points, and boosts above the key-on level are clamped.
* Long vibrato/pitch automation is written as stepped, slurred note segments;
  semitone-aligned curves may become `nt` `#TABLE` envelopes and gain curves
  may become `na`.  Arbitrary fractional pitch curves stay as ties because
  SiON's `np` frame mapping is not sample-exact.  SiON LFO commands are
  emitted from OPM/OPN LFO registers when present; OPM LFOs faster than
  SiON's one-frame minimum are dropped rather than clamped, because a
  clamped high-rate AM produced incorrect sidebands on sustained notes.
* SiON has 256 FM voice slots, 256 custom wavetable slots and 128 tracks.
  Voice/wavetable overflow reuses the nearest timbre and is warned about.
  The converter currently emits one track per hardware channel; logs with
  more than 128 simultaneous channels are not supported.
* HuC6280/VSU/Game Boy noise is approximate (SiON noise tables, not an exact
  LFSR/waveform match); HuC6280 gates shorter than 100 samples are dropped.
  K005289 needs its 0xC3 PROM data block to produce
  wavetable notes.
* Auto tempo only chooses a musical grid; pass `--bpm N` when the source has
  a known tempo.  Optimization is conservative around ties and stops before
  creating constructs SiON's parser rejects.
* Dense multi-chip logs can clip in the player. The CLI defaults to `--gain 0.75` for headroom; pass `--gain 1.0` to keep the converted mix level or lower it further.

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
    opll.py       YM2413/OPLL + VRC7, including rhythm
    opll_banks.py YM2413 and VRC7 preset data
    psg.py        AY-3-8910/YM2149 + SN76489
    nes.py        NES 2A03
    scc.py        Konami SCC / SCC+
    k005289.py    Konami K005289
    huc6280.py    Hudson HuC6280
    vsu.py        Virtual Boy VSU
    gb.py         Game Boy DMG
  envelopes.py    pitch/gain #TABLE detection and attachment
  optimize.py     two-pass tempo/grid and MML size optimizer
tests/            synthetic VGM fixtures + sample-song checks
```

## References

* [SiON](https://github.com/keim/SiON) MML syntax: <https://keim.github.io/SiON/mmlref/sion_mml_reference_e.html>
* GDSiON / libSiON-cpp/rs: [Godot](https://github.com/YuriSizov/gdsion), [C++](https://github.com/djtubig-MaliceX/libSiON-cpp), [Rust](https://github.com/djtubig-MaliceX/libSiON-rs)
* libvgm: <https://github.com/valleybell/libvgm>
* vgm2151fur: <https://github.com/me-when-the-uh/vgm2151fur>
