from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vgm2sionmml.convert import ConversionError, convert_vgm
from vgm2sionmml.mml import duration_token, render_song, ticks_from_samples
from vgm2sionmml.model import FmVoice, Note, Song, Track, VoiceLibrary
from vgm2sionmml.vgm import ChipWrite, VgmFile, load_vgm
from tests import synth


class VgmParserTests(unittest.TestCase):
    def test_header_writes_and_loop(self):
        commands = synth.opm(0x08, 0x78) + synth.wait(735) + synth.opm(0x08, 0)
        data = synth.make_vgm(
            commands,
            44100,
            clocks={0x30: 3579545},
            loop_sample=44100 - 735,
            loop_offset=0x103,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.vgm"
            path.write_bytes(data)
            vgm = load_vgm(path)
        self.assertEqual(vgm.version, 0x150)
        self.assertEqual(vgm.clock("ym2151"), 3579545)
        self.assertEqual(len(vgm.writes), 2)
        self.assertEqual([w.sample for w in vgm.writes], [0, 735])
        # 0x20 is the loop *length*; the loop starts at the command addressed
        # by 0x1C, which is the second write at sample 735.
        self.assertEqual(vgm.loop_sample, 735)
        self.assertEqual(vgm.total_samples, 44100)


    def test_ym2612_pcm_commands_advance_sample_time(self):
        commands = bytes((0x80, 0x81, 0x8F)) + synth.opm(0x08, 0x78)
        data = synth.make_vgm(commands, 16, clocks={0x30: 3579545})
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.vgm"
            path.write_bytes(data)
            vgm = load_vgm(path)
        # 0x80 waits 0, 0x81 waits 1, 0x8F waits 15 samples.
        self.assertEqual(vgm.writes[-1].sample, 16)
        self.assertIn("DAC/PCM", " ".join(vgm.warnings))

    def test_rom_data_block_size_high_bit_is_chip_id(self):
        # OKIM6295 ROM blocks (type 0x8B) use bit 31 of the size field as
        # the chip-instance flag.  It must not be mistaken for a 2 GiB block.
        commands = (
            bytes([0x67, 0x66, 0x8B])
            + (0x80000008).to_bytes(4, "little")
            + b"12345678"
            + synth.opm(0x08, 0x78)
            + synth.wait(100)
        )
        data = synth.make_vgm(commands, 100, clocks={0x30: 4_000_000})
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.vgm"
            path.write_bytes(data)
            vgm = load_vgm(path)
        self.assertEqual(len(vgm.writes), 1)
        self.assertEqual(vgm.writes[0].chip, "ym2151")
        self.assertEqual((vgm.writes[0].reg, vgm.writes[0].val), (0x08, 0x78))

    def test_vgz_roundtrip(self):
        import gzip

        commands = synth.opm(0x08, 0x78) + synth.wait(44100)
        data = synth.make_vgm(commands, 44100, clocks={0x30: 4000000})
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.vgz"
            path.write_bytes(gzip.compress(data))
            vgm = load_vgm(path)
        self.assertEqual(vgm.clock("ym2151"), 4000000)
        self.assertEqual(len(vgm.writes), 1)

    def test_dual_opm_header_clock_applies_to_both_instances(self):
        data = synth.make_vgm(
            b"",
            44100,
            raw_clocks={0x30: 0x40000000 | 4_000_000},
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.vgm"
            path.write_bytes(data)
            vgm = load_vgm(path)
        self.assertEqual(vgm.clock("ym2151", 0), 4_000_000)
        self.assertEqual(vgm.clock("ym2151", 1), 4_000_000)


class DurationTests(unittest.TestCase):
    def test_note_values(self):
        self.assertEqual(duration_token(1920), "1")
        self.assertEqual(duration_token(960), "2")
        self.assertEqual(duration_token(480), "4")
        self.assertEqual(duration_token(120), "16")

    def test_nonstandard_duration_sums_exactly(self):
        for ticks in (1, 7, 11, 153, 1919, 1921, 5000):
            token = duration_token(ticks)
            total = 0
            for piece in token.split("^"):
                total += 1920 // int(piece)
            self.assertEqual(total, ticks, f"{ticks}: {token}")

    def test_ticks_from_samples(self):
        self.assertEqual(ticks_from_samples(735, 120), 16)
        self.assertEqual(ticks_from_samples(44100, 120), 960)


class MmlTests(unittest.TestCase):
    def test_voice_headers_precede_tracks(self):
        voices = VoiceLibrary()
        voice = FmVoice("opm", 7, 0, ((1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),) * 4)
        tl = Track("t", 6, "fm", _timeline(voice), order=0)
        song = Song("T", [tl], voices, 44100)
        text = render_song(song)
        self.assertIn("#OPM@0{", text)
        self.assertLess(text.index("#OPM@0{"), text.index("%6"))

    def test_loop_body_pads_each_track_to_song_end(self):
        voices = VoiceLibrary()
        voice = FmVoice("opm", 7, 0, ((1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),) * 4)
        tracks = [
            Track("short", 6, "fm", _timeline(voice), order=0),
            Track("long", 6, "fm", _timeline(voice), order=1),
        ]
        tracks[0].timeline.notes[0].end = 22050
        tracks[1].timeline.notes[0].end = 44100
        song = Song("T", tracks, voices, 44100, loop_sample=0, bpm=120)
        text = render_song(song)
        track_lines = [line for line in text.splitlines() if line.startswith("%")]
        self.assertEqual(len(track_lines), 2)
        self.assertTrue(all("$" in line for line in track_lines))
        # The short track must be padded to the same loop-body end.
        self.assertIn("r", track_lines[0].split("$", 1)[1])
        self.assertNotIn("r", track_lines[1].split("$", 1)[1])

    def test_tempo_is_repeated_on_every_track(self):
        voices = VoiceLibrary()
        voice = FmVoice("opm", 7, 0, ((1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),) * 4)
        tracks = [
            Track("a", 6, "fm", _timeline(voice), order=0),
            Track("b", 6, "fm", _timeline(voice), order=1),
        ]
        song = Song("T", tracks, voices, 44100, bpm=102)
        text = render_song(song)
        track_lines = [line for line in text.splitlines() if line.startswith("%")]
        self.assertEqual(len(track_lines), 2)
        self.assertTrue(all("t102" in line for line in track_lines), track_lines)

    def test_loop_marker_and_valid_body(self):
        voices = VoiceLibrary()
        voice = FmVoice("opm", 7, 0, ((1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),) * 4)
        tl = Track("t", 6, "fm", _timeline(voice), order=0)
        song = Song("T", [tl], voices, 88200, loop_sample=44100)
        text = render_song(song)
        self.assertIn("$", text)
        self.assertIn("%6", text)


class ExtractorTests(unittest.TestCase):
    def _tmp_vgm(self, data: bytes) -> Path:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        path = Path(td.name) / "test.vgm"
        path.write_bytes(data)
        return path

    def test_opm_a4(self):
        cmds = bytearray()
        cmds += synth.opm(0x20, 0xC7) + synth.opm(0x28, 0x4A) + synth.opm(0x30, 0)
        for op in range(4):
            cmds += (
                synth.opm(0x40 + 8 * op, 0x01)
                + synth.opm(0x60 + 8 * op, 0x00)
                + synth.opm(0x80 + 8 * op, 0x1F)
                + synth.opm(0xA0 + 8 * op, 0x00)
                + synth.opm(0xC0 + 8 * op, 0x00)
                + synth.opm(0xE0 + 8 * op, 0xF0)
            )
        cmds += synth.opm(0x08, 0x78) + synth.wait(44100) + synth.opm(0x08, 0)
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(bytes(cmds), 44100, clocks={0x30: 3579545})))
        result = convert_vgm(vgm)
        self.assertEqual(result.tracks, 1)
        self.assertEqual(result.notes, 1)
        note = _only_note(vgm, "opm")
        self.assertAlmostEqual(note.pitch / 64.0, 69.0, places=2)
        self.assertIn("#OPM@", result.text)
        self.assertIn("a2", result.text)
        self.assertIn("%6", result.text)

    def test_dual_opm_instances_are_not_duplicated(self):
        # A VGM may contain two YM2151 instances.  convert_vgm calls the
        # extractor once per instance; the extractor must honour that
        # selection instead of returning both instances on each call.
        writes = []
        for cid in (0, 1):
            for reg, val in (
                (0x20, 0xC7), (0x28, 0x4A), (0x30, 0),
                (0x40, 0x01), (0x60, 0x00), (0x80, 0x1F),
                (0xA0, 0x00), (0xC0, 0x00), (0xE0, 0xF0),
                (0x08, 0x78),
            ):
                writes.append(ChipWrite(0, "ym2151", cid, reg, val))
        vgm = VgmFile(
            path=Path("dual-opm.vgm"),
            version=0x151,
            total_samples=44100,
            loop_samples=0,
            loop_sample=None,
            writes=writes,
            clocks={("ym2151", 0): 3_579_545, ("ym2151", 1): 3_579_545},
        )
        result = convert_vgm(vgm)
        self.assertEqual(result.tracks, 2)
        self.assertEqual(result.notes, 2)

    def test_opm_fast_lfo_is_not_emitted(self):
        from vgm2sionmml.chips.opm import OpmState

        state = OpmState.new()
        state.lfrq = 0xC8
        state.amd = 10
        state.pms_ams[0] = 3
        self.assertIsNone(state.lfo(0, 4_000_000))

    def test_opn_a4(self):
        cmds = bytearray()
        cmds += synth.ym2612(0, 0x22, 0) + synth.ym2612(0, 0x27, 0) + synth.ym2612(0, 0x2B, 0)
        cmds += synth.ym2612(0, 0xB4, 0xC0)
        cmds += synth.ym2612(0, 0xA0, 0x1D) + synth.ym2612(0, 0xA4, 0x2A) + synth.ym2612(0, 0xB0, 0xC7)
        for op in range(4):
            base = 0x30 + op * 4
            cmds += (
                synth.ym2612(0, base, 1)
                + synth.ym2612(0, base + 0x10, 0)
                + synth.ym2612(0, base + 0x20, 0x1F)
                + synth.ym2612(0, base + 0x30, 0)
                + synth.ym2612(0, base + 0x40, 0)
                + synth.ym2612(0, base + 0x50, 0xF0)
            )
        cmds += synth.ym2612(0, 0x28, 0xF0) + synth.wait(44100) + synth.ym2612(0, 0x28, 0)
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(bytes(cmds), 44100, clocks={0x2C: 7680000})))
        result = convert_vgm(vgm)
        self.assertEqual(result.notes, 1)
        note = _only_note(vgm, "opn")
        self.assertAlmostEqual(note.pitch / 64.0, 69.0, delta=0.05)
        self.assertEqual(note.gain, 1.0)
        self.assertIn("#OPN@", result.text)

    def test_opl2_a4(self):
        cmds = bytearray()
        for reg, val in (
            (0x20, 1), (0x23, 0x21), (0x40, 0x38), (0x43, 0),
            (0x60, 0xF0), (0x63, 0xF0), (0x80, 0), (0x83, 0),
            (0xE0, 0), (0xE3, 0), (0xA0, 0x44), (0xC0, 0), (0xB0, 0x32),
        ):
            cmds += synth.ym3812(reg, val)
        cmds += synth.wait(44100) + synth.ym3812(0xB0, 0x12)
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(bytes(cmds), 44100, clocks={0x50: 3579545})))
        result = convert_vgm(vgm)
        self.assertEqual(result.notes, 1)
        note = _only_note(vgm, "opl")
        self.assertEqual(len(note.voice.ops), 2)
        self.assertAlmostEqual(note.pitch / 64.0, 69.0, places=2)
        self.assertIn("#OPL@", result.text)

    def test_opl3_four_op_a4(self):
        cmds = bytearray()
        cmds += synth.ymf262(1, 0x04, 1)
        regs = (
            (0x20, 1), (0x23, 1), (0x28, 0x21), (0x2B, 0x21),
            (0x40, 0x38), (0x43, 0x38), (0x48, 0x38), (0x4B, 0),
            (0x60, 0xF0), (0x63, 0xF0), (0x68, 0xF0), (0x6B, 0xF0),
            (0x80, 0), (0x83, 0), (0x88, 0), (0x8B, 0),
            (0xA0, 0x44), (0xA3, 0x44), (0xA8, 0x44), (0xAB, 0x44),
            (0xC0, 0x30), (0xC3, 0x30), (0xC8, 0x30), (0xCB, 0x30),
            (0xB0, 0x32), (0xB3, 0x32),
        )
        for reg, val in regs:
            cmds += synth.ymf262(0, reg, val)
        cmds += synth.wait(44100) + synth.ymf262(0, 0xB0, 0x12)
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(bytes(cmds), 44100, clocks={0x5C: 14318180})))
        result = convert_vgm(vgm)
        self.assertEqual(result.notes, 1)
        note = _only_note(vgm, "opl")
        self.assertEqual(len(note.voice.ops), 4)
        self.assertAlmostEqual(note.pitch / 64.0, 69.0, places=2)

    def test_opll_a4_and_factory_preset(self):
        cmds = bytearray()
        cmds += synth.ym2413(0x30, 0x10)
        cmds += synth.ym2413(0x10, 0x21)
        cmds += synth.ym2413(0x20, 0x19)
        cmds += synth.wait(44100) + synth.ym2413(0x20, 0x09)
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(bytes(cmds), 44100, clocks={0x10: 3579545})))
        result = convert_vgm(vgm)
        self.assertEqual(result.notes, 1)
        note = _only_note(vgm, "opll")
        self.assertEqual(len(note.voice.ops), 2)
        self.assertAlmostEqual(note.pitch / 64.0, 69.0, delta=0.1)
        self.assertIn("#OPL@", result.text)
        # OPLL factory preset 1 (Violin) from libvgm's default bank.
        self.assertEqual(
            note.voice.ops,
            (
                (0, 13, 0, 0, 1, 0, 30, 1, 0, 1, 0),
                (1, 7, 8, 7, 1, 1, 0, 0, 0, 1, 0),
            ),
        )

    def test_vrc7_flag_selects_bank_and_six_channels(self):
        cmds = bytearray()
        cmds += synth.ym2413(0x30, 0x10)
        cmds += synth.ym2413(0x10, 0x21)
        cmds += synth.ym2413(0x20, 0x19)
        cmds += synth.wait(44100) + synth.ym2413(0x20, 0x09)
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(
            bytes(cmds), 44100, raw_clocks={0x10: 3579545 | 0x80000000},
        )))
        self.assertEqual(vgm.chip_flags.get("ym2413_vrc7"), 1)
        result = convert_vgm(vgm)
        self.assertEqual(result.notes, 1)
        self.assertIn("VRC7", result.text)
        self.assertIn("VRC7 1 ch 1", result.text)
        note = _only_note(vgm, "opll")
        # VRC7 preset 1 (Violin) is a different factory bank from OPLL's.
        self.assertEqual(
            note.voice.ops,
            (
                (0, 14, 8, 2, 0, 4, 5, 0, 0, 3, 0),
                (0, 8, 1, 7, 1, 2, 0, 0, 0, 1, 0),
            ),
        )

    def test_opll_rhythm_bd_track(self):
        cmds = bytearray()
        cmds += synth.ym2413(0x0E, 0x20)
        cmds += synth.ym2413(0x16, 0x21)
        cmds += synth.ym2413(0x26, 0x1B)
        cmds += synth.ym2413(0x36, 0)
        cmds += synth.ym2413(0x0E, 0x30)
        cmds += synth.wait(44100) + synth.ym2413(0x0E, 0x20)
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(bytes(cmds), 44100, clocks={0x10: 3579545})))
        result = convert_vgm(vgm)
        self.assertEqual(result.stats["ym2413_0"]["drums"], 1)
        self.assertIn("YM2413 1 BD", result.text)

    def test_ay_a4(self):
        cmds = bytearray()
        for reg, val in ((0, 0xFE), (1, 0), (7, 0xFE), (8, 15), (9, 0), (10, 0)):
            cmds += synth.ay8910(reg, val)
        cmds += synth.wait(44100) + synth.ay8910(8, 0)
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(bytes(cmds), 44100, clocks={0x74: 1789772})))
        result = convert_vgm(vgm)
        note = _only_note(vgm, "ay")
        self.assertAlmostEqual(note.pitch / 64.0, 69.0, delta=0.05)
        # A standalone AY log has no FM partner and keeps full output.
        self.assertAlmostEqual(note.gain, 1.0, delta=0.01)
        self.assertIn("%0", result.text)

    def test_sn76489_channel_headroom(self):
        from vgm2sionmml.chips.psg import _sn_gain

        self.assertEqual(_sn_gain(0), 1.0)
        self.assertEqual(_sn_gain(0, headroom=True), 0.25)
        self.assertEqual(_sn_gain(15, headroom=True), 0.0)
        self.assertLess(_sn_gain(8, headroom=True), _sn_gain(0, headroom=True))

    def test_sn76489_headroom_only_when_fm_present(self):
        cmds = bytearray([0x50, 0x8E, 0x50, 0x0F, 0x50, 0x90])
        cmds += synth.ym2612(0, 0x22, 0x00)
        cmds += synth.wait(44100) + bytes([0x50, 0x9F])
        vgm = load_vgm(
            self._tmp_vgm(
                synth.make_vgm(
                    bytes(cmds),
                    44100,
                    clocks={0x0C: 3579545, 0x2C: 7670453},
                )
            )
        )
        note = _only_note(vgm, "sn")
        self.assertAlmostEqual(note.gain, 0.25, delta=0.01)

    def test_sn76489_a4(self):
        cmds = bytearray([0x50, 0x8E, 0x50, 0x0F, 0x50, 0x90])
        cmds += synth.wait(44100) + bytes([0x50, 0x9F])
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(bytes(cmds), 44100, clocks={0x0C: 3579545})))
        result = convert_vgm(vgm)
        note = _only_note(vgm, "sn")
        self.assertAlmostEqual(note.pitch / 64.0, 69.0, delta=0.05)




    def test_ym2608_ssg_clock_is_master_divided_by_4(self):
        # divider 284 at 7.9872 MHz / 4 = 1.9968 MHz -> 440 Hz.
        cmds = bytearray()
        for reg, val in (
            (0, 284 & 0xFF), (1, 1), (7, 0xFE), (8, 15), (9, 0), (10, 0),
        ):
            cmds += synth.ym2608(0, reg, val)
        cmds += synth.wait(44100)
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(bytes(cmds), 44100, clocks={0x48: 7987200})))
        self.assertEqual(vgm.clock("ym2608_ssg"), 1_996_800)
        from vgm2sionmml.model import VoiceLibrary
        from vgm2sionmml.chips import extract_ay8910
        result = extract_ay8910(vgm, VoiceLibrary(), chip="ym2608_ssg")
        note = [n for t in result.tracks for n in t.timeline.notes][0]
        self.assertAlmostEqual(note.pitch / 64.0, 69.0, delta=0.05)

    def test_ym2203_a4_uses_72_divisor(self):
        # YM2203's FM prescaler is /72, not the YM2612's /144.  Block 4 /
        # fnum 1040 at 3.9936 MHz is A4 = 440 Hz.
        cmds = bytearray()
        cmds += synth.ym2203(0x22, 0) + synth.ym2203(0x27, 0)
        cmds += synth.ym2203(0xB4, 0xC0) + synth.ym2203(0xB0, 0x07)
        cmds += synth.ym2203(0xA4, 0x24) + synth.ym2203(0xA0, 0x10)
        for op in range(4):
            base = 0x30 + op * 4
            cmds += (
                synth.ym2203(base, 1)
                + synth.ym2203(base + 0x10, 0)
                + synth.ym2203(base + 0x20, 0x1F)
                + synth.ym2203(base + 0x30, 0)
                + synth.ym2203(base + 0x40, 0)
                + synth.ym2203(base + 0x50, 0xF0)
            )
        cmds += synth.ym2203(0x28, 0xF0) + synth.wait(44100)
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(bytes(cmds), 44100, clocks={0x44: 3993600})))
        from vgm2sionmml.model import VoiceLibrary
        from vgm2sionmml.chips import extract_opn
        result = extract_opn(vgm, VoiceLibrary(), "ym2203")
        note = [n for t in result.tracks for n in t.timeline.notes][0]
        self.assertAlmostEqual(note.pitch / 64.0, 69.0, delta=0.05)

    def test_opn_stereo_and_ssg(self):
        cmds = bytearray()
        cmds += synth.ym2612(0, 0x22, 0) + synth.ym2612(0, 0xB4, 0x40)
        cmds += synth.ym2612(0, 0xA0, 0x1D) + synth.ym2612(0, 0xA4, 0x2A)
        cmds += synth.ym2612(0, 0xB0, 0xC7)
        cmds += synth.ym2612(0, 0x28, 0xF0) + synth.wait(44100) + synth.ym2612(0, 0x28, 0)
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(bytes(cmds), 44100, clocks={0x2C: 7680000})))
        note = _only_note(vgm, "opn")
        self.assertEqual(round(note.pan), 8)


    def test_nes_triangle_uses_ma3_and_folds_low_octaves(self):
        # NES timer 1000 -> ~111.7 Hz (A2).  The triangle moves to %3@16 so
        # SiON's PSG pitch table cannot fold the octave.
        timer = 1000
        cmds = bytearray()
        for reg, val in (
            (0x08, 0x80), (0x0A, timer & 0xFF),
            (0x0B, ((timer >> 8) & 7) | 0x20),
        ):
            cmds += synth.nes(reg, val)
        cmds += synth.wait(44100) + synth.nes(0x08, 0x00)
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(bytes(cmds), 44100, clocks={0x84: 1789773})))
        from vgm2sionmml.model import VoiceLibrary
        from vgm2sionmml.chips import extract_nes
        result = extract_nes(vgm, VoiceLibrary())
        tri = [t for t in result.tracks if "triangle" in t.label]
        self.assertEqual(len(tri), 1)
        self.assertEqual(tri[0].module, 3)
        note = tri[0].timeline.notes[0]
        self.assertEqual(note.voice.wave, 16)
        self.assertGreaterEqual(note.pitch / 64.0, 24.0)

    def test_opm_alg4_operator_order(self):
        # C1/C2 (logical ops 1 and 3) are the carriers; M1/M2 are silent
        # modulators.  The #OPM@ array must be in M1,C1,M2,C2 order.
        cmds = bytearray()
        cmds += synth.opm(0x20, 0xC4) + synth.opm(0x28, 0x4A)
        for op in range(4):
            tl = 0 if op in (2, 3) else 63
            cmds += (
                synth.opm(0x40 + 8 * op, 1)
                + synth.opm(0x60 + 8 * op, tl)
                + synth.opm(0x80 + 8 * op, 0x1F)
                + synth.opm(0xA0 + 8 * op, 0)
                + synth.opm(0xC0 + 8 * op, 0)
                + synth.opm(0xE0 + 8 * op, 0xF0)
            )
        cmds += synth.opm(0x08, 0x78) + synth.wait(44100) + synth.opm(0x08, 0)
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(bytes(cmds), 44100, clocks={0x30: 3579545})))
        note = _only_note(vgm, "opm")
        tls = [op[5] for op in note.voice.ops]
        self.assertEqual(tls, [63, 0, 63, 0])

    def test_opl4_alg2_operator_order(self):
        cmds = bytearray()
        cmds += synth.ymf262(1, 0x04, 1) + synth.ymf262(1, 0x05, 1)
        for slot in (0x00, 0x03, 0x08, 0x0B):
            tl = 0 if slot in (0x03, 0x0B) else 63
            for reg, val in (
                (0x20 + slot, 1), (0x40 + slot, tl), (0x60 + slot, 0xF0),
                (0x80 + slot, 0), (0xE0 + slot, 0),
            ):
                cmds += synth.ymf262(0, reg, val)
        for reg, val in ((0xA0, 0x44), (0xA3, 0x44), (0xB0, 0x32), (0xB3, 0x32)):
            cmds += synth.ymf262(0, reg, val)
        cmds += synth.ymf262(0, 0xC0, 0x30) + synth.ymf262(0, 0xC3, 0x31)
        cmds += synth.wait(44100)
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(bytes(cmds), 44100, clocks={0x5C: 14318180})))
        note = _only_note(vgm, "opl")
        self.assertEqual(len(note.voice.ops), 4)
        self.assertEqual([op[6] for op in note.voice.ops], [0, 0, 0, 0])
        self.assertEqual(note.op_tl, ((0, 63), (2, 63)))


    def test_dual_nes_instances_keep_separate_state(self):
        def nes_reg(instance: int, reg: int, val: int) -> bytes:
            return bytes([0xB4, (0x80 if instance else 0) | reg, val])

        cmds = bytearray()
        for instance, timer in ((0, 1000), (1, 500)):
            cmds += nes_reg(instance, 0x00, 0x3F)
            cmds += nes_reg(instance, 0x02, timer & 0xFF)
            cmds += nes_reg(instance, 0x03, (1 << 3) | ((timer >> 8) & 7))
        cmds += synth.wait(2000)
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(
            bytes(cmds), 2000, raw_clocks={0x84: 0x40000000 | 1_789_773},
        )))
        from vgm2sionmml.chips import extract_nes
        result0 = extract_nes(vgm, VoiceLibrary(), instance=0)
        result1 = extract_nes(vgm, VoiceLibrary(), instance=1)
        self.assertEqual(len(result0.tracks), 1)
        self.assertEqual(len(result1.tracks), 1)
        self.assertIn("pulse 1", result0.tracks[0].label)
        self.assertIn("#2 pulse 1", result1.tracks[0].label)
        self.assertNotEqual(
            round(result0.tracks[0].timeline.notes[0].pitch),
            round(result1.tracks[0].timeline.notes[0].pitch),
        )

    def test_nes_ultrasonic_timer_is_silent_not_clamped(self):
        cmds = (
            synth.nes(0x00, 0x3F)
            + synth.nes(0x02, 0x00)
            + synth.nes(0x03, 0x08)
            + synth.wait(2000)
        )
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(
            bytes(cmds), 2000, clocks={0x84: 1_789_773},
        )))
        from vgm2sionmml.chips import extract_nes
        result = extract_nes(vgm, VoiceLibrary())
        self.assertEqual(result.stats["notes"], 0)
        self.assertFalse(any("clamped" in w for w in result.warnings))

    def test_nes_noise_same_sample_period(self):
        cmds = bytearray()
        cmds += synth.nes(0x0C, 0x1F)
        cmds += synth.nes(0x0E, 0x03)
        cmds += synth.nes(0x0F, 0x08)
        cmds += synth.wait(1000)
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(bytes(cmds), 1000, clocks={0x84: 1789773})))
        from vgm2sionmml.model import VoiceLibrary
        from vgm2sionmml.chips import extract_nes
        result = extract_nes(vgm, VoiceLibrary())
        noise = [t for t in result.tracks if "noise" in t.label]
        self.assertEqual(len(noise), 1)
        note = noise[0].timeline.notes[0]
        # APU noise indices are 1-based and must stay inside the 16-entry
        # noise pitch table.
        self.assertAlmostEqual(note.pitch / 64.0, 4.0, delta=0.01)
        self.assertEqual(note.voice.wave, 9)

    def test_opl_non_sustaining_voice_gets_release_time(self):
        from vgm2sionmml.chips.opl import _opl_auto_release_samples

        ops = (
            (0, 15, 15, 15, 0, 0, 0, 0, 0, 1, 0),
            (0, 15, 15, 15, 0, 0, 0, 0, 0, 1, 0),
        )
        frames = _opl_auto_release_samples(0, ops, 3_579_545)
        self.assertIsNotNone(frames)
        self.assertLess(frames, 1000)

    def test_opl2_channel3_operator_addresses(self):
        # Channel 3's operators sit at slots 8/0x0B.  Reading them through the
        # channel number instead of the slot address used to duplicate op2.
        cmds = bytearray()
        for reg, val in (
            (0x28, 0x01), (0x48, 0x20), (0x68, 0xF0), (0x88, 0),
            (0x2B, 0x02), (0x4B, 0x10), (0x6B, 0xF0), (0x8B, 0),
            (0xA3, 0x44), (0xC3, 0x00), (0xB3, 0x32),
        ):
            cmds += synth.ym3812(reg, val)
        cmds += synth.wait(1000)
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(bytes(cmds), 1000, clocks={0x50: 3579545})))
        from vgm2sionmml.model import VoiceLibrary
        from vgm2sionmml.chips import extract_opl
        result = extract_opl(vgm, VoiceLibrary(), "ym3812")
        notes = [n for t in result.tracks for n in t.timeline.notes]
        self.assertEqual(len(notes), 1)
        ops = notes[0].voice.ops
        self.assertEqual(ops[0][9], 1)   # op1 MUL
        self.assertEqual(ops[1][9], 2)   # op2 MUL
        self.assertEqual([op[6] for op in ops], [0, 0])
        self.assertEqual(notes[0].op_tl, ((0, 32),))

    def test_opl_multiplier_compatibility_mapping(self):
        # SiON's #OPL@ parser maps OPL multiplier entries 11/13/14 to the
        # generic OPM table values 10/12/15.  Keep the voice equivalent.
        cmds = bytearray()
        for reg, val in (
            (0x20, 11), (0x23, 14), (0x40, 0), (0x43, 0),
            (0x60, 0xF0), (0x63, 0xF0), (0x80, 0), (0x83, 0),
            (0xA0, 0x44), (0xC0, 0), (0xB0, 0x32),
        ):
            cmds += synth.ym3812(reg, val)
        cmds += synth.wait(1000)
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(bytes(cmds), 1000, clocks={0x50: 3579545})))
        note = _only_note(vgm, "opl")
        self.assertEqual(note.voice.ops[0][9], 10)
        self.assertEqual(note.voice.ops[1][9], 15)

    def test_opl_rhythm_drums(self):
        cmds = bytearray()
        cmds += synth.ymf262(0, 0xBD, 0x20)
        for reg, val in (
            (0x20, 0x21), (0x23, 0x21), (0x40, 0x10), (0x43, 0x00),
            (0x60, 0xF0), (0x63, 0xF0), (0x80, 0x00), (0x83, 0x00),
            (0xA0, 0x44), (0xB0, 0x12), (0xC0, 0x30),
        ):
            cmds += synth.ymf262(0, reg, val)
        cmds += synth.ymf262(0, 0xBD, 0x30) + synth.wait(20000)
        cmds += synth.ymf262(0, 0xBD, 0x20) + synth.wait(20000)
        cmds += synth.ymf262(0, 0xBD, 0x30) + synth.wait(1000)
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(bytes(cmds), 41000, clocks={0x5C: 14318180})))
        from vgm2sionmml.model import VoiceLibrary
        from vgm2sionmml.chips import extract_opl
        result = extract_opl(vgm, VoiceLibrary(), "ymf262")
        # The bass drum is normal FM channel 6 (one-based channel 7), not a
        # separate one-operator drum track.
        drums = [t for t in result.tracks if t.label.endswith("ch 7")]
        self.assertEqual(len(drums), 1)
        self.assertEqual(len(drums[0].timeline.notes), 2)

    def test_opl_rhythm_linked_pitch_changes(self):
        cmds = bytearray()
        cmds += synth.ymf262(0, 0xBD, 0x20)
        for reg, val in (
            (0x11, 0x21), (0x14, 0x21),
            (0x51, 0), (0x54, 0), (0x71, 0xF0), (0x74, 0xF0),
            (0x91, 0), (0x94, 0), (0xA7, 0x44), (0xB7, 0x12),
            (0xC7, 0x30),
        ):
            cmds += synth.ymf262(0, reg, val)
        cmds += synth.ymf262(0, 0xBD, 0x28) + synth.wait(1000)
        cmds += synth.ymf262(0, 0xA7, 0x54) + synth.ymf262(0, 0xB7, 0x12)
        cmds += synth.wait(1000)
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(bytes(cmds), 2000, clocks={0x5C: 14318180})))
        from vgm2sionmml.chips import extract_opl
        result = extract_opl(vgm, VoiceLibrary(), "ymf262")
        sd = [t for t in result.tracks if t.label.endswith(" SD")]
        self.assertEqual(len(sd), 1)
        note = sd[0].timeline.notes[0]
        self.assertTrue(note.pitch_points)
        self.assertNotEqual(note.pitch_points[-1][1], note.pitch)

    def test_opl_tl_change_becomes_gain(self):
        cmds = bytearray()
        for reg, val in (
            (0x20, 0x01), (0x23, 0x21), (0x40, 0x38), (0x43, 0),
            (0x60, 0xF0), (0x63, 0xF0), (0x80, 0), (0x83, 0),
            (0xA0, 0x44), (0xC0, 0), (0xB0, 0x32),
        ):
            cmds += synth.ym3812(reg, val)
        cmds += synth.wait(10000) + synth.ym3812(0x43, 0x20) + synth.wait(10000)
        cmds += synth.ym3812(0xB0, 0x12)
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(bytes(cmds), 21000, clocks={0x50: 3579545})))
        note = _only_note(vgm, "opl")
        self.assertTrue(note.gain_points)
        self.assertLess(note.gain_points[-1][1], note.gain)

    def test_sn_gg_stereo(self):
        cmds = bytearray([0x4F, 0x10]) + bytearray([0x50, 0x8E, 0x50, 0x0F, 0x50, 0x90])
        cmds += synth.wait(44100) + bytes([0x50, 0x9F])
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(bytes(cmds), 44100, clocks={0x0C: 3579545})))
        note = _only_note(vgm, "sn")
        self.assertEqual(round(note.pan), 8)

    def test_nes_a4(self):
        cmds = bytearray()
        for reg, val in ((0x00, 0xBF), (0x02, 0xFD), (0x03, 0x20)):
            cmds += synth.nes(reg, val)
        cmds += synth.wait(44100) + synth.nes(0x00, 0x30)
        vgm = load_vgm(self._tmp_vgm(synth.make_vgm(bytes(cmds), 44100, clocks={0x84: 1789773})))
        result = convert_vgm(vgm)
        note = _only_note(vgm, "nes")
        self.assertAlmostEqual(note.pitch / 64.0, 69.0, delta=0.05)


def _timeline(voice: FmVoice):
    from vgm2sionmml.model import ChannelTimeline

    tl = ChannelTimeline("test")
    tl.note_on(0, 69 * 64, voice, 1.0, 4.0)
    tl.note_off(44100)
    return tl


def _only_note(vgm, family: str):
    from vgm2sionmml.chips import (
        extract_ay8910,
        extract_nes,
        extract_opl,
        extract_opll,
        extract_opn,
        extract_sn76489,
        extract_ym2151,
    )
    from vgm2sionmml.model import VoiceLibrary

    voices = VoiceLibrary()
    if family == "opm":
        result = extract_ym2151(vgm, voices)
    elif family == "opn":
        result = extract_opn(vgm, voices, "ym2612")
    elif family == "opl":
        kind = "ym3812" if any(w.chip == "ym3812" for w in vgm.writes) else "ymf262"
        result = extract_opl(vgm, voices, kind)
    elif family == "opll":
        result = extract_opll(vgm, voices)
    elif family == "ay":
        result = extract_ay8910(vgm, voices)
    elif family == "sn":
        result = extract_sn76489(vgm, voices)
    elif family == "nes":
        result = extract_nes(vgm, voices)
    else:
        raise AssertionError(family)
    notes = [n for t in result.tracks for n in t.timeline.notes]
    assert len(notes) == 1, [(t.label, len(t.timeline.notes)) for t in result.tracks]
    return notes[0]


class CompileTests(unittest.TestCase):
    """End-to-end: generated MML must compile in the reference SiON player."""

    @unittest.skipUnless(synth.sion_play_available(), "sion-play not available")
    def test_opll_output_compiles(self):
        cmds = bytearray()
        cmds += synth.ym2413(0x30, 0x10)
        cmds += synth.ym2413(0x10, 0x21)
        cmds += synth.ym2413(0x20, 0x1B)
        cmds += synth.ym2413(0x0E, 0x20)
        cmds += synth.ym2413(0x16, 0x21)
        cmds += synth.ym2413(0x26, 0x1B)
        cmds += synth.ym2413(0x36, 0)
        cmds += synth.ym2413(0x0E, 0x30)
        cmds += synth.wait(44100)
        cmds += synth.ym2413(0x20, 0x0B) + synth.ym2413(0x0E, 0x20)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.vgm"
            path.write_bytes(synth.make_vgm(bytes(cmds), 44100, clocks={0x10: 3579545}))
            result = convert_vgm(load_vgm(path))
            wav = synth.compile_mml(result.text, Path(td))
            self.assertTrue(synth.wav_has_signal(wav))

    @unittest.skipUnless(synth.sion_play_available(), "sion-play not available")
    def test_opm_output_compiles(self):
        cmds = bytearray()
        cmds += synth.opm(0x20, 0xC7) + synth.opm(0x28, 0x4A) + synth.opm(0x30, 0)
        for op in range(4):
            cmds += (
                synth.opm(0x40 + 8 * op, 1)
                + synth.opm(0x60 + 8 * op, 0)
                + synth.opm(0x80 + 8 * op, 0x1F)
                + synth.opm(0xA0 + 8 * op, 0)
                + synth.opm(0xC0 + 8 * op, 0)
                + synth.opm(0xE0 + 8 * op, 0xF0)
            )
        cmds += synth.opm(0x08, 0x78) + synth.wait(44100) + synth.opm(0x08, 0)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.vgm"
            path.write_bytes(synth.make_vgm(bytes(cmds), 44100, clocks={0x30: 3579545}))
            result = convert_vgm(load_vgm(path))
            wav = synth.compile_mml(result.text, Path(td))
            self.assertTrue(synth.wav_has_signal(wav))


if __name__ == "__main__":
    unittest.main()
