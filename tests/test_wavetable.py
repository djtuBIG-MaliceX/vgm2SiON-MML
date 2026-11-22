from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vgm2sionmml.convert import convert_vgm
from vgm2sionmml.chips import (
    extract_gb,
    extract_huc6280,
    extract_k005289,
    extract_scc,
    extract_vsu,
)
from vgm2sionmml.model import VoiceLibrary, WaveTableLibrary
from vgm2sionmml.optimize import compress_loops, detect_bpm, find_macros
from vgm2sionmml.envelopes import assign_envelopes
from vgm2sionmml.model import ChannelTimeline, Note, SimpleVoice, Song, Track, VoiceLibrary
from vgm2sionmml.mml import MmlRenderer
from vgm2sionmml.vgm import load_vgm
from tests import synth


class WavetableTests(unittest.TestCase):
    def _load(self, commands: bytes, total: int, clocks: dict[int, int]):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        path = Path(td.name) / "t.vgm"
        path.write_bytes(synth.make_vgm(commands, total, clocks=clocks))
        return load_vgm(path)

    def test_scc_note_and_wave_table(self):
        cmds = bytearray()
        for i, value in enumerate(range(32)):
            cmds += synth.scc(0, i, value)
        cmds += synth.scc(1, 0, 253) + synth.scc(1, 1, 0)
        cmds += synth.scc(2, 0, 15) + synth.scc(3, 0, 1)
        cmds += synth.wait(1000)
        vgm = self._load(bytes(cmds), 1000, {0x9C: 1_789_772})
        waves = WaveTableLibrary()
        result = extract_scc(vgm, waves)
        self.assertEqual(len(waves.tables), 1)
        notes = [n for t in result.tracks for n in t.timeline.notes]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].voice.module, 4)
        self.assertAlmostEqual(notes[0].pitch / 64.0, 69.0, delta=0.08)

    def test_huc6280_note_and_wave_table(self):
        cmds = bytearray()
        cmds += synth.huc(0, 0)
        for i in range(32):
            cmds += synth.huc(6, i)
        cmds += synth.huc(2, 254) + synth.huc(3, 0)
        cmds += synth.huc(4, 0x8F) + synth.huc(5, 0xFF)
        cmds += synth.wait(1000)
        vgm = self._load(bytes(cmds), 1000, {0xA4: 3_579_545})
        waves = WaveTableLibrary()
        result = extract_huc6280(vgm, waves)
        notes = [n for t in result.tracks for n in t.timeline.notes]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].voice.module, 4)
        self.assertAlmostEqual(notes[0].pitch / 64.0, 69.0, delta=0.08)

    def test_huc6280_drops_sub_frame_percussion_gates(self):
        cmds = bytearray()
        cmds += synth.huc(0, 5)
        cmds += synth.huc(7, 0x9F)
        cmds += synth.huc(4, 0x9F)
        cmds += synth.wait(50)
        cmds += synth.huc(4, 0)
        cmds += synth.wait(500)
        vgm = self._load(bytes(cmds), 550, {0xA4: 3_579_545})
        result = extract_huc6280(vgm, WaveTableLibrary())
        self.assertEqual(result.stats["glitches"], 1)
        self.assertFalse(any("noise" in t.label for t in result.tracks))
        self.assertTrue(any("sub-frame" in w for w in result.warnings))

    def test_vsu_note_and_wave_table(self):
        cmds = bytearray()
        for i in range(32):
            cmds += synth.vsu(i, i)
        cmds += synth.vsu(0x100, 0x00)  # INTL off
        cmds += synth.vsu(0x102, 1693 & 0xFF) + synth.vsu(0x103, (1693 >> 8) & 7)
        cmds += synth.vsu(0x101, 0xFF) + synth.vsu(0x104, 0xF0)
        cmds += synth.vsu(0x106, 0)
        cmds += synth.vsu(0x100, 0x80)
        cmds += synth.wait(1000)
        vgm = self._load(bytes(cmds), 1000, {0xC4: 5_000_000})
        waves = WaveTableLibrary()
        result = extract_vsu(vgm, waves)
        notes = [n for t in result.tracks for n in t.timeline.notes]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].voice.module, 4)
        self.assertAlmostEqual(notes[0].pitch / 64.0, 69.0, delta=0.1)

    def test_gb_pulse_wave_and_noise(self):
        cmds = bytearray()
        # Pulse 1: 12.5% duty, full volume, 440 Hz.
        cmds += synth.gb_apu(0x01, 0x00) + synth.gb_apu(0x02, 0xF0)
        cmds += synth.gb_apu(0x03, 1750 & 0xFF)
        cmds += synth.gb_apu(0x04, 0x80 | ((1750 >> 8) & 7))
        # Wave: 32 nibbles, 50% level, 440 Hz.
        for i in range(16):
            cmds += synth.gb_apu(0x20 + i, i * 0x11)
        cmds += synth.gb_apu(0x0A, 0x80) + synth.gb_apu(0x0C, 0x20)
        cmds += synth.gb_apu(0x0D, 1750 & 0xFF)
        cmds += synth.gb_apu(0x0E, 0x80 | ((1750 >> 8) & 7))
        # Noise.
        cmds += synth.gb_apu(0x11, 0xF0) + synth.gb_apu(0x12, 0)
        cmds += synth.gb_apu(0x13, 0x80)
        cmds += synth.wait(1000)
        vgm = self._load(bytes(cmds), 1000, {0x80: 4_194_304})
        waves = WaveTableLibrary()
        result = extract_gb(vgm, waves)
        notes = [n for t in result.tracks for n in t.timeline.notes]
        self.assertGreaterEqual(len(notes), 3)
        self.assertTrue(any(n.voice.module == 8 for n in notes))
        self.assertTrue(any(n.voice.module == 4 for n in notes))
        self.assertTrue(any(n.voice.module == 2 for n in notes))

    def test_k005289_prom_and_note(self):
        prom = bytes((i & 0x0F for i in range(0x200)))
        cmds = bytearray(synth.data_block(0xC3, 0, prom))
        cmds += synth.k005289(2, 0xF02)  # latch period 253
        cmds += synth.k005289(4, 0)      # trigger it
        cmds += synth.k005289(0, 0x0F)   # volume 15, bank 0
        cmds += synth.wait(1000)
        vgm = self._load(bytes(cmds), 1000, {0xEC: 3_579_545})
        self.assertTrue(vgm.data_blocks)
        waves = WaveTableLibrary()
        result = extract_k005289(vgm, waves)
        self.assertEqual(len(waves.tables), 1)
        notes = [n for t in result.tracks for n in t.timeline.notes]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].voice.module, 4)
        self.assertAlmostEqual(notes[0].pitch / 64.0, 69.0, delta=0.1)

    def test_scc_output_compiles(self):
        if not synth.sion_play_available():
            self.skipTest("sion-play not available")
        cmds = bytearray()
        for i, value in enumerate(range(32)):
            cmds += synth.scc(0, i, value)
        cmds += synth.scc(1, 0, 253) + synth.scc(1, 1, 0)
        cmds += synth.scc(2, 0, 15) + synth.scc(3, 0, 1)
        cmds += synth.wait(44100)
        vgm = self._load(bytes(cmds), 44100, {0x9C: 1_789_772})
        result = convert_vgm(vgm)
        self.assertIn("#WAVB0", result.text)
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            wav = synth.compile_mml(result.text, Path(td))
            self.assertTrue(synth.wav_has_signal(wav))

    def test_macros_need_four_commands_and_ten_repeats(self):
        body = ["c", "d", "e", "f"]
        short = body * 10
        long = body * 11
        self.assertEqual(find_macros([short])[0], [])
        macros, _ = find_macros([long])
        self.assertEqual(len(macros), 1)
        self.assertEqual(macros[0][1], body)

    def test_macros_need_three_notes(self):
        # Four commands, but only two are notes: don't spend a macro on it.
        body = ["c", "d", "x1", "x2"]
        self.assertEqual(find_macros([body * 20])[0], [])
        body = ["c", "d", "e", "x1"]
        macros, _ = find_macros([body * 20])
        self.assertEqual(len(macros), 1)

    def test_pitch_and_gain_table_envelopes_render(self):
        tl = ChannelTimeline("tables")
        for start in (0, 44100):
            note = tl.note_on(
                start, 69 * 64, SimpleVoice(0, 0), gain=0.5, pan=4.0
            )
            for i in range(1, 20):
                tick = start + i * 735
                note.add_pitch(tick, 69 * 64 + i * 64)
                note.add_gain(tick, 0.5 + (i % 5) * 0.1)
            note.close(start + 44100)
        tracks = [Track("t", 0, "psg", tl)]
        tables, stats = assign_envelopes(tracks, 88200)
        self.assertGreater(len(tables.tables), 0)
        self.assertGreater(stats["pitch_tables"], 0)
        self.assertGreater(stats["gain_tables"], 0)
        song = Song("T", tracks, VoiceLibrary(), 88200, tables=tables)
        text = MmlRenderer(song, optimize=False).render()
        self.assertIn("#TABLE", text)
        self.assertTrue("np" in text or "nt" in text)
        self.assertIn("na", text)
        self.assertNotIn("&", text.split("// t", 1)[1])
        if synth.sion_play_available():
            with tempfile.TemporaryDirectory() as td:
                wav = synth.compile_mml(text, Path(td), "tables")
                self.assertTrue(synth.wav_has_signal(wav))

    def test_optimizer_loops_and_detects_frame_tempo(self):
        toks = ["c16", "d16", "e16", "f16", "c16", "d16", "e16", "f16"]
        out, loops = compress_loops(toks)
        self.assertGreaterEqual(loops, 1)
        self.assertIn("[", out)
        tl = ChannelTimeline("t")
        for i in range(8):
            tl.note_on(i * 735, 69 * 64, None, 1.0, 4.0)
            tl.note_off(i * 735 + 735)
        bpm, _ = detect_bpm([Track("t", 6, "fm", tl)], 8 * 735, 60)
        self.assertEqual(bpm, 120)
