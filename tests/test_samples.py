from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vgm2sionmml.convert import convert_file
from tests import synth

DOWNLOADS = Path("D:/Downloads")

SAMPLES = {
    "opm": (DOWNLOADS / "FZ2DX" / "03 Start ~ Cholacoray [Round 1].vgz", "#OPM@"),
    "opn": (
        DOWNLOADS / "moreVGMs" / "Mamono Hunter Youko" / "Mamono Hunter Youko - Scene 1 - Flower Field.vgz",
        "#OPN@",
    ),
    "opl3_4op": (
        DOWNLOADS / "PhantomBreakerBattleGrounds_Shinjuku-OPL3_optimized.vgm",
        "#OPL@",
    ),
    "opl3_2op": (DOWNLOADS / "moreVGMs" / "MoonDriverVGM" / "01 OPL3 Demo 001.vgz", "#OPL@"),
    "sn76489": (
        DOWNLOADS / "PowerStrike2GG" / "Power Strike II (GG) - 01 - Theme.vgz",
        "%0",
    ),
    "ay_opn": (
        Path("F:/NASbackup/JamesAlan/_devspace/grbsd/ripwip/_finished/madouars/done/01 Opening Concentro~.vgz"),
        "#OPN@",
    ),
    "nes_ay": (
        DOWNLOADS / "moreVGMs" / "Gimmick!_(Family_Computer)" / "03 Happy Birthday (Stage 1).vgz",
        "%1",
    ),
    "opl2": (
        Path("F:/NASbackup/JamesAlan/_devspace/grbsd/ripwip/_finished/angelempire2/done/16 Enemy Phase 4.vgz"),
        "#OPL@",
    ),
    "ym2203": (
        Path("F:/NASbackup/JamesAlan/_devspace/grbsd/ripwip/_finished/madouars/done/06 A - Fiend Empire.vgz"),
        "#OPN@",
    ),
    "ym2608": (
        Path("D:/Downloads/StarfireOPNA/02 Title Screen.vgz"),
        "#OPN@",
    ),
    "scc": (
        DOWNLOADS / "moreVGMs" / "KittenKaboodleVGM" / "03 Favorite Meow, Cute Meow (Stage 1).vgz",
        "#WAVB",
    ),
    "huc6280": (
        DOWNLOADS / "moreVGMs" / "ShanghaiPCE" / "02 BGM 1.vgz",
        "#WAVB",
    ),
    "vsu": (
        DOWNLOADS / "moreVGMs" / "Jack_Bros._(Nintendo_Virtual_Boy)" / "03 Title.vgz",
        "#WAVB",
    ),
    "gb": (
        DOWNLOADS / "moreVGMs" / "KirbyDreamLand2VGM" / "01 Title Theme.vgz",
        "%8",
    ),
    "k005289": (
        DOWNLOADS / "moreVGMs" / "Gradius VGM Rip" / "03 Challenger 1985 (Stage 1).vgz",
        "#WAVB",
    ),
}


class SampleConversionTests(unittest.TestCase):
    def _check(self, key: str):
        path, marker = SAMPLES[key]
        if not path.is_file():
            self.skipTest(f"sample not present: {path}")
        result = convert_file(path)
        self.assertGreater(result.tracks, 0)
        self.assertGreater(result.notes, 0)
        self.assertIn(marker, result.text)
        if synth.sion_play_available():
            with tempfile.TemporaryDirectory() as td:
                wav = synth.compile_mml(result.text, Path(td), key)
                self.assertTrue(synth.wav_has_signal(wav))
        return result

    def test_opm_sample(self):
        result = self._check("opm")
        self.assertEqual(result.stats["ym2151_0"]["notes"], 2259)

    def test_opn_sample(self):
        result = self._check("opn")
        self.assertEqual(result.stats["ym2612_0"]["notes"], 1000)

    def test_opl3_four_op_sample(self):
        result = self._check("opl3_4op")
        self.assertEqual(result.stats["ymf262_0"]["four_op_notes"], 901)
        self.assertGreater(result.stats["ymf262_0"]["drum_notes"], 0)

    def test_opl3_two_op_sample(self):
        result = self._check("opl3_2op")
        self.assertEqual(result.stats["ymf262_0"]["four_op_notes"], 0)
        self.assertEqual(result.voices, 12)

    def test_opl2_sample(self):
        result = self._check("opl2")
        self.assertEqual(result.stats["ym3812_0"]["notes"], 2518)
        self.assertLessEqual(result.voices, 18)

    def test_ym2203_sample(self):
        result = self._check("ym2203")
        self.assertEqual(result.stats["ym2203_0"]["notes"], 663)

    def test_ym2608_sample_loop_starts_at_intro(self):
        result = self._check("ym2608")
        # Header 0x20 is the loop length (3,758,643 samples); the loop starts
        # at 4,158,189 - 3,758,643 = 399,546.
        self.assertEqual(result.stats["loop_sample"], 399_546)
        self.assertIn("$", result.text)

    def test_sn76489_sample(self):
        self._check("sn76489")

    def test_ay_and_opn_sample(self):
        result = self._check("ay_opn")
        self.assertEqual(result.stats["ym2203_0"]["notes"], 2188)

    def test_nes_and_ay_sample(self):
        self._check("nes_ay")
        result = convert_file(SAMPLES["nes_ay"][0])
        self.assertTrue(any("DPCM" in w for w in result.warnings))

    def test_scc_sample(self):
        result = self._check("scc")
        self.assertGreater(result.stats["waves"], 0)
        self.assertIn("%4", result.text)

    def test_huc6280_sample(self):
        result = self._check("huc6280")
        self.assertGreater(result.stats["waves"], 0)

    def test_vsu_sample(self):
        result = self._check("vsu")
        self.assertGreater(result.stats["waves"], 0)

    def test_gb_sample(self):
        result = self._check("gb")
        self.assertIn("%8", result.text)
        self.assertIn("%2", result.text)

    def test_k005289_sample(self):
        result = self._check("k005289")
        self.assertGreater(result.stats["waves"], 0)


if __name__ == "__main__":
    unittest.main()
