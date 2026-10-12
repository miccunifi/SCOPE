"""Guard the released full-model defaults and isolation of the JPEG2000 study."""
from pathlib import Path
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]
FULL_CKPT = "runs/preedit_d2fpn_nojp2_unbiased_linearsev_mediumbpg_sevbasis_scratch_FULL_20260701_140557/best_val.pt"
OUTPUTS = {
    "eval.yaml": "outputs_scope_full_main",
    "ablation.yaml": "outputs_scope_full_jpeg2000",
    "timing.yaml": "outputs_scope_full_timing",
}

class FullCheckpointDefaultsTests(unittest.TestCase):
    def test_all_configs_default_to_full_weights(self):
        for filename, out in OUTPUTS.items():
            with self.subTest(config=filename):
                cfg = yaml.safe_load((ROOT / "configs" / filename).read_text())
                self.assertEqual(cfg["output_dir"], out)
                self.assertEqual(cfg["scope_common"]["scope_ckpt"], FULL_CKPT)
                self.assertEqual(cfg["scope_common"]["condition_mode"], "family_severity")
                self.assertTrue(any(exp["mode"].startswith("scope_") for exp in cfg["experiments"]))
                for exp in cfg["experiments"]:
                    if exp["mode"].startswith("scope_"):
                        self.assertEqual(exp["scope_ckpt"], FULL_CKPT)

    def test_main_eval_has_no_wavelet_family_or_jpeg2000_templates(self):
        cfg = yaml.safe_load((ROOT / "configs/eval.yaml").read_text())
        content = (ROOT / "configs/eval.yaml").read_text()
        self.assertNotIn("wavelet", content.lower())
        self.assertFalse(any("jp2" in name.lower() for name in cfg["codec_templates"]))
        self.assertFalse(any("jpeg2000" in exp["mode"].lower() or "jp2" in exp["name"].lower() for exp in cfg["experiments"]))
        self.assertFalse(any(exp.get("codec_family") in {"wavelet", "jp2_like"} for exp in cfg["experiments"]))

    def test_jpeg2000_baseline_is_not_a_scope_condition(self):
        cfg = yaml.safe_load((ROOT / "configs/ablation.yaml").read_text())
        base = [exp for exp in cfg["experiments"] if exp["mode"] == "jpeg2000"]
        edited = [exp for exp in cfg["experiments"] if exp["mode"].startswith("scope_")]
        self.assertTrue(base and edited)
        self.assertTrue(all(exp["codec_family"] == "wavelet" for exp in base))
        self.assertTrue(all(exp["codec_family"] in {"block_dct", "learned", "intra_smooth"} for exp in edited))

if __name__ == "__main__":
    unittest.main()
