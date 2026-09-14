"""Checks for the run-report figures and tables."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hbn_r3 import report  # noqa: E402


def fake_run(folder, with_budget=True):
    """Minimal set of artefacts an evaluation writes."""
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    pooled = [
        {"model": "mi", "duration": 40, "n": 100, "balanced_accuracy": 0.52,
         "ba_conditional_ci95": [0.47, 0.57]},
        {"model": "proposed", "duration": 40, "n": 100, "balanced_accuracy": 0.55,
         "ba_conditional_ci95": [0.50, 0.60]},
        {"model": "proposed", "duration": 20, "n": 100, "balanced_accuracy": 0.51,
         "ba_conditional_ci95": [0.46, 0.56]},
    ]
    (root / "pooled_metrics.json").write_text(json.dumps(pooled))
    pd.DataFrame([{"held_release": "R4_L100", "model": "proposed", "duration": 40,
                   "n": 25, "balanced_accuracy": 0.55}]).to_csv(
        root / "release_metrics.csv", index=False)
    pd.DataFrame([{"participant_id": "a", "group_id": "a", "release": "R4_L100",
                   "label": 1, "score": 0.7, "threshold": 0.5, "prediction": 1,
                   "score_kind": "probability", "probability": 0.7,
                   "model": "proposed", "duration": 40}]).to_csv(
        root / "oof_predictions.csv", index=False)
    (root / "paired_comparisons.json").write_text(json.dumps([
        {"model": "proposed", "baseline": "mi", "duration": 40,
         "delta_balanced_accuracy": 0.03, "conditional_family_ci95": [-0.02, 0.08]}]))
    pd.DataFrame([{"model": "proposed", "feature": "f__ec_sp_LP_alpha",
                   "outer_folds_selected": 3, "total_outer_folds": 4}]).to_csv(
        root / "selection_frequency.csv", index=False)
    if with_budget:
        (root / "budget_metrics.json").write_text(json.dumps([
            {"model": "proposed", "policy": "agreement", "release": "pooled",
             "coverage": 0.62, "accepted_balanced_accuracy": 0.68,
             "mean_clean_seconds_per_condition": 31.0},
            {"model": "proposed", "policy": "margin", "release": "pooled",
             "coverage": 0.80, "accepted_balanced_accuracy": 0.58,
             "mean_clean_seconds_per_condition": 36.0}]))
    return root


class ReportTests(unittest.TestCase):
    def test_builds_every_supported_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = fake_run(Path(tmp) / "run")
            out = Path(tmp) / "figures"
            written = report.summarise(run, out, duration=40)
            for name in ("accuracy", "effects", "budget", "frequency"):
                self.assertIn(name, written)
            for stem in ("model_accuracy", "paired_effects",
                         "budget_tradeoff", "selection_frequency"):
                self.assertTrue((out / f"{stem}.pdf").exists(), stem)
                self.assertTrue((out / f"{stem}.png").exists(), stem)

    def test_accuracy_table_filters_to_the_requested_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = fake_run(Path(tmp) / "run")
            written = report.summarise(run, Path(tmp) / "figures", duration=40)
            self.assertEqual(sorted(written["accuracy"].model), ["mi", "proposed"])
            self.assertTrue(np.isclose(
                written["accuracy"].set_index("model").loc["proposed", "balanced_accuracy"],
                0.55))

    def test_runs_without_a_budget_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = fake_run(Path(tmp) / "run", with_budget=False)
            written = report.summarise(run, Path(tmp) / "figures", duration=40)
            self.assertNotIn("budget", written)
            self.assertIn("accuracy", written)


if __name__ == "__main__":
    unittest.main()
