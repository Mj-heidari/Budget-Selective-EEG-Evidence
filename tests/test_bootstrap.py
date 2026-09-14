"""Cluster bootstrap: correctness, and cost under participant-level blocking."""
import sys
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hbn_r3.evaluation import _balanced, _group_members, cluster_interval, paired_interval  # noqa: E402


def predictions(n=400, seed=3, clusters=None):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    keys = [f"p{i}" for i in range(n)] if clusters is None else \
           [f"fam{i % clusters}" for i in range(n)]
    return pd.DataFrame({
        "participant_id": [f"s{i}" for i in range(n)],
        "group_id": keys, "label": labels,
        "prediction": np.where(rng.random(n) < .65, labels, 1 - labels),
        "probability": rng.random(n), "model": "proposed", "duration": 40,
    })


class BootstrapTests(unittest.TestCase):
    def test_balanced_matches_sklearn(self):
        from sklearn.metrics import balanced_accuracy_score
        rng = np.random.default_rng(0)
        for _ in range(20):
            y = rng.integers(0, 2, 60)
            pred = rng.integers(0, 2, 60)
            if len(np.unique(y)) < 2:
                continue
            self.assertAlmostEqual(_balanced(y, pred), balanced_accuracy_score(y, pred))

    def test_balanced_returns_none_for_one_class(self):
        y = np.ones(10, int)
        self.assertIsNone(_balanced(y, y))

    def test_group_members_partition_every_row(self):
        keys = np.array(["a", "b", "a", "c", "b"])
        members = _group_members(keys)
        self.assertEqual(len(members), 3)
        self.assertEqual(sorted(np.concatenate(members).tolist()), [0, 1, 2, 3, 4])
        self.assertEqual(sorted(members[0].tolist()), [0, 2])

    def test_interval_brackets_the_point_estimate(self):
        rows = predictions()
        low, high = cluster_interval(rows, n_boot=400)
        point = _balanced(rows.label.to_numpy(int), rows.prediction.to_numpy(int))
        self.assertLess(low, point)
        self.assertGreater(high, point)

    def test_interval_is_deterministic_for_a_seed(self):
        rows = predictions()
        self.assertEqual(cluster_interval(rows, 200, seed=11),
                         cluster_interval(rows, 200, seed=11))

    def test_paired_difference_of_identical_models_is_zero(self):
        a = predictions()
        b = a.copy()
        b["model"] = "mi"
        result = paired_interval(pd.concat([a, b], ignore_index=True),
                                 "proposed", "mi", 40, n_boot=200)
        self.assertAlmostEqual(result["delta_balanced_accuracy"], 0.0)
        self.assertEqual(result["n_pairs"], len(a))

    def test_participant_level_blocking_stays_fast(self):
        """One cluster per participant is the realistic case without family IDs."""
        rows = predictions(n=514)
        self.assertEqual(rows.group_id.nunique(), 514)
        start = time.perf_counter()
        cluster_interval(rows, n_boot=1000)
        self.assertLess(time.perf_counter() - start, 10.0,
                        "cluster bootstrap must not scale with per-cluster frame copies")


if __name__ == "__main__":
    unittest.main()
