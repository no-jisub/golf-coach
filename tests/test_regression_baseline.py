import unittest
from copy import deepcopy

from utils.regression_baseline import (
    build_synthetic_runtime_snapshot,
    compare_runtime_snapshots,
    render_comparison_markdown,
    validate_baseline,
)


class RegressionBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = build_synthetic_runtime_snapshot()

    def test_synthetic_snapshot_runs_all_eight_stages(self):
        validate_baseline(self.snapshot)
        self.assertEqual(len(self.snapshot["stages"]), 8)
        self.assertTrue(all(item["passed"] for item in self.snapshot["stages"].values()))

    def test_identical_snapshot_passes(self):
        comparison = compare_runtime_snapshots(self.snapshot, deepcopy(self.snapshot))
        self.assertTrue(comparison["passed"])
        self.assertEqual(comparison["change_count"], 0)
        self.assertIn("일치합니다", render_comparison_markdown(comparison))

    def test_score_and_status_changes_fail(self):
        changed = deepcopy(self.snapshot)
        changed["stages"]["downswing"]["final_score"] -= 2
        changed["stages"]["follow_through"]["passed"] = False
        comparison = compare_runtime_snapshots(self.snapshot, changed)
        self.assertFalse(comparison["passed"])
        self.assertEqual(comparison["change_count"], 2)

    def test_score_tolerance_allows_small_numeric_drift(self):
        changed = deepcopy(self.snapshot)
        changed["stages"]["impact"]["guide_score"] -= 1
        comparison = compare_runtime_snapshots(
            self.snapshot,
            changed,
            score_tolerance=1,
        )
        self.assertTrue(comparison["passed"])


if __name__ == "__main__":
    unittest.main()
