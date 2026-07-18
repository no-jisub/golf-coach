import unittest

from tools.analyze_guide_caddieset_alignment import build_metric_snapshot
from tools.audit_guide_caddieset_alignment import build_alignment_report
from tools.align_guide_poses_to_caddieset import build_aligned_guide
from utils.guide_alignment import (
    STAGE_KEYS,
    MAX_JOINT_DISPLACEMENT,
    align_guide_poses_to_caddieset,
    audit_guide_stage_metrics,
    calculate_guide_stage_metrics,
)
from utils.guide_skeleton import GUIDE_POSES, SWING_HAND


class GuideCaddieSetMetricTests(unittest.TestCase):
    def test_calculates_metrics_for_all_runtime_stages(self):
        direction_multiplier = -1.0 if SWING_HAND == "right" else 1.0
        metrics = calculate_guide_stage_metrics(
            GUIDE_POSES,
            direction_multiplier=direction_multiplier,
        )
        self.assertEqual(tuple(metrics), STAGE_KEYS)
        for stage_key in STAGE_KEYS:
            self.assertEqual(len(metrics[stage_key]), 20)
            self.assertIsNotNone(metrics[stage_key]["shoulder_angle"])

    def test_address_relative_movements_are_zero(self):
        snapshot = build_metric_snapshot()
        address = snapshot["stages"]["address"]
        self.assertAlmostEqual(address["head_loc"], 0.0)
        self.assertAlmostEqual(address["hip_shifted"], 0.0)
        self.assertAlmostEqual(address["hip_rotation"], 0.0)

    def test_snapshot_records_runtime_orientation(self):
        snapshot = build_metric_snapshot()
        self.assertEqual(snapshot["swing_hand"], SWING_HAND)
        self.assertEqual(snapshot["stage_order"], list(STAGE_KEYS))
        self.assertEqual(
            snapshot["direction_multiplier"],
            -1.0 if SWING_HAND == "right" else 1.0,
        )


class GuideCaddieSetAuditTests(unittest.TestCase):
    def test_audit_checks_all_40_faceon_stage_items(self):
        direction_multiplier = -1.0 if SWING_HAND == "right" else 1.0
        metrics = calculate_guide_stage_metrics(
            GUIDE_POSES,
            direction_multiplier=direction_multiplier,
        )
        audit = audit_guide_stage_metrics(metrics)
        self.assertEqual(audit["summary"]["total_count"], 40)
        self.assertEqual(
            audit["summary"]["pass_count"] + audit["summary"]["warning_count"],
            40,
        )
        self.assertEqual(tuple(audit["stages"]), STAGE_KEYS)

    def test_report_keeps_item_level_ranges_and_relations(self):
        report = build_alignment_report()
        shoulder = report["stages"]["address"]["items"]["shoulder_angle"]
        self.assertIsNotNone(shoulder["measured_value"])
        self.assertEqual(len(shoulder["reference_range"]), 2)
        self.assertIn(
            shoulder["relation"],
            {
                "within_reference",
                "below_reference",
                "above_reference",
                "below_outer",
                "above_outer",
            },
        )

    def test_report_summary_matches_stage_summaries(self):
        report = build_alignment_report()
        stage_passes = sum(
            stage["summary"]["pass_count"] for stage in report["stages"].values()
        )
        self.assertEqual(report["summary"]["pass_count"], stage_passes)


class GuideCaddieSetOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.direction_multiplier = -1.0 if SWING_HAND == "right" else 1.0
        cls.aligned, cls.report = align_guide_poses_to_caddieset(
            GUIDE_POSES,
            direction_multiplier=cls.direction_multiplier,
        )

    def test_alignment_never_increases_warning_count(self):
        self.assertLessEqual(
            self.report["after"]["warning_count"],
            self.report["before"]["warning_count"],
        )
        self.assertGreaterEqual(
            self.report["after"]["pass_count"],
            self.report["before"]["pass_count"],
        )

    def test_joint_displacement_stays_within_limit(self):
        for stage_key in STAGE_KEYS:
            self.assertLessEqual(
                self.report["stages"][stage_key]["max_joint_displacement"],
                MAX_JOINT_DISPLACEMENT * 1.01,
            )

    def test_aligned_output_has_all_runtime_landmarks(self):
        for stage_key in STAGE_KEYS:
            self.assertEqual(set(self.aligned[stage_key]), set(GUIDE_POSES[stage_key]))
            for point in self.aligned[stage_key].values():
                self.assertTrue(0.0 <= point[0] <= 1.0)
                self.assertTrue(0.0 <= point[1] <= 1.0)

    def test_serialized_output_records_before_and_after(self):
        output = build_aligned_guide()
        self.assertEqual(output["stage_order"], list(STAGE_KEYS))
        self.assertIn("before", output["alignment"])
        self.assertIn("after", output["alignment"])


if __name__ == "__main__":
    unittest.main()
