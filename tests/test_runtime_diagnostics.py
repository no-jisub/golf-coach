import unittest

import numpy as np

from utils.runtime_diagnostics import (
    build_runtime_diagnostics,
    diagnostic_lines,
    draw_runtime_diagnostics,
)


class RuntimeDiagnosticSnapshotTests(unittest.TestCase):
    def test_flattens_quality_stability_and_scores(self):
        diagnostics = build_runtime_diagnostics(
            phase="analysis",
            pose_detected=True,
            visibility={"passed": True, "missing": [], "clipped": []},
            stability={
                "ready": True,
                "stable": True,
                "duration_sec": 1.67,
                "mean_jitter": 0.0042,
                "max_joint_jitter": 0.011,
            },
            feedback={
                "passed": True,
                "status": "pass",
                "metrics": {
                    "guide_score": 82,
                    "caddieset_score": 76,
                    "final_score": 79,
                },
            },
            pass_progress=0.5,
        )

        self.assertEqual(diagnostics["scores"]["final"], 79)
        self.assertEqual(diagnostics["blocker"], "PASS_HOLD")
        self.assertEqual(diagnostics["stability"]["mean_jitter"], 0.0042)

    def test_full_body_failure_has_priority(self):
        diagnostics = build_runtime_diagnostics(
            phase="calibration",
            pose_detected=True,
            visibility={
                "passed": False,
                "missing": ["오른쪽 발목"],
                "clipped": [],
            },
            address={"passed": False, "score": 30},
        )
        self.assertEqual(diagnostics["blocker"], "FULL_BODY")
        self.assertTrue(
            any("CHECK (1)" in line for line in diagnostic_lines(diagnostics))
        )

    def test_no_pose_is_reported(self):
        diagnostics = build_runtime_diagnostics(
            phase="calibration",
            pose_detected=False,
        )
        self.assertEqual(diagnostics["blocker"], "NO_POSE")


class RuntimeDiagnosticDrawingTests(unittest.TestCase):
    def test_enabled_panel_changes_frame_pixels(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        diagnostics = build_runtime_diagnostics(
            phase="analysis",
            pose_detected=False,
        )
        draw_runtime_diagnostics(frame, diagnostics)
        self.assertGreater(np.count_nonzero(frame), 0)

    def test_disabled_panel_leaves_frame_unchanged(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        diagnostics = build_runtime_diagnostics(
            phase="analysis",
            pose_detected=False,
        )
        draw_runtime_diagnostics(frame, diagnostics, enabled=False)
        self.assertEqual(np.count_nonzero(frame), 0)


if __name__ == "__main__":
    unittest.main()
