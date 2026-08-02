import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from utils.diagnostic_capture import (
    classify_discrepancy,
    save_runtime_sample,
)
from utils.runtime_sample_replay import replay_runtime_sample


def make_landmarks():
    return [
        SimpleNamespace(
            x=index / 100,
            y=index / 100,
            z=0.0,
            visibility=0.9,
            presence=0.8,
        )
        for index in range(33)
    ]


class DiagnosticCaptureTests(unittest.TestCase):
    def test_classifies_false_reject_and_false_accept(self):
        self.assertEqual(
            classify_discrepancy("expected_pass", {"passed": False}),
            "false_reject",
        )
        self.assertEqual(
            classify_discrepancy("expected_fail", {"passed": True}),
            "false_accept",
        )

    def test_saves_images_landmarks_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            result = save_runtime_sample(
                raw_frame=np.full((80, 120, 3), 30, dtype=np.uint8),
                overlay_frame=np.full((80, 120, 3), 90, dtype=np.uint8),
                stage_key="address",
                landmarks=make_landmarks(),
                diagnostics={"scores": {"final": 65}, "blocker": "POSE_RULES"},
                feedback={"passed": False, "status": "warning"},
                expected_label="expected_pass",
                output_root=directory,
                captured_at=datetime(2026, 7, 27, 12, 30, tzinfo=timezone.utc),
            )

            sample_dir = Path(result["sample_dir"])
            self.assertTrue((sample_dir / "raw.jpg").exists())
            self.assertTrue((sample_dir / "overlay.jpg").exists())
            metadata = json.loads(
                (sample_dir / "sample.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["schema"], "golf-coach-runtime-sample-v1")
            self.assertEqual(metadata["discrepancy"], "false_reject")
            self.assertEqual(len(metadata["landmarks"]), 33)
            self.assertTrue(metadata["local_only"])

    def test_rejects_unknown_expected_label(self):
        with self.assertRaises(ValueError):
            save_runtime_sample(
                raw_frame=np.zeros((10, 10, 3), dtype=np.uint8),
                overlay_frame=np.zeros((10, 10, 3), dtype=np.uint8),
                stage_key="address",
                landmarks=[],
                diagnostics={},
                feedback=None,
                expected_label="unknown",
            )

    def test_saves_reproducible_v3_window_and_replays_it(self):
        with tempfile.TemporaryDirectory() as directory:
            landmarks = make_landmarks()
            result = save_runtime_sample(
                raw_frame=np.full((80, 120, 3), 30, dtype=np.uint8),
                overlay_frame=np.full((80, 120, 3), 90, dtype=np.uint8),
                stage_key="address",
                landmarks=landmarks,
                landmark_samples=[(10.0, landmarks), (11.5, landmarks)],
                decision_frames=[
                    (10.0, np.full((80, 120, 3), 30, dtype=np.uint8)),
                    (11.5, np.full((80, 120, 3), 40, dtype=np.uint8)),
                ],
                calibration_profile={
                    "body_ratio": 4.0,
                    "shoulder_mid": (60, 20),
                    "shoulder_width": 20.0,
                    "samples": 2,
                    "caddieset_address_points": {0: (0.5, 0.2)},
                },
                runtime_provenance={
                    "git_commit": "abc123",
                    "model": {"sha256": "model-hash"},
                    "reference_data": {},
                    "runtime_settings": {"analysis_window_sec": 2.0},
                },
                diagnostics={},
                feedback={
                    "passed": False,
                    "status": "warning",
                    "metrics": {"final_score": 65},
                },
                expected_label="expected_pass",
                output_root=directory,
            )
            metadata = json.loads(
                Path(result["metadata_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["schema"], "golf-coach-runtime-sample-v3")
            reproducibility = metadata["reproducibility"]
            self.assertEqual(
                reproducibility["decision_window"]["sample_count"],
                2,
            )
            self.assertEqual(
                reproducibility["decision_window"]["duration_ms"],
                1500.0,
            )
            self.assertEqual(len(metadata["artifacts"]["decision_frames"]), 2)

            def fake_analyzer(stage, samples, calibration, width, height):
                self.assertEqual(stage, "address")
                self.assertEqual(len(samples), 2)
                self.assertIn(0, calibration["caddieset_address_points"])
                self.assertEqual((width, height), (120, 80))
                return {
                    "passed": True,
                    "status": "pass",
                    "metrics": {"final_score": 75},
                }

            replay = replay_runtime_sample(metadata, analyzer=fake_analyzer)
            self.assertTrue(replay["differences"]["passed_changed"])
            self.assertEqual(
                replay["differences"]["score_deltas"]["final_score"],
                10.0,
            )


if __name__ == "__main__":
    unittest.main()
