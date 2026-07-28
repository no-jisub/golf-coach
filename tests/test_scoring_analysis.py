import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from utils.guide_alignment import STAGE_KEYS
from utils.scoring_analysis import (
    analyze_regression_bottlenecks,
    analyze_thresholds,
    confusion_metrics,
)
from utils.stage_candidate_extractor import extract_stage_candidates


def make_regression_report():
    stages = {}
    for stage_index, stage_key in enumerate(STAGE_KEYS):
        stages[stage_key] = {
            "passed": stage_index == 0,
            "final_score": 80 - stage_index * 5,
            "guide_score": 60 - stage_index * 5,
            "caddieset_score": 90,
            "guide_group_distances": {
                "head": 0.1,
                "arms": 0.4 + stage_index * 0.1,
                "body": 0.2,
                "lower": 0.3,
            },
            "joint_distances": {"15": 0.5 + stage_index * 0.1, "27": 0.2},
            "caddieset_items": {
                "left_arm_angle": {
                    "status": "warning" if stage_index else "pass",
                    "relation": "above_reference",
                }
            },
        }
    return {"videos": {"video01": {"status": "ok", "stages": stages}}}


class ScoringAnalysisTests(unittest.TestCase):
    def test_confusion_metrics_reports_false_accept_and_false_reject(self):
        records = [
            {"label": "good", "model": {"final_score": 80}},
            {"label": "good", "model": {"final_score": 60}},
            {"label": "bad", "model": {"final_score": 75}},
            {"label": "bad", "model": {"final_score": 40}},
        ]
        result = confusion_metrics(records, 70)
        self.assertEqual(
            (result["tp"], result["fp"], result["tn"], result["fn"]),
            (1, 1, 1, 1),
        )
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)

    def test_threshold_analysis_is_split_by_stage_and_body_profile(self):
        ground_truth = {
            "records": [
                {
                    "stage_key": "address",
                    "label": "good",
                    "model": {"final_score": 80},
                    "body_profile": {
                        "height_band": "170_179",
                        "body_build": "average",
                    },
                },
                {
                    "stage_key": "address",
                    "label": "bad",
                    "model": {"final_score": 50},
                    "body_profile": {
                        "height_band": "180_189",
                        "body_build": "athletic",
                    },
                },
            ]
        }
        result = analyze_thresholds(ground_truth, thresholds=[60, 70])
        self.assertEqual(result["best_f1_threshold"]["f1"], 1.0)
        self.assertEqual(result["by_stage"]["address"][1]["sample_count"], 2)
        self.assertIn("170_179", result["subgroups_at_70"]["height_band"])

    def test_bottleneck_analysis_ranks_low_stage_and_worst_joint(self):
        result = analyze_regression_bottlenecks(make_regression_report())
        self.assertEqual(result["stage_order_by_bottleneck"][0], "finish")
        finish = result["stages"]["finish"]
        self.assertEqual(finish["primary_component"], "guide")
        self.assertEqual(
            finish["top_joint_bottlenecks"][0]["label"],
            "left_wrist",
        )


class StageCandidateExtractorTests(unittest.TestCase):
    def test_extracts_ranked_candidates_for_all_stages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "swing.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                20.0,
                (64, 48),
            )
            for value in range(40):
                writer.write(np.full((48, 64, 3), value * 5, dtype=np.uint8))
            writer.release()
            events = {
                "stages": {
                    stage_key: {"frame_index": 2 + index * 5}
                    for index, stage_key in enumerate(STAGE_KEYS)
                }
            }
            output_dir = Path(temp_dir) / "candidates"
            report = extract_stage_candidates(
                video_path,
                events,
                output_dir,
                radius_frames=2,
                candidate_count=3,
            )
            self.assertEqual(set(report["stages"]), set(STAGE_KEYS))
            self.assertEqual(
                len(report["stages"]["impact"]["candidates"]),
                3,
            )
            self.assertTrue((output_dir / "stage_candidates.jpg").exists())
            self.assertTrue((output_dir / "candidate_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
