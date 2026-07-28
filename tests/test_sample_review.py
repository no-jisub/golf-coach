import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from utils.sample_review import (
    apply_coach_review,
    build_review_manifest,
    export_coach_ground_truth,
    preliminary_label,
    review_priority,
)


def write_sample(root, sample_id, *, expected_label, actual_passed, score, discrepancy):
    sample_dir = Path(root) / "address" / sample_id
    sample_dir.mkdir(parents=True)
    payload = {
        "schema": "golf-coach-runtime-sample-v2",
        "sample_id": sample_id,
        "stage_key": "address",
        "expected_label": expected_label,
        "actual_passed": actual_passed,
        "discrepancy": discrepancy,
        "landmarks": [{"index": 0, "x": 0.5, "y": 0.2}],
        "feedback": {
            "status": "pass" if actual_passed else "warning",
            "messages": [],
            "metrics": {
                "final_score": score,
                "guide_score": score,
                "caddieset_score": score,
            },
        },
        "collection": {
            "participant_id": "p_test01",
            "session_id": "s_test01",
            "body_profile": {"height_band": "170_179"},
            "capture_conditions": {"view": "FACEON", "club_type": "I7"},
        },
        "artifacts": {"raw_frame": "raw.jpg", "overlay_frame": "overlay.jpg"},
    }
    (sample_dir / "sample.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return payload


class SampleReviewTests(unittest.TestCase):
    def test_collector_label_has_priority_over_model_suggestion(self):
        sample = {
            "expected_label": "expected_pass",
            "actual_passed": False,
        }
        result = preliminary_label(sample)
        self.assertEqual(result["label"], "good")
        self.assertEqual(result["source"], "collector")

    def test_disagreement_and_threshold_margin_raise_priority(self):
        ambiguous = {
            "expected_label": "expected_pass",
            "actual_passed": False,
            "discrepancy": "false_reject",
            "feedback": {"status": "warning", "metrics": {"final_score": 69}},
        }
        clear = {
            "expected_label": "expected_pass",
            "actual_passed": True,
            "discrepancy": "match",
            "feedback": {"status": "pass", "metrics": {"final_score": 95}},
        }
        self.assertGreater(
            review_priority(ambiguous)["score"],
            review_priority(clear)["score"],
        )

    def test_manifest_preserves_review_and_exports_ground_truth(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            write_sample(
                temp_dir,
                "sample001",
                expected_label="expected_pass",
                actual_passed=False,
                score=65,
                discrepancy="false_reject",
            )
            manifest = build_review_manifest([temp_dir])
            reviewed_at = datetime(2026, 7, 29, tzinfo=timezone.utc)
            apply_coach_review(
                manifest,
                "sample001",
                "good",
                reviewed_by="coach-a",
                reviewed_at=reviewed_at,
            )
            refreshed = build_review_manifest([temp_dir], manifest)
            exported = export_coach_ground_truth(refreshed)

            self.assertEqual(
                refreshed["samples"]["sample001"]["coach_review"]["label"],
                "good",
            )
            self.assertEqual(exported["summary"]["good_count"], 1)
            record = exported["records"][0]
            self.assertEqual(record["participant_id"], "p_test01")
            self.assertEqual(record["model"]["final_score"], 65)
            self.assertNotIn("private_profile", record)


if __name__ == "__main__":
    unittest.main()
