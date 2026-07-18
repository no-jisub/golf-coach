import copy
import unittest
from pathlib import Path

from utils.guide_alignment import STAGE_KEYS
from utils.swing_stage_audit import (
    GROUND_TRUTH_SCHEMA,
    load_ground_truth_manifest,
    update_manifest_summary,
    validate_ground_truth_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "reference_data" / "swing_stage_ground_truth.json"


def make_manifest(status="pending"):
    events = {
        stage_key: index * 10 if status == "reviewed" else None
        for index, stage_key in enumerate(STAGE_KEYS)
    }
    manifest = {
        "schema": GROUND_TRUTH_SCHEMA,
        "view": "FACEON",
        "stage_order": list(STAGE_KEYS),
        "videos": {
            "sample": {
                "source": "sample.mp4",
                "review_status": status,
                "events": events,
                "note": "제외" if status == "excluded" else "",
            }
        },
    }
    update_manifest_summary(manifest)
    return manifest


class SwingStageGroundTruthTests(unittest.TestCase):
    def test_repository_manifest_lists_existing_pending_videos(self):
        manifest = load_ground_truth_manifest(
            MANIFEST_PATH,
            project_root=PROJECT_ROOT,
            require_files=True,
        )

        self.assertEqual(manifest["summary"]["video_count"], 8)
        self.assertEqual(manifest["summary"]["pending_count"], 8)
        self.assertEqual(manifest["summary"]["reviewed_count"], 0)

    def test_reviewed_events_must_be_complete_and_strictly_ordered(self):
        manifest = make_manifest("reviewed")
        validate_ground_truth_manifest(manifest)

        incomplete = copy.deepcopy(manifest)
        incomplete["videos"]["sample"]["events"]["impact"] = None
        with self.assertRaises(ValueError):
            validate_ground_truth_manifest(incomplete)

        unordered = copy.deepcopy(manifest)
        unordered["videos"]["sample"]["events"]["impact"] = 20
        with self.assertRaises(ValueError):
            validate_ground_truth_manifest(unordered)

    def test_excluded_video_requires_note(self):
        manifest = make_manifest("excluded")
        validate_ground_truth_manifest(manifest)
        manifest["videos"]["sample"]["note"] = ""
        with self.assertRaises(ValueError):
            validate_ground_truth_manifest(manifest)

    def test_summary_mismatch_is_rejected(self):
        manifest = make_manifest()
        manifest["summary"]["pending_count"] = 0
        with self.assertRaises(ValueError):
            validate_ground_truth_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
