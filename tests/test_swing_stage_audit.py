import copy
import tempfile
import unittest
from pathlib import Path

from utils.guide_alignment import STAGE_KEYS
from utils.swing_stage_audit import (
    AUDIT_SCHEMA,
    GROUND_TRUTH_SCHEMA,
    audit_manifest_videos,
    load_ground_truth_manifest,
    update_manifest_summary,
    validate_ground_truth_manifest,
)
from utils.swing_video import LANDMARK_SCHEMA


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


class SwingStageBatchAuditTests(unittest.TestCase):
    def test_batch_audit_creates_and_reuses_landmark_cache(self):
        manifest = make_manifest()
        calls = []

        def fake_extractor(video_path, model_path, **kwargs):
            calls.append(str(video_path))
            return {
                "schema": LANDMARK_SCHEMA,
                "source_video": str(video_path),
                "video": {"fps": 30.0, "total_frames": 100},
                "sampling": {"detection_ratio": 1.0},
                "frames": [],
            }

        def fake_detector(payload):
            return {
                "method": "fake",
                "diagnostics": {"usable_pose_frames": 100},
                "stages": {
                    stage_key: {
                        "frame_index": index * 10,
                        "timestamp_ms": index * 333,
                        "confidence": 1.0,
                    }
                    for index, stage_key in enumerate(STAGE_KEYS)
                },
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "sample.mp4").touch()
            first = audit_manifest_videos(
                manifest,
                project_root=root,
                output_root=root / "audit",
                model_path=root / "model.task",
                landmark_extractor=fake_extractor,
                stage_detector=fake_detector,
            )
            second = audit_manifest_videos(
                manifest,
                project_root=root,
                output_root=root / "audit",
                model_path=root / "model.task",
                landmark_extractor=fake_extractor,
                stage_detector=fake_detector,
            )

        self.assertEqual(first["schema"], AUDIT_SCHEMA)
        self.assertEqual(first["summary"]["created_cache_count"], 1)
        self.assertEqual(second["summary"]["reused_cache_count"], 1)
        self.assertEqual(len(calls), 1)

    def test_batch_audit_rejects_unknown_video_filter(self):
        manifest = make_manifest()
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                audit_manifest_videos(
                    manifest,
                    project_root=temp_dir,
                    output_root=Path(temp_dir) / "audit",
                    model_path=Path(temp_dir) / "model.task",
                    video_ids=["unknown"],
                )


if __name__ == "__main__":
    unittest.main()
