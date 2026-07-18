import copy
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

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
from utils.swing_stage_contact_sheet import (
    PANEL_HEIGHT,
    PANEL_WIDTH,
    SHEET_COLUMNS,
    SHEET_ROWS,
    build_stage_contact_sheet,
    candidate_frame_indexes,
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


def write_review_video(path, frame_count=80, fps=20.0):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (160, 120),
    )
    if not writer.isOpened():
        raise RuntimeError("검수 시트 테스트 영상을 만들 수 없습니다.")
    for frame_index in range(frame_count):
        frame = np.full((120, 160, 3), frame_index * 3 % 255, dtype=np.uint8)
        cv2.putText(frame, str(frame_index), (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        writer.write(frame)
    writer.release()


class SwingStageContactSheetTests(unittest.TestCase):
    def test_candidate_indexes_are_clamped(self):
        self.assertEqual(candidate_frame_indexes(2, 100, 5), [0, 2, 7])
        self.assertEqual(candidate_frame_indexes(98, 100, 5), [93, 98, 99])

    def test_contact_sheet_contains_all_eight_stage_panels(self):
        events = {
            stage_key: {"frame_index": 5 + index * 9}
            for index, stage_key in enumerate(STAGE_KEYS)
        }
        ground_truth = make_manifest()["videos"]["sample"]
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "review.avi"
            write_review_video(video_path)
            sheet, metadata = build_stage_contact_sheet(video_path, events, ground_truth)

        self.assertEqual(
            sheet.shape,
            (PANEL_HEIGHT * SHEET_ROWS, PANEL_WIDTH * SHEET_COLUMNS, 3),
        )
        self.assertEqual(tuple(metadata["candidates"]), STAGE_KEYS)
        self.assertGreater(np.count_nonzero(sheet), 0)


if __name__ == "__main__":
    unittest.main()
