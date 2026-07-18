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
from utils.swing_stage_accuracy import ACCURACY_SCHEMA, build_stage_accuracy_report


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
        self.assertEqual(manifest["summary"]["pending_count"], 7)
        self.assertEqual(manifest["summary"]["reviewed_count"], 1)
        self.assertEqual(manifest["videos"]["pro03"]["sequence_mode"], "cyclic")

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

    def test_cyclic_video_allows_one_frame_boundary_wrap(self):
        manifest = make_manifest("reviewed")
        video = manifest["videos"]["sample"]
        video["sequence_mode"] = "cyclic"
        video["events"] = {
            stage_key: frame
            for stage_key, frame in zip(STAGE_KEYS, [70, 0, 10, 20, 30, 40, 50, 60])
        }
        validate_ground_truth_manifest(manifest)

        video["sequence_mode"] = "linear"
        with self.assertRaises(ValueError):
            validate_ground_truth_manifest(manifest)

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


def make_audit(automatic_frames, *, fps=20.0, total_frames=100, status="ok"):
    video = {
        "status": status,
        "source": "sample.mp4",
        "ground_truth_status": "reviewed",
    }
    if status == "ok":
        video.update(
            {
                "video": {"fps": fps, "total_frames": total_frames},
                "stage_detection": {
                    "events": {
                        stage_key: {"frame_index": automatic_frames[index]}
                        for index, stage_key in enumerate(STAGE_KEYS)
                    }
                },
            }
        )
    return {"videos": {"sample": video}}


class SwingStageAccuracyTests(unittest.TestCase):
    def test_reviewed_events_produce_frame_and_time_error_metrics(self):
        manifest = make_manifest("reviewed")
        automatic_frames = [0, 11, 18, 30, 44, 50, 58, 70]

        report = build_stage_accuracy_report(
            make_audit(automatic_frames),
            manifest,
            tolerance_ms=100,
        )

        self.assertEqual(report["schema"], ACCURACY_SCHEMA)
        self.assertEqual(report["summary"]["reviewed_count"], 1)
        self.assertEqual(report["summary"]["evaluated_count"], 8)
        self.assertEqual(report["summary"]["within_tolerance_count"], 7)
        self.assertEqual(report["summary"]["within_tolerance_rate"], 0.875)
        backswing = report["videos"]["sample"]["comparisons"]["backswing"]
        self.assertEqual(backswing["signed_error_frames"], -2)
        self.assertEqual(backswing["absolute_error_ms"], 100.0)

    def test_pending_events_never_count_as_accuracy_ground_truth(self):
        manifest = make_manifest("pending")
        report = build_stage_accuracy_report(
            make_audit([index * 10 for index in range(8)]),
            manifest,
        )

        self.assertEqual(report["summary"]["reviewed_count"], 0)
        self.assertEqual(report["summary"]["pending_review_count"], 1)
        self.assertEqual(report["summary"]["evaluated_count"], 0)
        self.assertIsNone(report["summary"]["within_tolerance_rate"])
        self.assertEqual(report["videos"]["sample"]["status"], "pending_review")

    def test_cyclic_video_uses_shortest_distance_across_frame_boundary(self):
        manifest = make_manifest("reviewed")
        manifest["videos"]["sample"]["sequence_mode"] = "cyclic"
        manifest["videos"]["sample"]["events"]["address"] = 90
        automatic_frames = [0, 10, 20, 30, 40, 50, 60, 70]

        report = build_stage_accuracy_report(
            make_audit(automatic_frames, fps=20.0, total_frames=100),
            manifest,
        )

        address = report["videos"]["sample"]["comparisons"]["address"]
        self.assertEqual(address["raw_signed_error_frames"], -90)
        self.assertEqual(address["signed_error_frames"], 10)
        self.assertEqual(address["absolute_error_ms"], 500.0)
        self.assertTrue(address["crosses_frame_boundary"])


if __name__ == "__main__":
    unittest.main()
