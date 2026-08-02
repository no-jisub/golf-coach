import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from utils.guide_alignment import STAGE_KEYS
from utils.regression_comparison import build_stage_comparison_sheet
from utils.stage_candidate_review import (
    finalize_stage_candidate_review,
    finalize_stage_frame_review,
)
from utils.stage_video_review import (
    FrameNavigator,
    build_review_progress,
    load_review_progress,
    save_review_progress,
)


def make_ground_truth():
    return {
        "schema": "golf-coach-swing-stage-ground-truth-v1",
        "view": "FACEON",
        "stage_order": list(STAGE_KEYS),
        "videos": {
            "video01": {
                "source": "reference_data/raw_videos/video01.mp4",
                "review_status": "pending",
                "sequence_mode": "linear",
                "events": {stage: None for stage in STAGE_KEYS},
                "reviewed_by": None,
                "reviewed_at": None,
                "note": "",
            }
        },
        "summary": {
            "video_count": 1,
            "excluded_count": 0,
            "pending_count": 1,
            "reviewed_count": 0,
        },
    }


def make_candidate_manifest():
    return {
        "schema": "golf-coach-stage-candidates-v1",
        "source_video": "video01.mp4",
        "stages": {
            stage: {
                "candidates": [
                    {"frame_index": index * 5 + offset}
                    for offset in (1, 2, 3)
                ]
            }
            for index, stage in enumerate(STAGE_KEYS)
        },
    }


class StageCandidateReviewTests(unittest.TestCase):
    def test_finalizes_complete_candidate_selection_without_mutating_source(self):
        manifest = make_ground_truth()
        candidates = make_candidate_manifest()
        selections = {
            stage: index * 5 + 2
            for index, stage in enumerate(STAGE_KEYS)
        }
        updated = finalize_stage_candidate_review(
            manifest,
            video_id="video01",
            candidate_manifest=candidates,
            selections=selections,
            reviewed_by="coach-a",
            reviewed_at=datetime(2026, 7, 29, tzinfo=timezone.utc),
        )
        self.assertEqual(manifest["videos"]["video01"]["review_status"], "pending")
        self.assertEqual(updated["videos"]["video01"]["events"], selections)
        self.assertEqual(updated["summary"]["reviewed_count"], 1)
        self.assertEqual(updated["summary"]["pending_count"], 0)

    def test_rejects_frame_outside_candidate_list(self):
        selections = {
            stage: index * 5 + 2
            for index, stage in enumerate(STAGE_KEYS)
        }
        selections["impact"] = 999
        with self.assertRaises(ValueError):
            finalize_stage_candidate_review(
                make_ground_truth(),
                video_id="video01",
                candidate_manifest=make_candidate_manifest(),
                selections=selections,
                reviewed_by="coach-a",
            )

    def test_full_timeline_review_accepts_frames_outside_candidates(self):
        selections = {
            stage: index * 5 + 4
            for index, stage in enumerate(STAGE_KEYS)
        }
        updated = finalize_stage_frame_review(
            make_ground_truth(),
            video_id="video01",
            selections=selections,
            reviewed_by="coach-a",
            total_frames=50,
        )
        self.assertEqual(updated["videos"]["video01"]["events"], selections)
        self.assertEqual(
            updated["videos"]["video01"]["review_source"]["type"],
            "full_timeline_frame_selection",
        )


class StageVideoReviewTests(unittest.TestCase):
    def test_navigator_clamps_steps_and_calculates_timestamp(self):
        navigator = FrameNavigator(100, fps=20.0, initial_frame=98)
        self.assertEqual(navigator.step(10), 99)
        self.assertEqual(navigator.step(-200), 0)
        navigator.seek(50)
        self.assertEqual(navigator.timestamp_ms, 2500.0)

    def test_review_progress_round_trip_preserves_partial_selection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "progress.json"
            progress = build_review_progress(
                video_id="video01",
                source_video="video.avi",
                selections={"address": 3, "top": 21},
                current_stage_index=4,
                current_frame=24,
            )
            save_review_progress(path, progress)
            loaded = load_review_progress(path, video_id="video01")
            self.assertEqual(loaded["selections"], {"address": 3, "top": 21})
            self.assertEqual(loaded["current_frame"], 24)


class RegressionComparisonTests(unittest.TestCase):
    def test_renders_automatic_and_reviewed_frames_for_all_stages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "swing.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                20.0,
                (64, 48),
            )
            for value in range(50):
                writer.write(np.full((48, 64, 3), value * 4, dtype=np.uint8))
            writer.release()
            reviewed = {
                stage: 2 + index * 5
                for index, stage in enumerate(STAGE_KEYS)
            }
            automatic = {
                "stages": {
                    stage: {"frame_index": reviewed[stage] + 1}
                    for stage in STAGE_KEYS
                }
            }
            output_path = Path(temp_dir) / "comparison.jpg"
            result = build_stage_comparison_sheet(
                video_path,
                reviewed,
                automatic,
                output_path,
            )
            image = cv2.imread(str(output_path))
            self.assertIsNotNone(image)
            self.assertEqual(set(result["comparisons"]), set(STAGE_KEYS))
            self.assertTrue(
                all(
                    item["signed_error_frames"] == 1
                    for item in result["comparisons"].values()
                )
            )


if __name__ == "__main__":
    unittest.main()
