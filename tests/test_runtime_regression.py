import json
import tempfile
import unittest
from pathlib import Path

from utils.guide_alignment import STAGE_KEYS
from utils.guide_skeleton import GUIDE_POSES
from utils.runtime_regression import (
    build_reviewed_runtime_regression,
    build_runtime_regression,
    diagnose_stage,
    render_runtime_regression_markdown,
    select_stage_frames,
)


def serialized_landmarks(points):
    records = []
    for index in range(33):
        point = points.get(index, (0.0, 0.0))
        records.append(
            {
                "index": index,
                "x": point[0],
                "y": point[1],
                "z": 0.0,
                "visibility": 1.0 if index in points else 0.0,
                "presence": 1.0 if index in points else 0.0,
            }
        )
    return records


def make_landmark_payload():
    return {
        "schema": "golf-coach-swing-video-landmarks-v1",
        "source_video": "test.mp4",
        "video": {
            "fps": 30.0,
            "width": 1000,
            "height": 1000,
            "total_frames": 80,
        },
        "frames": [
            {
                "frame_index": order * 10,
                "timestamp_ms": order * 333,
                "detected": True,
                "landmarks": serialized_landmarks(GUIDE_POSES[stage_key]),
            }
            for order, stage_key in enumerate(STAGE_KEYS)
        ],
    }


def make_events():
    return {
        "schema": "golf-coach-swing-stage-events-v1",
        "stages": {
            stage_key: {"frame_index": order * 10}
            for order, stage_key in enumerate(STAGE_KEYS)
        },
    }


class RuntimeRegressionTests(unittest.TestCase):
    def test_reviewed_frames_take_priority_over_automatic_events(self):
        manifest_video = {
            "review_status": "reviewed",
            "events": {
                stage_key: order * 10 + 1
                for order, stage_key in enumerate(STAGE_KEYS)
            },
        }
        selected, source = select_stage_frames(manifest_video, make_events())
        self.assertEqual(source, "reviewed")
        self.assertEqual(selected["impact"], 51)

    def test_existing_cache_runs_through_all_eight_runtime_stages(self):
        manifest = {
            "view": "FACEON",
            "videos": {
                "pro_test": {
                    "source": "test.mp4",
                    "review_status": "pending",
                    "events": {stage_key: None for stage_key in STAGE_KEYS},
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = Path(temporary_directory) / "pro_test"
            session.mkdir()
            (session / "frame_landmarks.json").write_text(
                json.dumps(make_landmark_payload()),
                encoding="utf-8",
            )
            (session / "stage_events.json").write_text(
                json.dumps(make_events()),
                encoding="utf-8",
            )

            report = build_runtime_regression(manifest, temporary_directory)

        video = report["videos"]["pro_test"]
        self.assertEqual(video["status"], "ok")
        self.assertEqual(video["event_source"], "automatic")
        self.assertEqual(tuple(video["stages"]), STAGE_KEYS)
        self.assertTrue(all(stage["passed"] for stage in video["stages"].values()))
        self.assertEqual(report["summary"]["evaluated_video_count"], 1)
        self.assertEqual(report["summary"]["automatic_event_video_count"], 1)

    def test_report_marks_strict_and_lenient_candidates(self):
        self.assertEqual(diagnose_stage(60.0, 0.9), "strict_candidate")
        self.assertEqual(diagnose_stage(95.0, 1.0), "lenient_candidate")
        self.assertEqual(diagnose_stage(80.0, 0.75), "balanced_candidate")

        report = {
            "summary": {
                "video_count": 1,
                "evaluated_video_count": 1,
                "failed_video_count": 0,
                "reviewed_event_video_count": 0,
                "automatic_event_video_count": 1,
                "stages": {
                    stage_key: {
                        "label": stage_key,
                        "sample_count": 1,
                        "mean_final_score": 80.0,
                        "mean_guide_score": 80.0,
                        "mean_caddieset_score": 80.0,
                        "pass_rate": 1.0,
                        "diagnosis": "balanced_candidate",
                    }
                    for stage_key in STAGE_KEYS
                },
            },
            "videos": {},
        }
        markdown = render_runtime_regression_markdown(report)
        self.assertIn("Stage Diagnostics", markdown)
        self.assertIn("balanced_candidate", markdown)

    def test_reviewed_only_report_excludes_automatic_videos(self):
        manifest = {
            "view": "FACEON",
            "videos": {
                "reviewed_video": {
                    "source": "reviewed.mp4",
                    "review_status": "reviewed",
                    "events": {
                        stage_key: order * 10
                        for order, stage_key in enumerate(STAGE_KEYS)
                    },
                },
                "pending_video": {
                    "source": "pending.mp4",
                    "review_status": "pending",
                    "events": {stage_key: None for stage_key in STAGE_KEYS},
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = Path(temporary_directory) / "reviewed_video"
            session.mkdir()
            (session / "frame_landmarks.json").write_text(
                json.dumps(make_landmark_payload()),
                encoding="utf-8",
            )
            (session / "stage_events.json").write_text(
                json.dumps(make_events()),
                encoding="utf-8",
            )
            report = build_reviewed_runtime_regression(
                manifest,
                temporary_directory,
            )

        self.assertEqual(set(report["videos"]), {"reviewed_video"})
        self.assertEqual(report["scope"]["event_scope"], "reviewed_only")
        self.assertEqual(report["summary"]["automatic_event_video_count"], 0)
        self.assertFalse(report["dataset_quality"]["criterion_tuning_allowed"])
        self.assertIn("최소 5개", report["dataset_quality"]["warnings"][0])
        markdown = render_runtime_regression_markdown(report)
        self.assertIn("Reviewed-only", markdown)
        self.assertIn("기준 조정 허용: no", markdown)

    def test_reviewed_only_report_handles_zero_reviewed_videos(self):
        manifest = {
            "view": "FACEON",
            "videos": {
                "pending_video": {
                    "source": "pending.mp4",
                    "review_status": "pending",
                    "events": {stage_key: None for stage_key in STAGE_KEYS},
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = build_reviewed_runtime_regression(
                manifest,
                temporary_directory,
            )
        self.assertEqual(report["summary"]["video_count"], 0)
        self.assertEqual(report["videos"], {})
        self.assertIn("검수 완료 영상이 없어", report["dataset_quality"]["warnings"][0])


if __name__ == "__main__":
    unittest.main()
