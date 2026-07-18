import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from utils.swing_video import LANDMARK_SCHEMA, extract_video_landmarks
from utils.guide_alignment import STAGE_KEYS
from utils.swing_stage_detector import (
    STAGE_SCHEMA,
    detect_stage_events,
    save_representative_frames,
)
from utils.swing_video_evaluator import ANALYSIS_SCHEMA, evaluate_detected_stages
from utils.swing_video_renderer import render_annotated_video, stage_for_frame


def make_landmark(index):
    return SimpleNamespace(
        x=0.1 + index * 0.001,
        y=0.2 + index * 0.001,
        z=-0.01 * index,
        visibility=0.9,
        presence=0.8,
    )


class FakeLandmarker:
    def __init__(self):
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def detect_for_video(self, image, timestamp_ms):
        self.calls += 1
        if self.calls == 2:
            return SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])
        points = [make_landmark(index) for index in range(33)]
        return SimpleNamespace(pose_landmarks=[points], pose_world_landmarks=[points])


def write_test_video(path, frame_count=4, fps=20.0):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (64, 48),
    )
    if not writer.isOpened():
        raise RuntimeError("테스트 영상 writer를 열 수 없습니다.")
    for value in range(frame_count):
        writer.write(np.full((48, 64, 3), value * 30, dtype=np.uint8))
    writer.release()


def make_pose_record(frame_index, wrist_x, wrist_y, shoulder_angle=0.0):
    points = []
    coordinates = {
        11: (0.42, 0.28),
        12: (0.58, 0.28 + math.tan(math.radians(shoulder_angle)) * 0.16),
        15: (wrist_x - 0.02, wrist_y),
        16: (wrist_x + 0.02, wrist_y),
        23: (0.45, 0.52),
        24: (0.55, 0.52),
        27: (0.38, 0.92),
        28: (0.62, 0.92),
    }
    for index in range(33):
        x, y = coordinates.get(index, (0.5, 0.5))
        points.append(
            {
                "index": index,
                "x": x,
                "y": y,
                "z": 0.0,
                "visibility": 0.99,
            }
        )
    return {
        "frame_index": frame_index,
        "timestamp_ms": frame_index * 20,
        "detected": True,
        "landmarks": points,
        "world_landmarks": [],
    }


def make_synthetic_swing_payload(frame_count=81):
    anchors = [
        (0, 0.50, 0.60, 0.0),
        (10, 0.50, 0.60, 0.0),
        (35, 0.30, 0.16, -18.0),
        (58, 0.50, 0.59, 2.0),
        (72, 0.72, 0.24, 16.0),
        (80, 0.72, 0.24, 16.0),
    ]
    frames = []
    for frame_index in range(frame_count):
        for anchor_index in range(len(anchors) - 1):
            start = anchors[anchor_index]
            end = anchors[anchor_index + 1]
            if start[0] <= frame_index <= end[0]:
                ratio = (frame_index - start[0]) / max(end[0] - start[0], 1)
                values = [
                    start[value_index] + (end[value_index] - start[value_index]) * ratio
                    for value_index in range(1, 4)
                ]
                frames.append(make_pose_record(frame_index, *values))
                break
    return {
        "source_video": "synthetic.avi",
        "sampling": {"detection_ratio": 1.0},
        "frames": frames,
    }


class SwingVideoLandmarkTests(unittest.TestCase):
    def test_extracts_timeline_and_keeps_detection_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "swing.avi"
            write_test_video(video_path)
            payload = extract_video_landmarks(
                video_path,
                model_path="unused.task",
                landmarker_factory=lambda _: FakeLandmarker(),
            )

        self.assertEqual(payload["schema"], LANDMARK_SCHEMA)
        self.assertEqual(payload["video"]["decoded_frames"], 4)
        self.assertEqual(payload["sampling"]["sampled_frames"], 4)
        self.assertEqual(payload["sampling"]["detected_frames"], 3)
        self.assertEqual([frame["detected"] for frame in payload["frames"]], [True, False, True, True])
        self.assertEqual(len(payload["frames"][0]["landmarks"]), 33)
        self.assertEqual(payload["frames"][1]["landmarks"], [])
        timestamps = [frame["timestamp_ms"] for frame in payload["frames"]]
        self.assertEqual(timestamps, sorted(set(timestamps)))

    def test_sample_step_retains_original_frame_indexes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "swing.avi"
            write_test_video(video_path, frame_count=5)
            payload = extract_video_landmarks(
                video_path,
                model_path="unused.task",
                sample_step=2,
                landmarker_factory=lambda _: FakeLandmarker(),
            )

        self.assertEqual(
            [frame["frame_index"] for frame in payload["frames"]],
            [0, 2, 4],
        )

    def test_rejects_invalid_sample_step(self):
        with self.assertRaises(ValueError):
            extract_video_landmarks("swing.mp4", "model.task", sample_step=0)


class SwingStageDetectorTests(unittest.TestCase):
    def test_detects_eight_strictly_ordered_stages(self):
        result = detect_stage_events(make_synthetic_swing_payload())

        self.assertEqual(result["schema"], STAGE_SCHEMA)
        self.assertEqual(result["stage_order"], list(STAGE_KEYS))
        frames = [result["stages"][stage]["frame_index"] for stage in STAGE_KEYS]
        self.assertEqual(frames, sorted(set(frames)))
        self.assertTrue(25 <= result["stages"]["top"]["frame_index"] <= 45)
        self.assertTrue(45 <= result["stages"]["impact"]["frame_index"] <= 68)
        self.assertGreaterEqual(result["stages"]["finish"]["frame_index"], 70)
        self.assertLess(result["stages"]["finish"]["frame_index"], 81)
        self.assertGreater(result["stages"]["top"]["motion"]["shoulder_turn"], 5.0)
        self.assertLess(result["stages"]["top"]["motion"]["shoulder_turn"], 90.0)

    def test_saves_one_representative_image_per_stage(self):
        result = detect_stage_events(make_synthetic_swing_payload())
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "swing.avi"
            output_dir = Path(temp_dir) / "stages"
            write_test_video(video_path, frame_count=81, fps=50.0)
            saved = save_representative_frames(video_path, result, output_dir)
            sizes = [Path(path).stat().st_size for path in saved.values()]

        self.assertEqual(tuple(saved), STAGE_KEYS)
        self.assertTrue(all(size > 0 for size in sizes))

    def test_rejects_too_few_valid_pose_frames(self):
        payload = make_synthetic_swing_payload(frame_count=7)
        with self.assertRaises(ValueError):
            detect_stage_events(payload)


class SwingVideoEvaluationTests(unittest.TestCase):
    def test_serializes_stage_coordinates_metrics_and_classification(self):
        payload = make_synthetic_swing_payload()
        detection = detect_stage_events(payload)
        analysis = evaluate_detected_stages(payload, detection)

        self.assertEqual(analysis["schema"], ANALYSIS_SCHEMA)
        self.assertEqual(analysis["stage_order"], list(STAGE_KEYS))
        self.assertEqual(analysis["summary"]["total_count"], 40)
        self.assertEqual(
            analysis["summary"]["pass_count"]
            + analysis["summary"]["warning_count"]
            + analysis["summary"]["unavailable_count"],
            40,
        )
        for stage_key in STAGE_KEYS:
            stage = analysis["stages"][stage_key]
            self.assertEqual(len(stage["landmarks"]), 33)
            self.assertEqual(len(stage["metrics"]), 20)
            self.assertIn(stage["evaluation"]["status"], {"pass", "warning", "unavailable"})
            self.assertTrue(stage["evaluation"]["items"])
            self.assertTrue(stage["evaluation"]["messages"])

    def test_address_is_relative_movement_origin(self):
        payload = make_synthetic_swing_payload()
        detection = detect_stage_events(payload)
        analysis = evaluate_detected_stages(payload, detection)
        address_metrics = analysis["stages"]["address"]["metrics"]

        self.assertAlmostEqual(address_metrics["head_loc"], 0.0)
        self.assertAlmostEqual(address_metrics["hip_shifted"], 0.0)
        self.assertAlmostEqual(address_metrics["hip_rotation"], 0.0)

    def test_missing_selected_frame_is_rejected(self):
        payload = make_synthetic_swing_payload()
        detection = detect_stage_events(payload)
        missing_frame = detection["stages"]["impact"]["frame_index"]
        payload["frames"] = [
            frame for frame in payload["frames"] if frame["frame_index"] != missing_frame
        ]
        with self.assertRaises(ValueError):
            evaluate_detected_stages(payload, detection)


class SwingVideoRenderTests(unittest.TestCase):
    def test_stage_assignment_switches_in_order(self):
        detection = detect_stage_events(make_synthetic_swing_payload())
        assigned = [stage_for_frame(frame, detection) for frame in range(81)]
        stage_positions = [assigned.index(stage_key) for stage_key in STAGE_KEYS]

        self.assertEqual(stage_positions, sorted(stage_positions))

    def test_renders_guide_tolerance_and_user_pose_video(self):
        payload = make_synthetic_swing_payload()
        detection = detect_stage_events(payload)
        analysis = evaluate_detected_stages(payload, detection)
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.avi"
            output_path = Path(temp_dir) / "annotated.avi"
            write_test_video(input_path, frame_count=81, fps=50.0)
            result = render_annotated_video(
                input_path,
                payload,
                detection,
                analysis,
                output_path,
            )
            capture = cv2.VideoCapture(str(output_path))
            success, rendered_frame = capture.read()
            rendered_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()

        self.assertTrue(success)
        self.assertEqual(result["frame_count"], 81)
        self.assertEqual(rendered_count, 81)
        self.assertGreater(np.count_nonzero(rendered_frame), 0)
        self.assertFalse(result["audio_preserved"])


if __name__ == "__main__":
    unittest.main()
