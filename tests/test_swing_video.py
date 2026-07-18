import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from utils.swing_video import LANDMARK_SCHEMA, extract_video_landmarks


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


if __name__ == "__main__":
    unittest.main()
