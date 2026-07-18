import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from utils.loop_video_normalizer import (
    MAPPING_SCHEMA,
    frame_from_seconds,
    map_frame_index,
    map_ground_truth_events,
    normalize_loop_video,
)


def write_test_video(path, values, fps=20.0, size=(64, 48)):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create test video")
    for value in values:
        writer.write(np.full((size[1], size[0], 3), value, dtype=np.uint8))
    writer.release()


def read_video(path):
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    values = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        values.append(float(frame.mean()))
    capture.release()
    return fps, size, values


class FrameMappingTests(unittest.TestCase):
    def test_maps_frames_across_video_boundary(self):
        self.assertEqual(map_frame_index(243, 260, 243), 0)
        self.assertEqual(map_frame_index(259, 260, 243), 16)
        self.assertEqual(map_frame_index(0, 260, 243), 17)
        self.assertEqual(map_frame_index(241, 260, 243), 258)

    def test_maps_ground_truth_without_mutating_source(self):
        source = {"address": 243, "takeaway": 27, "pending": None}
        mapped = map_ground_truth_events(source, total_frames=260, split_frame=243)

        self.assertEqual(mapped, {"address": 0, "takeaway": 44, "pending": None})
        self.assertEqual(source["address"], 243)

    def test_rejects_out_of_range_frames(self):
        with self.assertRaises(ValueError):
            map_frame_index(260, total_frames=260, split_frame=243)
        with self.assertRaises(ValueError):
            map_frame_index(0, total_frames=260, split_frame=260)

    def test_converts_seconds_to_nearest_frame(self):
        self.assertEqual(frame_from_seconds(8.11, 30.0), 243)


class NormalizeLoopVideoTests(unittest.TestCase):
    def test_rotates_frames_and_preserves_video_properties(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.avi"
            output_path = Path(temp_dir) / "output.avi"
            write_test_video(input_path, [0, 40, 80, 120, 160], fps=20.0)

            result = normalize_loop_video(input_path, output_path, split_frame=2)
            fps, size, values = read_video(output_path)

        self.assertEqual(result["schema"], MAPPING_SCHEMA)
        self.assertEqual(result["video"]["total_frames"], 5)
        self.assertAlmostEqual(fps, 20.0, places=2)
        self.assertEqual(size, (64, 48))
        self.assertEqual(len(values), 5)
        for actual, expected in zip(values, [80, 120, 160, 0, 40]):
            self.assertAlmostEqual(actual, expected, delta=4.0)
        self.assertEqual(
            result["frame_mapping"],
            {"0": 3, "1": 4, "2": 0, "3": 1, "4": 2},
        )

    def test_never_overwrites_source_or_existing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.avi"
            output_path = Path(temp_dir) / "output.avi"
            write_test_video(input_path, [0, 40])
            write_test_video(output_path, [80])

            with self.assertRaises(ValueError):
                normalize_loop_video(input_path, input_path, split_frame=0)
            with self.assertRaises(FileExistsError):
                normalize_loop_video(input_path, output_path, split_frame=0)


if __name__ == "__main__":
    unittest.main()
