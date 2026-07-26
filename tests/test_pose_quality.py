import unittest
from types import SimpleNamespace

from utils.guide_skeleton import GUIDE_POSES, LEFT_WRIST, RIGHT_WRIST
from utils.pose_quality import (
    check_address_similarity,
    check_full_body_visibility,
    evaluate_calibration_frame,
)


def make_landmarks(points, *, visibility=1.0):
    landmarks = [
        SimpleNamespace(x=0.5, y=0.5, visibility=visibility)
        for _ in range(33)
    ]
    for index, point in points.items():
        landmarks[index] = SimpleNamespace(
            x=float(point[0]),
            y=float(point[1]),
            visibility=visibility,
        )
    return landmarks


class PoseInputQualityTests(unittest.TestCase):
    def test_aligned_address_is_ready_for_calibration(self):
        landmarks = make_landmarks(GUIDE_POSES["address"])
        result = evaluate_calibration_frame(landmarks)
        self.assertTrue(result["passed"])
        self.assertEqual(result["reason"], "ready")

    def test_hidden_ankle_blocks_full_body_check(self):
        landmarks = make_landmarks(GUIDE_POSES["address"])
        landmarks[28].visibility = 0.1
        result = check_full_body_visibility(landmarks)
        self.assertFalse(result["passed"])
        self.assertIn("오른쪽 발목", result["missing"])

    def test_clipped_head_blocks_full_body_check(self):
        landmarks = make_landmarks(GUIDE_POSES["address"])
        landmarks[0].y = 0.0
        result = check_full_body_visibility(landmarks)
        self.assertFalse(result["passed"])
        self.assertIn("머리", result["clipped"])

    def test_non_address_arm_position_is_rejected(self):
        points = dict(GUIDE_POSES["address"])
        points[LEFT_WRIST] = (0.15, 0.78)
        points[RIGHT_WRIST] = (0.85, 0.78)
        result = check_address_similarity(make_landmarks(points))
        self.assertFalse(result["passed"])
        self.assertTrue(result["messages"])


if __name__ == "__main__":
    unittest.main()
