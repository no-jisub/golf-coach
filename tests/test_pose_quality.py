import unittest
from types import SimpleNamespace

from utils.guide_skeleton import GUIDE_POSES, LEFT_WRIST, RIGHT_WRIST
from utils.pose_quality import (
    check_address_similarity,
    check_full_body_visibility,
    evaluate_calibration_frame,
    evaluate_pose_stability,
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


class PoseStabilityTests(unittest.TestCase):
    def test_requires_real_elapsed_time(self):
        landmarks = make_landmarks(GUIDE_POSES["address"])
        samples = [(index * 0.05, landmarks) for index in range(20)]
        result = evaluate_pose_stability(samples, min_duration_sec=1.5)
        self.assertFalse(result["ready"])
        self.assertLess(result["duration_sec"], 1.5)

    def test_stable_pose_passes_after_hold_duration(self):
        samples = []
        for index in range(31):
            offset = 0.001 if index % 2 else -0.001
            points = {
                joint: (point[0] + offset, point[1])
                for joint, point in GUIDE_POSES["address"].items()
            }
            samples.append((index * 0.05, make_landmarks(points)))

        result = evaluate_pose_stability(samples, min_duration_sec=1.5)
        self.assertTrue(result["ready"])
        self.assertTrue(result["stable"])

    def test_moving_wrist_resets_stability(self):
        samples = []
        for index in range(31):
            points = dict(GUIDE_POSES["address"])
            points[LEFT_WRIST] = (
                points[LEFT_WRIST][0] + index * 0.004,
                points[LEFT_WRIST][1],
            )
            samples.append((index * 0.05, make_landmarks(points)))

        result = evaluate_pose_stability(samples, min_duration_sec=1.5)
        self.assertTrue(result["ready"])
        self.assertFalse(result["stable"])
        self.assertEqual(result["max_joint_index"], LEFT_WRIST)


if __name__ == "__main__":
    unittest.main()
