import math
import unittest
from types import SimpleNamespace

from utils.caddieset_metrics import (
    average_landmark_points,
    calculate_pose_metrics,
    joint_angle,
)
from utils.guide_skeleton import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)


def make_points():
    return {
        NOSE: (0.50, 0.10),
        LEFT_SHOULDER: (0.40, 0.25),
        RIGHT_SHOULDER: (0.60, 0.25),
        LEFT_ELBOW: (0.40, 0.40),
        RIGHT_ELBOW: (0.60, 0.40),
        LEFT_WRIST: (0.40, 0.55),
        RIGHT_WRIST: (0.60, 0.55),
        LEFT_HIP: (0.44, 0.50),
        RIGHT_HIP: (0.56, 0.50),
        LEFT_KNEE: (0.42, 0.70),
        RIGHT_KNEE: (0.58, 0.70),
        LEFT_ANKLE: (0.35, 0.90),
        RIGHT_ANKLE: (0.65, 0.90),
    }


class CaddieSetMetricTests(unittest.TestCase):
    def test_joint_angle_returns_straight_arm(self):
        self.assertAlmostEqual(joint_angle((0, 0), (0, 1), (0, 2)), 180.0)
        self.assertIsNone(joint_angle((0, 0), (0, 0), (0, 2)))

    def test_calculates_address_geometry(self):
        points = make_points()
        metrics = calculate_pose_metrics(points, points)

        self.assertAlmostEqual(metrics["shoulder_angle"], 0.0)
        self.assertAlmostEqual(metrics["stance_ratio"], 1.5)
        self.assertAlmostEqual(metrics["left_arm_angle"], 180.0)
        self.assertAlmostEqual(metrics["right_arm_angle"], 180.0)
        self.assertAlmostEqual(metrics["head_loc"], 0.0)
        self.assertAlmostEqual(metrics["hip_shifted"], 0.0)
        self.assertTrue(math.isfinite(metrics["upper_tilt"]))

    def test_relative_metrics_use_address_and_direction(self):
        address = make_points()
        current = make_points()
        current[NOSE] = (0.53, current[NOSE][1])
        current[LEFT_HIP] = (0.47, current[LEFT_HIP][1])
        current[RIGHT_HIP] = (0.59, current[RIGHT_HIP][1])

        metrics = calculate_pose_metrics(current, address)
        mirrored = calculate_pose_metrics(current, address, direction_multiplier=-1.0)

        self.assertAlmostEqual(metrics["head_loc"], 0.1)
        self.assertAlmostEqual(metrics["hip_shifted"], 0.1)
        self.assertAlmostEqual(mirrored["head_loc"], -0.1)
        self.assertAlmostEqual(mirrored["hip_shifted"], -0.1)
        self.assertAlmostEqual(
            mirrored["shoulder_loc"],
            metrics["shoulder_loc"],
        )
        self.assertAlmostEqual(
            mirrored["finish_angle"],
            180.0 - metrics["finish_angle"],
        )

    def test_relative_metrics_are_unavailable_without_address(self):
        metrics = calculate_pose_metrics(make_points())
        self.assertIsNone(metrics["head_loc"])
        self.assertIsNone(metrics["hip_shifted"])
        self.assertIsNone(metrics["hip_rotation"])

    def test_average_landmark_points_ignores_low_visibility(self):
        frame_a = [SimpleNamespace(x=0.0, y=0.0, visibility=0.0) for _ in range(33)]
        frame_b = [SimpleNamespace(x=0.0, y=0.0, visibility=0.0) for _ in range(33)]
        frame_a[NOSE] = SimpleNamespace(x=0.4, y=0.2, visibility=1.0)
        frame_b[NOSE] = SimpleNamespace(x=0.6, y=0.4, visibility=1.0)
        frame_b[LEFT_WRIST] = SimpleNamespace(x=0.3, y=0.5, visibility=0.2)

        points = average_landmark_points([frame_a, frame_b])

        self.assertEqual(points[NOSE], (0.5, 0.30000000000000004))
        self.assertNotIn(LEFT_WRIST, points)


if __name__ == "__main__":
    unittest.main()
