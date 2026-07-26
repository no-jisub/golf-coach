import unittest
from types import SimpleNamespace

import main
from utils.golf_rules import (
    STAGE_CONFIGS,
    analyze_stage_pose,
    build_caddieset_feedback,
    build_caddieset_messages,
    build_metric_feedback_message,
)
from utils.guide_skeleton import (
    GUIDE_POSES,
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
    create_calibration_profile,
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


def make_landmarks(points):
    landmarks = [SimpleNamespace(x=0.0, y=0.0, visibility=0.0) for _ in range(33)]
    for index, point in points.items():
        landmarks[index] = SimpleNamespace(x=point[0], y=point[1], visibility=1.0)
    return landmarks


def make_comparison(metric_key, relation, warning_level, normalized_delta=1.0):
    return {
        "metric_key": metric_key,
        "description": metric_key,
        "unit": "degree",
        "target": 10.0,
        "measured_value": 20.0,
        "reference_range": [5.0, 15.0],
        "outer_range": [2.0, 18.0],
        "relation": relation,
        "normalized_delta": normalized_delta,
        "status": "warning",
        "warning_level": warning_level,
    }


class CaddieSetFeedbackTests(unittest.TestCase):
    def test_position_feedback_uses_display_direction(self):
        comparison = make_comparison(
            "head_loc",
            "below_reference",
            "outside_reference",
        )
        comparison.update(
            {
                "unit": "ratio",
                "target": 0.0,
                "measured_value": -0.5,
                "reference_range": [-0.2, 0.2],
            }
        )
        message = build_metric_feedback_message(comparison, direction_multiplier=-1.0)
        self.assertIn("화면 왼쪽", message)
        self.assertIn("참조", message)

    def test_strongest_warning_is_shown_first(self):
        classified = {
            "overall_status": "warning",
            "summary": {"pass_count": 0},
            "comparisons": {
                "shoulder_angle": make_comparison(
                    "shoulder_angle", "above_reference", "outside_reference", 3.0
                ),
                "right_arm_angle": make_comparison(
                    "right_arm_angle", "above_outer", "outside_observed", 1.0
                ),
            },
        }
        messages = build_caddieset_messages("address", classified)
        self.assertIn("오른팔", messages[0])

    def test_unavailable_stage_explains_camera_visibility(self):
        unavailable = make_comparison("left_arm_angle", "unavailable", None)
        unavailable.update(
            {
                "status": "unavailable",
                "measured_value": None,
                "description": "왼팔 각도",
            }
        )
        classified = {
            "overall_status": "unavailable",
            "summary": {"pass_count": 0},
            "comparisons": {"left_arm_angle": unavailable},
        }
        messages = build_caddieset_messages("top", classified)
        self.assertIn("전신과 양팔", messages[0])
        self.assertIn("측정", messages[1])

    def test_build_feedback_runs_end_to_end_for_address(self):
        points = make_points()
        result = build_caddieset_feedback(
            "address",
            [make_landmarks(points)],
            {"caddieset_address_points": points},
        )
        self.assertEqual(result["source"], "caddieset")
        self.assertIn(result["status"], {"pass", "warning", "unavailable"})
        self.assertEqual(set(result["item_results"]), {"shoulder_angle", "stance_ratio", "upper_tilt"})
        self.assertTrue(result["messages"])

    def test_all_eight_guide_stages_run_through_caddieset_pipeline(self):
        address_points = GUIDE_POSES["address"]
        expected_item_counts = [3, 6, 6, 5, 6, 7, 5, 2]
        for stage, expected_count in zip(STAGE_CONFIGS, expected_item_counts):
            with self.subTest(stage=stage["key"]):
                result = build_caddieset_feedback(
                    stage["key"],
                    [make_landmarks(GUIDE_POSES[stage["key"]])],
                    {"caddieset_address_points": address_points},
                )
                self.assertEqual(len(result["item_results"]), expected_count)
                self.assertIn(result["status"], {"pass", "warning", "unavailable"})

    def test_runtime_combines_guide_and_i7_reference_scores(self):
        address_landmarks = make_landmarks(GUIDE_POSES["address"])
        calibration_profile = create_calibration_profile(
            [address_landmarks],
            1000,
            1000,
        )
        calibration_profile["caddieset_address_points"] = GUIDE_POSES["address"]

        result = analyze_stage_pose(
            "address",
            [address_landmarks],
            calibration_profile,
            1000,
            1000,
        )

        self.assertEqual(result["source"], "combined")
        self.assertEqual(result["metrics"]["profile_id"], "faceon_i7")
        self.assertIn("final_score", result["metrics"])
        self.assertIn("guide_score", result["metrics"])
        self.assertIn("caddieset_score", result["metrics"])
        self.assertTrue(result["passed"])

    def test_all_visible_runtime_guides_pass_combined_scoring(self):
        address_landmarks = make_landmarks(GUIDE_POSES["address"])
        calibration_profile = create_calibration_profile(
            [address_landmarks],
            1000,
            1000,
        )
        calibration_profile["caddieset_address_points"] = GUIDE_POSES["address"]

        for stage in STAGE_CONFIGS:
            with self.subTest(stage=stage["key"]):
                result = analyze_stage_pose(
                    stage["key"],
                    [make_landmarks(GUIDE_POSES[stage["key"]])],
                    calibration_profile,
                    1000,
                    1000,
                )
                self.assertTrue(result["passed"], result["messages"])
                self.assertGreaterEqual(result["metrics"]["final_score"], 70)

    def test_screen_title_distinguishes_warning_and_unavailable(self):
        stage = STAGE_CONFIGS[0]
        warning_title, _ = main.get_stage_status_text(
            stage,
            {"passed": False, "status": "warning"},
        )
        unavailable_title, _ = main.get_stage_status_text(
            stage,
            {"passed": False, "status": "unavailable"},
        )
        self.assertIn("주의", warning_title)
        self.assertIn("측정 불가", unavailable_title)


if __name__ == "__main__":
    unittest.main()
