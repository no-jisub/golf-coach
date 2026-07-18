import unittest

from utils.caddieset_evaluator import (
    CaddieSetProfileError,
    STAGE_KEYS,
    classify_metric_comparison,
    classify_stage_comparisons,
    compare_metric_value,
    compare_stage_metrics,
    load_evaluation_profiles,
    make_profile_id,
    select_evaluation_profile,
    select_stage_evaluation_items,
)


class CaddieSetProfileSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_evaluation_profiles()

    def test_profile_id_normalizes_view_and_club(self):
        self.assertEqual(make_profile_id("faceon", "w1"), "faceon_w1")
        self.assertEqual(make_profile_id("DTL"), "dtl_all")

    def test_selects_stage_items_for_all_current_stages(self):
        total_items = 0
        for expected_index, stage_key in enumerate(STAGE_KEYS):
            selection = select_stage_evaluation_items(
                stage_key,
                data=self.data,
                view="FACEON",
            )
            self.assertEqual(selection["source_stage_index"], expected_index)
            self.assertTrue(selection["evaluation_items"])
            total_items += len(selection["evaluation_items"])

        self.assertEqual(total_items, 40)

    def test_selects_requested_club_profile(self):
        selection = select_stage_evaluation_items(
            "address",
            data=self.data,
            view="FACEON",
            club_type="W1",
        )
        self.assertEqual(selection["profile_id"], "faceon_w1")
        self.assertFalse(selection["used_fallback"])

    def test_unknown_club_falls_back_to_view_profile(self):
        selection = select_evaluation_profile(
            self.data,
            view="FACEON",
            club_type="SW",
        )
        self.assertEqual(selection["profile_id"], "faceon_all")
        self.assertEqual(selection["requested_profile_id"], "faceon_sw")
        self.assertTrue(selection["used_fallback"])

    def test_unknown_stage_is_rejected(self):
        with self.assertRaises(CaddieSetProfileError):
            select_stage_evaluation_items("transition", data=self.data)

    def test_missing_stage_in_profile_is_rejected(self):
        malformed = {
            "profiles": {
                "faceon_all": {
                    "view": "FACEON",
                    "club_type": "ALL",
                    "stages": {"address": {"evaluation_items": {"x": {}}}},
                }
            }
        }
        with self.assertRaises(CaddieSetProfileError):
            select_evaluation_profile(malformed)


class CaddieSetMetricComparisonTests(unittest.TestCase):
    def setUp(self):
        self.item = {
            "source_column": "0-SHOULDER-ANGLE",
            "description": "어깨선 각도",
            "unit": "degree",
            "target": 10.0,
            "observed_reference_range": [5.0, 15.0],
            "observed_outer_range": [2.0, 18.0],
        }

    def test_compares_value_to_reference_and_outer_ranges(self):
        cases = [
            (1.9, "below_outer"),
            (2.0, "below_reference"),
            (5.0, "within_reference"),
            (15.0, "within_reference"),
            (18.0, "above_reference"),
            (18.1, "above_outer"),
        ]
        for value, expected_relation in cases:
            with self.subTest(value=value):
                comparison = compare_metric_value("shoulder_angle", value, self.item)
                self.assertEqual(comparison["relation"], expected_relation)

    def test_comparison_includes_target_delta(self):
        comparison = compare_metric_value("shoulder_angle", 12.5, self.item)
        self.assertEqual(comparison["delta_to_target"], 2.5)
        self.assertEqual(comparison["normalized_delta"], 0.5)

    def test_missing_measurement_is_preserved(self):
        comparison = compare_metric_value("shoulder_angle", None, self.item)
        self.assertEqual(comparison["relation"], "unavailable")
        self.assertIsNone(comparison["measured_value"])

    def test_stage_comparison_uses_only_selected_items(self):
        stage_selection = {
            "stage_key": "address",
            "profile_id": "faceon_all",
            "view": "FACEON",
            "club_type": "ALL",
            "used_fallback": False,
            "evaluation_items": {
                "shoulder_angle": self.item,
                "stance_ratio": {
                    **self.item,
                    "source_column": "0-STANCE-RATIO",
                    "unit": "ratio",
                },
            },
        }
        result = compare_stage_metrics(
            {"shoulder_angle": 9.0, "stance_ratio": 11.0, "head_loc": 99.0},
            stage_selection,
        )
        self.assertEqual(set(result["comparisons"]), {"shoulder_angle", "stance_ratio"})
        self.assertEqual(result["stage_key"], "address")


class CaddieSetClassificationTests(unittest.TestCase):
    def make_comparison(self, relation):
        return {
            "metric_key": "shoulder_angle",
            "relation": relation,
            "measured_value": None if relation == "unavailable" else 10.0,
        }

    def test_classifies_each_relation(self):
        expected = {
            "within_reference": ("pass", None),
            "below_reference": ("warning", "outside_reference"),
            "above_reference": ("warning", "outside_reference"),
            "below_outer": ("warning", "outside_observed"),
            "above_outer": ("warning", "outside_observed"),
            "unavailable": ("unavailable", None),
        }
        for relation, (status, warning_level) in expected.items():
            with self.subTest(relation=relation):
                result = classify_metric_comparison(self.make_comparison(relation))
                self.assertEqual(result["status"], status)
                self.assertEqual(result["warning_level"], warning_level)

    def test_stage_passes_only_when_every_item_passes(self):
        result = classify_stage_comparisons(
            {
                "stage_key": "address",
                "comparisons": {
                    "a": self.make_comparison("within_reference"),
                    "b": self.make_comparison("within_reference"),
                },
            }
        )
        self.assertEqual(result["overall_status"], "pass")
        self.assertTrue(result["passed"])

    def test_stage_warns_for_outlier_or_partial_measurement(self):
        warning_result = classify_stage_comparisons(
            {
                "comparisons": {
                    "a": self.make_comparison("within_reference"),
                    "b": self.make_comparison("above_outer"),
                }
            }
        )
        partial_result = classify_stage_comparisons(
            {
                "comparisons": {
                    "a": self.make_comparison("within_reference"),
                    "b": self.make_comparison("unavailable"),
                }
            },
            minimum_measurement_ratio=0.5,
        )
        self.assertEqual(warning_result["overall_status"], "warning")
        self.assertEqual(partial_result["overall_status"], "warning")

    def test_stage_is_unavailable_when_too_few_items_are_measured(self):
        result = classify_stage_comparisons(
            {
                "comparisons": {
                    "a": self.make_comparison("within_reference"),
                    "b": self.make_comparison("unavailable"),
                    "c": self.make_comparison("unavailable"),
                }
            }
        )
        self.assertEqual(result["overall_status"], "unavailable")
        self.assertEqual(result["summary"]["measured_count"], 1)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
