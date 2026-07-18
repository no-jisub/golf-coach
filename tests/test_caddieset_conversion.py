import importlib.util
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(module_name, PROJECT_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


converter = load_module("convert_caddieset", "tools/convert_caddieset.py")


class CaddieSetConversionTests(unittest.TestCase):
    def test_stage_column_mapping(self):
        self.assertEqual(
            converter.split_feature_column("4-RIGHT-ARMPIT-ANGLE"),
            (4, "RIGHT-ARMPIT-ANGLE"),
        )
        self.assertIsNone(converter.split_feature_column("BallSpeed"))

    def test_reference_outcome_uses_paper_thresholds(self):
        self.assertTrue(
            converter.is_reference_outcome({"DirectionAngle": "-6", "SpinAxis": "10"})
        )
        self.assertFalse(
            converter.is_reference_outcome({"DirectionAngle": "6.01", "SpinAxis": "0"})
        )
        self.assertFalse(
            converter.is_reference_outcome({"DirectionAngle": "0", "SpinAxis": ""})
        )

    def test_target_weights_each_golfer_equally(self):
        rows = [
            {"GolferId": "1", "0-SPINE-ANGLE": "0"},
            {"GolferId": "1", "0-SPINE-ANGLE": "0"},
            {"GolferId": "1", "0-SPINE-ANGLE": "0"},
            {"GolferId": "2", "0-SPINE-ANGLE": "10"},
        ]
        summary = converter.summarize_metric(rows, "0-SPINE-ANGLE")
        self.assertEqual(summary["target"], 5.0)
        self.assertEqual(summary["shot_median"], 0.0)
        self.assertEqual(summary["golfer_count"], 2)

    def test_build_profile_keeps_all_eight_current_stages(self):
        fieldnames = ["View", "ClubType", "DirectionAngle", "SpinAxis", "GolferId"]
        row = {
            "View": "FACEON",
            "ClubType": "W1",
            "DirectionAngle": "0",
            "SpinAxis": "0",
            "GolferId": "1",
        }
        for index in range(8):
            column_name = f"{index}-SHOULDER-ANGLE"
            fieldnames.append(column_name)
            row[column_name] = str(index + 1)

        with tempfile.TemporaryDirectory() as temporary_dir:
            fake_input = Path(temporary_dir) / "CaddieSet.csv"
            fake_input.write_text("fixture", encoding="utf-8")
            output = converter.build_output(
                [row],
                fieldnames,
                fake_input,
                profile_specs=[("FACEON", None)],
            )

        stages = output["profiles"]["faceon_all"]["stages"]
        self.assertEqual(list(stages), converter.STAGES)
        self.assertEqual(
            stages["downswing"]["evaluation_items"]["shoulder_angle"]["source_column"],
            "4-SHOULDER-ANGLE",
        )


if __name__ == "__main__":
    unittest.main()
