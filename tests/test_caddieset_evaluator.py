import unittest

from utils.caddieset_evaluator import (
    CaddieSetProfileError,
    STAGE_KEYS,
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


if __name__ == "__main__":
    unittest.main()
