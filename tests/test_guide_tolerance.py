import unittest

import cv2
import numpy as np

from utils.guide_alignment import STAGE_KEYS
from utils.guide_skeleton import GUIDE_POSES, draw_guide_skeleton
from utils.guide_tolerance import (
    DEFAULT_MAX_OFFSET,
    calculate_stage_tolerance_regions,
    get_stage_tolerance_regions,
)


class GuideToleranceTests(unittest.TestCase):
    def test_all_stages_have_data_derived_joint_regions(self):
        for stage_key in STAGE_KEYS:
            regions = get_stage_tolerance_regions(stage_key)
            self.assertTrue(regions)
            self.assertTrue(set(regions).issubset(GUIDE_POSES[stage_key]))
            for region in regions.values():
                self.assertTrue(region["metric_keys"])

    def test_regions_include_center_guide_and_stay_local(self):
        for stage_key in STAGE_KEYS:
            regions = get_stage_tolerance_regions(stage_key)
            for joint_index, region in regions.items():
                point = GUIDE_POSES[stage_key][joint_index]
                x_bounds = region["bounds"]["x"]
                y_bounds = region["bounds"]["y"]
                self.assertLessEqual(x_bounds[0], point[0])
                self.assertGreaterEqual(x_bounds[1], point[0])
                self.assertLessEqual(y_bounds[0], point[1])
                self.assertGreaterEqual(y_bounds[1], point[1])
                self.assertLessEqual(x_bounds[1] - x_bounds[0], DEFAULT_MAX_OFFSET * 2.01)
                self.assertLessEqual(y_bounds[1] - y_bounds[0], DEFAULT_MAX_OFFSET * 2.01)

    def test_invalid_stage_is_rejected(self):
        with self.assertRaises(ValueError):
            calculate_stage_tolerance_regions("not-a-stage")

    def test_tolerance_overlay_changes_rendered_pixels(self):
        without_regions = np.zeros((720, 960, 3), dtype=np.uint8)
        with_regions = without_regions.copy()
        draw_guide_skeleton(without_regions, "address")
        draw_guide_skeleton(
            with_regions,
            "address",
            tolerance_regions=get_stage_tolerance_regions("address"),
        )
        difference = cv2.absdiff(without_regions, with_regions)
        self.assertGreater(np.count_nonzero(difference), 0)


if __name__ == "__main__":
    unittest.main()
