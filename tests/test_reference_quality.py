import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import audit_reference_samples as audit
from tools import build_guide_poses as builder
from tools import review_reference_samples as reviewer
from tools import visualize_guide_poses
from utils import guide_skeleton


def make_landmarks(low_visibility_index=None):
    points = {
        0: (0.50, 0.15),
        11: (0.40, 0.30),
        12: (0.60, 0.30),
        13: (0.43, 0.42),
        14: (0.57, 0.42),
        15: (0.48, 0.55),
        16: (0.52, 0.55),
        23: (0.43, 0.52),
        24: (0.57, 0.52),
        25: (0.40, 0.70),
        26: (0.60, 0.70),
        27: (0.35, 0.90),
        28: (0.65, 0.90),
    }
    return [
        {
            "index": index,
            "x": x,
            "y": y,
            "visibility": 0.10 if index == low_visibility_index else 0.95,
        }
        for index, (x, y) in points.items()
    ]


def make_sample(low_visibility_index=None, shaft_score=0.70):
    return {
        "stage": "address",
        "image": "reference_data/raw_images/address/sample.jpg",
        "detected": True,
        "image_width": 1000,
        "image_height": 1000,
        "landmarks": make_landmarks(low_visibility_index),
        "shaft": {
            "source": "test",
            "start": [0.50, 0.55],
            "end": [0.44, 0.82],
            "score": shaft_score,
        },
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class AuditReferenceSamplesTests(unittest.TestCase):
    def audit_one(self, data):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_path = root / "reference_data" / "extracted_landmarks" / "address" / "sample.json"
            write_json(sample_path, data)
            return audit.audit_sample(sample_path, project_root=root)

    def test_clean_sample_passes(self):
        result = self.audit_one(make_sample())

        self.assertEqual("pass", result["auto_check"]["status"])
        self.assertEqual([], result["auto_check"]["reasons"])

    def test_low_visibility_fails_with_reason(self):
        result = self.audit_one(make_sample(low_visibility_index=15))

        self.assertEqual("fail", result["auto_check"]["status"])
        reason_codes = {reason["code"] for reason in result["auto_check"]["reasons"]}
        self.assertIn("low_landmark_visibility", reason_codes)

    def test_low_shaft_score_warns(self):
        result = self.audit_one(make_sample(shaft_score=0.45))

        self.assertEqual("warning", result["auto_check"]["status"])
        reason_codes = {reason["code"] for reason in result["auto_check"]["reasons"]}
        self.assertIn("shaft_score_low", reason_codes)

    def test_manifest_refresh_preserves_human_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            extracted_dir = root / "reference_data" / "extracted_landmarks"
            sample_path = extracted_dir / "address" / "sample.json"
            manifest_path = root / "reference_data" / "review_manifest.json"
            write_json(sample_path, make_sample())
            sample_key = "reference_data/extracted_landmarks/address/sample.json"
            write_json(
                manifest_path,
                {
                    "schema": audit.REVIEW_SCHEMA,
                    "samples": {
                        sample_key: {
                            "human_review": {
                                "status": "accepted",
                                "override_auto_fail": False,
                                "note": "직접 확인함",
                            }
                        }
                    },
                },
            )

            manifest = audit.build_manifest(
                extracted_dir=extracted_dir,
                existing_manifest_path=manifest_path,
                preserve_reviews=True,
                project_root=root,
            )

        self.assertEqual("accepted", manifest["samples"][sample_key]["human_review"]["status"])
        self.assertEqual("직접 확인함", manifest["samples"][sample_key]["human_review"]["note"])

    def test_semantic_check_rejects_finish_pose_labeled_as_top(self):
        base_pose = {
            index: (0.0, 0.0)
            for index in audit.REQUIRED_LANDMARKS
        }
        address_pose = copy.deepcopy(base_pose)
        address_pose[audit.LEFT_ANKLE] = (-1.0, 1.0)
        address_pose[audit.RIGHT_ANKLE] = (1.0, 1.0)
        finish_pose = copy.deepcopy(base_pose)
        finish_pose[audit.LEFT_WRIST] = (-0.2, -0.5)
        finish_pose[audit.RIGHT_WRIST] = (0.2, -0.5)
        finish_pose[audit.LEFT_ANKLE] = (-0.2, 1.0)
        finish_pose[audit.RIGHT_ANKLE] = (0.2, 1.0)
        samples = {
            "address": {
                "stage": "address",
                "source": "reference_data/extracted_landmarks/address/pro01_video_address_auto.json",
                "auto_check": {
                    "status": "pass",
                    "reasons": [],
                    "metrics": {"shoulder_to_body_ratio": 0.3},
                },
                "_normalized_pose": address_pose,
            },
            "top": {
                "stage": "top",
                "source": "reference_data/extracted_landmarks/top/pro01_video_top_auto.json",
                "auto_check": {
                    "status": "pass",
                    "reasons": [],
                    "metrics": {"shoulder_to_body_ratio": 0.3},
                },
                "_normalized_pose": finish_pose,
            },
        }

        audit.add_stage_semantic_checks(samples)

        reason_codes = {reason["code"] for reason in samples["top"]["auto_check"]["reasons"]}
        self.assertEqual("fail", samples["top"]["auto_check"]["status"])
        self.assertIn("top_stance_looks_finished", reason_codes)

    def test_semantic_check_rejects_top_pose_labeled_as_downswing(self):
        pose = {
            index: (0.0, 0.0)
            for index in audit.REQUIRED_LANDMARKS
        }
        pose[audit.LEFT_WRIST] = (-0.3, -0.5)
        pose[audit.RIGHT_WRIST] = (0.3, -0.5)
        pose[audit.LEFT_ANKLE] = (-1.0, 1.0)
        pose[audit.RIGHT_ANKLE] = (1.0, 1.0)
        sample = {
            "stage": "downswing",
            "source": "reference_data/extracted_landmarks/downswing/pro01_video_downswing_auto.json",
            "auto_check": {"status": "pass", "reasons": [], "metrics": {}},
            "_normalized_pose": pose,
        }

        audit.add_stage_semantic_checks({"sample": sample})

        reason_codes = {reason["code"] for reason in sample["auto_check"]["reasons"]}
        self.assertIn("downswing_hands_still_at_top", reason_codes)


class GuidePoseInclusionTests(unittest.TestCase):
    def setUp(self):
        self.review_key = "reference_data/extracted_landmarks/address/sample.json"
        self.sample_path = builder.PROJECT_ROOT / self.review_key
        self.base_manifest = {
            "schema": builder.REVIEW_SCHEMA,
            "samples": {
                self.review_key: {
                    "auto_check": {"status": "pass"},
                    "human_review": {
                        "status": "pending",
                        "override_auto_fail": False,
                        "note": "",
                    },
                }
            },
        }

    def test_inclusion_policy(self):
        cases = [
            ("pending", "pass", False, (False, "pending")),
            ("rejected", "pass", False, (False, "rejected")),
            ("accepted", "warning", False, (True, "included")),
            ("accepted", "fail", False, (False, "auto_fail_without_override")),
            ("accepted", "fail", True, (True, "included")),
        ]
        for human_status, auto_status, override, expected in cases:
            with self.subTest(human_status=human_status, auto_status=auto_status, override=override):
                manifest = copy.deepcopy(self.base_manifest)
                review = manifest["samples"][self.review_key]
                review["auto_check"]["status"] = auto_status
                review["human_review"]["status"] = human_status
                review["human_review"]["override_auto_fail"] = override
                self.assertEqual(expected, builder.get_sample_inclusion(self.sample_path, manifest))

    def test_build_stage_uses_only_accepted_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage_dir = root / "reference_data" / "extracted_landmarks" / "address"
            accepted_path = stage_dir / "accepted.json"
            pending_path = stage_dir / "pending.json"
            write_json(accepted_path, make_sample())
            write_json(pending_path, make_sample())

            accepted_key = "reference_data/extracted_landmarks/address/accepted.json"
            pending_key = "reference_data/extracted_landmarks/address/pending.json"
            manifest = {
                "schema": builder.REVIEW_SCHEMA,
                "samples": {
                    accepted_key: {
                        "auto_check": {"status": "pass"},
                        "human_review": {
                            "status": "accepted",
                            "override_auto_fail": False,
                            "include_shaft": False,
                        },
                    },
                    pending_key: {
                        "auto_check": {"status": "pass"},
                        "human_review": {"status": "pending", "override_auto_fail": False},
                    },
                },
            }

            with mock.patch.object(builder, "PROJECT_ROOT", root), mock.patch.object(
                builder, "EXTRACTED_DIR", root / "reference_data" / "extracted_landmarks"
            ):
                pose, shaft, used_files, stats = builder.build_stage_pose("address", manifest)

        self.assertIsNotNone(pose)
        self.assertIsNone(shaft)
        self.assertEqual([accepted_key], used_files)
        self.assertEqual(1, stats["included"])
        self.assertEqual(1, stats["pending"])
        self.assertEqual(1, stats["shaft_excluded_by_review"])

    def test_normalize_shaft_orients_nearest_endpoint_as_grip(self):
        landmarks = {
            item["index"]: {
                "x": item["x"],
                "y": item["y"],
                "visibility": item["visibility"],
            }
            for item in make_landmarks()
        }
        shaft = {
            "start": (0.10, 0.80),
            "end": (0.50, 0.55),
        }

        normalized = builder.normalize_shaft(shaft, landmarks)

        self.assertGreater(normalized["club"]["y"], normalized["grip"]["y"])


class ReviewReferenceSamplesTests(unittest.TestCase):
    def make_manifest(self):
        return {
            "schema": reviewer.REVIEW_SCHEMA,
            "summary": {},
            "samples": {
                "address-pass.json": {
                    "stage": "address",
                    "auto_check": {"status": "pass", "reasons": [], "metrics": {}},
                    "human_review": {"status": "pending", "override_auto_fail": False, "note": ""},
                },
                "impact-fail.json": {
                    "stage": "impact",
                    "auto_check": {"status": "fail", "reasons": [], "metrics": {}},
                    "human_review": {"status": "pending", "override_auto_fail": False, "note": ""},
                },
            },
        }

    def test_filter_samples_combines_filters(self):
        manifest = self.make_manifest()

        rows = reviewer.filter_samples(manifest, stage="impact", auto_status="fail", review_status="pending")

        self.assertEqual(["impact-fail.json"], [key for key, _ in rows])

    def test_auto_fail_requires_explicit_override(self):
        sample = self.make_manifest()["samples"]["impact-fail.json"]

        with self.assertRaises(ValueError):
            reviewer.apply_decision(sample, "accepted")

        review = reviewer.apply_decision(sample, "accepted", override_auto_fail=True)
        self.assertEqual("accepted", review["status"])
        self.assertTrue(review["override_auto_fail"])

    def test_summary_counts_reviews_and_overrides(self):
        manifest = self.make_manifest()
        reviewer.apply_decision(manifest["samples"]["address-pass.json"], "accepted")
        reviewer.apply_decision(
            manifest["samples"]["impact-fail.json"],
            "accepted",
            override_auto_fail=True,
        )

        summary = reviewer.update_review_summary(manifest)

        self.assertEqual(2, summary["accepted"])
        self.assertEqual(0, summary["pending"])
        self.assertEqual(1, summary["auto_fail_overrides"])

    def test_shaft_inclusion_can_be_reviewed_separately(self):
        sample = self.make_manifest()["samples"]["address-pass.json"]

        reviewer.set_shaft_inclusion(sample, False)

        self.assertFalse(sample["human_review"]["include_shaft"])


class GeneratedGuideFallbackTests(unittest.TestCase):
    def test_generated_stages_override_defaults_and_missing_stages_fall_back(self):
        defaults = {
            "address": {0: (0.5, 0.1)},
            "downswing": {0: (0.4, 0.2)},
        }
        generated = {
            "address": {0: (0.6, 0.15)},
        }

        merged = guide_skeleton.merge_generated_guides(defaults, generated)

        self.assertEqual({0: (0.6, 0.15)}, merged["address"])
        self.assertEqual({0: (0.4, 0.2)}, merged["downswing"])

    def test_final_guide_contact_sheet_contains_all_eight_stages(self):
        sheet = visualize_guide_poses.build_contact_sheet()

        self.assertEqual(
            (visualize_guide_poses.PANEL_HEIGHT * 2, visualize_guide_poses.PANEL_WIDTH * 4, 3),
            sheet.shape,
        )


if __name__ == "__main__":
    unittest.main()
