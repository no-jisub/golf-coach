import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import audit_reference_samples as audit
from tools import build_guide_poses as builder


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
                        "human_review": {"status": "accepted", "override_auto_fail": False},
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
                pose, _, used_files, stats = builder.build_stage_pose("address", manifest)

        self.assertIsNotNone(pose)
        self.assertEqual([accepted_key], used_files)
        self.assertEqual(1, stats["included"])
        self.assertEqual(1, stats["pending"])


if __name__ == "__main__":
    unittest.main()
