import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from utils.dataset_collection import (
    create_collection_session,
    load_active_collection_context,
)
from utils.diagnostic_capture import save_runtime_sample


class DatasetCollectionTests(unittest.TestCase):
    def test_requires_consent_before_creating_private_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                create_collection_session(
                    participant_id="p_test01",
                    body_profile={},
                    capture_conditions={},
                    consent_confirmed=False,
                    dataset_root=temp_dir,
                )

    def test_separates_private_identity_from_session_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = create_collection_session(
                participant_id="p_test01",
                body_profile={
                    "height_band": "170_179",
                    "body_build": "average",
                    "experience_level": "beginner",
                },
                capture_conditions={
                    "view": "FACEON",
                    "club_type": "I7",
                    "camera_id": "studio_a",
                    "resolution": [1280, 720],
                },
                consent_confirmed=True,
                private_profile={"display_name": "Private Name"},
                dataset_root=temp_dir,
                session_id="s_test01",
                created_at=datetime(2026, 7, 29, tzinfo=timezone.utc),
            )
            public = json.loads(Path(session["session_path"]).read_text(encoding="utf-8"))
            private = json.loads(Path(session["private_path"]).read_text(encoding="utf-8"))

            self.assertNotIn("Private Name", json.dumps(public))
            self.assertEqual(private["private_profile"]["display_name"], "Private Name")
            active = load_active_collection_context(temp_dir, required=True)
            self.assertEqual(active["participant_id"], "p_test01")
            self.assertEqual(active["capture_conditions"]["resolution"], [1280, 720])

    def test_capture_uses_participant_session_hierarchy_without_private_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = create_collection_session(
                participant_id="p_test01",
                body_profile={"height_band": "170_179"},
                capture_conditions={"view": "FACEON", "club_type": "I7"},
                consent_confirmed=True,
                private_profile={"display_name": "Private Name"},
                dataset_root=temp_dir,
                session_id="s_test01",
            )
            frame = np.zeros((24, 32, 3), dtype=np.uint8)
            capture = save_runtime_sample(
                raw_frame=frame,
                overlay_frame=frame,
                stage_key="address",
                landmarks=[],
                diagnostics={},
                feedback=None,
                collection_context=context,
            )
            metadata_text = Path(capture["metadata_path"]).read_text(encoding="utf-8")
            metadata = json.loads(metadata_text)

            self.assertIn(
                str(Path("captures") / "p_test01" / "s_test01" / "address"),
                capture["sample_dir"],
            )
            self.assertEqual(metadata["schema"], "golf-coach-runtime-sample-v2")
            self.assertEqual(metadata["collection"]["participant_id"], "p_test01")
            self.assertNotIn("Private Name", metadata_text)


if __name__ == "__main__":
    unittest.main()
