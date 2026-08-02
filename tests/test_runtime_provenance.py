import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from utils.runtime_provenance import build_runtime_provenance, file_sha256


class RuntimeProvenanceTests(unittest.TestCase):
    def test_hash_and_provenance_capture_model_and_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "pose.task"
            model.write_bytes(b"model-data")
            provenance = build_runtime_provenance(
                model_path=model,
                runtime_settings={"threshold": 70},
                project_root=root,
                generated_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
            )
            self.assertEqual(provenance["model"]["sha256"], file_sha256(model))
            self.assertEqual(provenance["runtime_settings"]["threshold"], 70)
            self.assertEqual(
                provenance["generated_at"],
                "2026-08-02T00:00:00+00:00",
            )


if __name__ == "__main__":
    unittest.main()
