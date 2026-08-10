import tempfile
import unittest
from pathlib import Path

from utils.repository_safety import (
    find_sensitive_text,
    path_policy_violation,
    scan_repository_paths,
)


class RepositorySafetyTests(unittest.TestCase):
    def test_private_and_media_paths_are_rejected_but_gitkeep_is_allowed(self):
        self.assertEqual(
            path_policy_violation("reference_data/webcam_dataset/private/person.json"),
            "private_or_generated_data_path",
        )
        self.assertEqual(
            path_policy_violation("reference_data/raw_videos/swing.mp4"),
            "raw_or_debug_media_path",
        )
        self.assertEqual(path_policy_violation("exports/swing.mov"), "tracked_media_file")
        self.assertIsNone(path_policy_violation("reference_data/raw_videos/.gitkeep"))

    def test_high_confidence_secret_and_contact_patterns_are_detected(self):
        token = "ghp_" + "a" * 36
        email = "user" + "@" + "example.com"
        phone = "010" + "-1234" + "-5678"
        findings = find_sensitive_text(
            f"token={token}\ncontact={email}\nphone={phone}"
        )
        self.assertEqual(
            {item["type"] for item in findings},
            {"github_token", "email_address", "korean_phone"},
        )

    def test_scan_reports_large_files_without_exposing_contents(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "large.bin").write_bytes(b"x" * 9)
            violations = scan_repository_paths(
                root,
                ["large.bin"],
                max_file_bytes=8,
            )
        self.assertEqual(violations[0]["type"], "large_file")
        self.assertNotIn("content", violations[0])

    def test_scan_rejects_paths_outside_repository(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            violations = scan_repository_paths(temporary_directory, ["../outside.txt"])
        self.assertEqual(violations[0]["type"], "path_outside_repository")


if __name__ == "__main__":
    unittest.main()
