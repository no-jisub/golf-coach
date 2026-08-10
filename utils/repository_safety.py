"""Git에 추적된 파일의 개인정보·원본 데이터·비밀값 유출을 검사합니다."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


DEFAULT_MAX_FILE_BYTES = 5 * 1024 * 1024
ALLOWED_LARGE_FILES = {"pose_landmarker_full.task"}

PRIVATE_PATH_PREFIXES = (
    "analysis_sessions/",
    "reference_data/caddieset/source/",
    "reference_data/extracted_landmarks/",
    "reference_data/runtime_samples/",
    "reference_data/webcam_dataset/private/",
    "reference_data/webcam_dataset/sessions/",
    "reference_data/webcam_dataset/captures/",
    "reference_data/webcam_dataset/reviews/",
    "reference_data/webcam_dataset/exports/",
)
RAW_MEDIA_PREFIXES = (
    "reference_data/raw_images/",
    "reference_data/raw_videos/",
    "reference_data/debug_overlay/",
    "reference_data/debug_shaft_overlay/",
    "reference_data/debug_guide_overlay/",
)
BLOCKED_MEDIA_SUFFIXES = {
    ".avi",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".wav",
    ".webm",
}
TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}

SENSITIVE_PATTERNS = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("openai_api_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    (
        "email_address",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    ("korean_phone", re.compile(r"(?<!\d)01[016789][ -]\d{3,4}[ -]\d{4}(?!\d)")),
)


def normalize_repository_path(path):
    normalized = str(path).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def path_policy_violation(path):
    normalized = normalize_repository_path(path)
    if normalized.endswith("/.gitkeep"):
        return None
    if any(normalized.startswith(prefix) for prefix in PRIVATE_PATH_PREFIXES):
        return "private_or_generated_data_path"
    if any(normalized.startswith(prefix) for prefix in RAW_MEDIA_PREFIXES):
        return "raw_or_debug_media_path"
    if Path(normalized).suffix.lower() in BLOCKED_MEDIA_SUFFIXES:
        return "tracked_media_file"
    return None


def find_sensitive_text(text):
    findings = []
    for name, pattern in SENSITIVE_PATTERNS:
        match = pattern.search(text)
        if match:
            line = text.count("\n", 0, match.start()) + 1
            findings.append({"type": name, "line": line})
    return findings


def tracked_files(repository_root):
    root = Path(repository_root).resolve()
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={root.as_posix()}",
            "ls-files",
            "-z",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [
        item.decode("utf-8", errors="strict")
        for item in result.stdout.split(b"\0")
        if item
    ]


def scan_repository_paths(
    repository_root,
    paths,
    *,
    max_file_bytes=DEFAULT_MAX_FILE_BYTES,
):
    root = Path(repository_root).resolve()
    violations = []

    for relative_path in paths:
        normalized = normalize_repository_path(relative_path)
        policy = path_policy_violation(normalized)
        if policy:
            violations.append({"path": normalized, "type": policy})
            continue

        candidate = (root / normalized).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            violations.append({"path": normalized, "type": "path_outside_repository"})
            continue
        if not candidate.is_file():
            violations.append({"path": normalized, "type": "tracked_file_missing"})
            continue

        size = candidate.stat().st_size
        if size > max_file_bytes and normalized not in ALLOWED_LARGE_FILES:
            violations.append(
                {
                    "path": normalized,
                    "type": "large_file",
                    "size_bytes": size,
                    "limit_bytes": max_file_bytes,
                }
            )

        if candidate.suffix.lower() not in TEXT_SUFFIXES or size > 2 * 1024 * 1024:
            continue
        try:
            content = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            violations.append({"path": normalized, "type": "invalid_utf8_text"})
            continue
        for finding in find_sensitive_text(content):
            violations.append({"path": normalized, **finding})

    return violations


def scan_tracked_repository(repository_root, *, max_file_bytes=DEFAULT_MAX_FILE_BYTES):
    return scan_repository_paths(
        repository_root,
        tracked_files(repository_root),
        max_file_bytes=max_file_bytes,
    )
