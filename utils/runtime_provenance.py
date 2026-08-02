"""런타임 판정을 재현하는 데 필요한 코드·모델·데이터 버전을 기록합니다."""

import hashlib
import importlib.metadata
import platform
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_SCHEMA = "golf-coach-runtime-provenance-v1"


def file_sha256(path):
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(project_root=PROJECT_ROOT):
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={Path(project_root).resolve().as_posix()}",
                "rev-parse",
                "HEAD",
            ],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return completed.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_runtime_provenance(
    *,
    model_path,
    runtime_settings,
    project_root=PROJECT_ROOT,
    generated_at=None,
):
    """현재 판정 환경과 모든 입력 데이터의 지문을 만듭니다."""
    project_root = Path(project_root)
    generated_at = generated_at or datetime.now().astimezone()
    data_paths = {
        "caddieset_profile": (
            project_root / "reference_data" / "caddieset" / "evaluation_profiles.json"
        ),
        "aligned_guide": (
            project_root
            / "reference_data"
            / "guide_poses"
            / "caddieset_aligned_guide_poses.json"
        ),
        "generated_guide": (
            project_root
            / "reference_data"
            / "guide_poses"
            / "generated_guide_poses.json"
        ),
    }
    return {
        "schema": PROVENANCE_SCHEMA,
        "generated_at": generated_at.isoformat(),
        "git_commit": git_commit(project_root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "opencv-python": package_version("opencv-python"),
            "mediapipe": package_version("mediapipe"),
            "numpy": package_version("numpy"),
            "Pillow": package_version("Pillow"),
        },
        "model": {
            "path": str(Path(model_path).resolve()),
            "sha256": file_sha256(model_path),
        },
        "reference_data": {
            name: {
                "path": str(path.resolve()),
                "exists": path.exists(),
                "sha256": file_sha256(path),
            }
            for name, path in data_paths.items()
        },
        "runtime_settings": dict(runtime_settings),
    }
