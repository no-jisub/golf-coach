import json
from pathlib import Path

from utils.guide_alignment import STAGE_KEYS


GROUND_TRUTH_SCHEMA = "golf-coach-swing-stage-ground-truth-v1"
REVIEW_STATUSES = {"pending", "reviewed", "excluded"}


def update_manifest_summary(manifest):
    videos = manifest.get("videos", {})
    status_counts = {
        status: sum(video.get("review_status") == status for video in videos.values())
        for status in sorted(REVIEW_STATUSES)
    }
    manifest["summary"] = {
        "video_count": len(videos),
        **{f"{status}_count": count for status, count in status_counts.items()},
    }
    return manifest["summary"]


def validate_ground_truth_manifest(manifest, *, project_root=None, require_files=False):
    if manifest.get("schema") != GROUND_TRUTH_SCHEMA:
        raise ValueError(f"지원하지 않는 스윙 정답 스키마입니다: {manifest.get('schema')}")
    if manifest.get("view") != "FACEON":
        raise ValueError("현재 자동 단계 검수는 FACEON 영상만 지원합니다.")
    if tuple(manifest.get("stage_order", [])) != STAGE_KEYS:
        raise ValueError("정답 매니페스트의 8단계 순서가 현재 앱과 다릅니다.")
    videos = manifest.get("videos")
    if not isinstance(videos, dict) or not videos:
        raise ValueError("정답 매니페스트에 videos 객체가 필요합니다.")

    seen_sources = set()
    for video_id, video in videos.items():
        if not video_id or not isinstance(video, dict):
            raise ValueError("유효하지 않은 영상 항목이 있습니다.")
        source = video.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError(f"{video_id}: source 영상 경로가 없습니다.")
        normalized_source = source.replace("\\", "/")
        if normalized_source in seen_sources:
            raise ValueError(f"중복된 영상 경로입니다: {source}")
        seen_sources.add(normalized_source)
        if require_files:
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = Path(project_root or Path.cwd()) / source_path
            if not source_path.exists():
                raise FileNotFoundError(f"{video_id}: 영상 파일이 없습니다: {source_path}")

        status = video.get("review_status")
        if status not in REVIEW_STATUSES:
            raise ValueError(f"{video_id}: 지원하지 않는 검수 상태입니다: {status}")
        events = video.get("events")
        if not isinstance(events, dict) or set(events) != set(STAGE_KEYS):
            raise ValueError(f"{video_id}: 정확한 8단계 events 객체가 필요합니다.")
        values = [events[stage_key] for stage_key in STAGE_KEYS]
        invalid_values = [
            value
            for value in values
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0)
        ]
        if invalid_values:
            raise ValueError(f"{video_id}: 프레임 번호는 0 이상의 정수 또는 null이어야 합니다.")
        if status == "reviewed":
            if any(value is None for value in values):
                raise ValueError(f"{video_id}: reviewed 영상은 8단계 프레임이 모두 필요합니다.")
            if values != sorted(set(values)):
                raise ValueError(f"{video_id}: 8단계 프레임은 중복 없이 시간순이어야 합니다.")
        if status == "excluded" and not str(video.get("note", "")).strip():
            raise ValueError(f"{video_id}: 제외 사유 note가 필요합니다.")

    expected_summary = dict(manifest.get("summary", {}))
    actual_summary = update_manifest_summary(manifest)
    if expected_summary and expected_summary != actual_summary:
        raise ValueError(
            f"정답 매니페스트 summary가 실제 항목과 다릅니다: {expected_summary} != {actual_summary}"
        )
    return manifest


def load_ground_truth_manifest(path, *, project_root=None, require_files=False):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"스윙 단계 정답 매니페스트가 없습니다: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return validate_ground_truth_manifest(
        manifest,
        project_root=project_root,
        require_files=require_files,
    )
