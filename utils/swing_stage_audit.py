import json
from pathlib import Path

from utils.guide_alignment import STAGE_KEYS
from utils.swing_stage_detector import detect_stage_events
from utils.swing_video import LANDMARK_SCHEMA, extract_video_landmarks, write_json


GROUND_TRUTH_SCHEMA = "golf-coach-swing-stage-ground-truth-v1"
REVIEW_STATUSES = {"pending", "reviewed", "excluded"}
SEQUENCE_MODES = {"linear", "cyclic"}
AUDIT_SCHEMA = "golf-coach-swing-stage-audit-v1"


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
        sequence_mode = video.get("sequence_mode", "linear")
        if sequence_mode not in SEQUENCE_MODES:
            raise ValueError(f"{video_id}: 지원하지 않는 영상 순서 방식입니다: {sequence_mode}")
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
            if len(values) != len(set(values)):
                raise ValueError(f"{video_id}: 8단계 프레임은 중복될 수 없습니다.")
            wrap_count = sum(current < previous for previous, current in zip(values, values[1:]))
            if sequence_mode == "linear" and wrap_count:
                raise ValueError(f"{video_id}: linear 영상의 8단계 프레임은 시간순이어야 합니다.")
            if sequence_mode == "cyclic" and wrap_count > 1:
                raise ValueError(f"{video_id}: cyclic 영상은 프레임 경계를 한 번만 순환할 수 있습니다.")
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


def _resolve_source(project_root, source):
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = Path(project_root) / source_path
    return source_path.resolve()


def _load_cached_landmarks(path, *, expected_source=None):
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != LANDMARK_SCHEMA:
        return None
    if expected_source is not None:
        cached_source = payload.get("source_video")
        if not cached_source:
            return None
        if Path(cached_source).resolve() != Path(expected_source).resolve():
            return None
    return payload


def audit_manifest_videos(
    manifest,
    *,
    project_root,
    output_root,
    model_path,
    video_ids=None,
    sample_step=1,
    reuse_landmarks=True,
    progress=False,
    landmark_extractor=extract_video_landmarks,
    stage_detector=detect_stage_events,
):
    """Run automatic stage detection for every selected non-excluded manifest video."""
    validate_ground_truth_manifest(manifest, project_root=project_root, require_files=False)
    output_root = Path(output_root)
    requested = set(video_ids or manifest["videos"])
    unknown = requested - set(manifest["videos"])
    if unknown:
        raise ValueError(f"정답 매니페스트에 없는 영상 ID입니다: {', '.join(sorted(unknown))}")

    results = {}
    for video_id, video in manifest["videos"].items():
        if video_id not in requested:
            continue
        if video["review_status"] == "excluded":
            results[video_id] = {
                "status": "excluded",
                "source": video["source"],
                "ground_truth_status": "excluded",
                "note": video.get("note", ""),
            }
            continue

        source_path = _resolve_source(project_root, video["source"])
        session_dir = output_root / video_id
        landmarks_path = session_dir / "frame_landmarks.json"
        events_path = session_dir / "stage_events.json"
        try:
            payload = (
                _load_cached_landmarks(
                    landmarks_path,
                    expected_source=source_path,
                )
                if reuse_landmarks
                else None
            )
            cache_status = "reused"
            if payload is None:
                if not source_path.exists():
                    raise FileNotFoundError(f"영상 파일이 없습니다: {source_path}")
                if progress:
                    print(f"[{video_id}] MediaPipe 관절 추출 시작: {source_path.name}")
                payload = landmark_extractor(
                    source_path,
                    model_path,
                    sample_step=sample_step,
                    progress=progress,
                )
                write_json(landmarks_path, payload)
                cache_status = "created"
            detection = stage_detector(payload)
            write_json(events_path, detection)
            results[video_id] = {
                "status": "ok",
                "source": video["source"],
                "ground_truth_status": video["review_status"],
                "landmark_cache": cache_status,
                "video": payload.get("video", {}),
                "sampling": payload.get("sampling", {}),
                "stage_detection": {
                    "method": detection.get("method"),
                    "diagnostics": detection.get("diagnostics", {}),
                    "events": {
                        stage_key: detection["stages"][stage_key]
                        for stage_key in STAGE_KEYS
                    },
                },
                "artifacts": {
                    "frame_landmarks": str(landmarks_path),
                    "stage_events": str(events_path),
                },
            }
            if progress:
                frames = [
                    detection["stages"][stage_key]["frame_index"]
                    for stage_key in STAGE_KEYS
                ]
                print(f"[{video_id}] 8단계 완료: {frames}")
        except Exception as error:
            results[video_id] = {
                "status": "failed",
                "source": video["source"],
                "ground_truth_status": video["review_status"],
                "error_type": type(error).__name__,
                "error": str(error),
            }
            if progress:
                print(f"[{video_id}] 실패: {type(error).__name__}: {error}")

    summary = {
        "requested_count": len(requested),
        "processed_count": sum(result["status"] == "ok" for result in results.values()),
        "failed_count": sum(result["status"] == "failed" for result in results.values()),
        "excluded_count": sum(result["status"] == "excluded" for result in results.values()),
        "created_cache_count": sum(
            result.get("landmark_cache") == "created" for result in results.values()
        ),
        "reused_cache_count": sum(
            result.get("landmark_cache") == "reused" for result in results.values()
        ),
    }
    return {
        "schema": AUDIT_SCHEMA,
        "stage_order": list(STAGE_KEYS),
        "summary": summary,
        "videos": results,
    }
