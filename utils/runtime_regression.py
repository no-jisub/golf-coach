import json
import math
from pathlib import Path
from types import SimpleNamespace

from utils.caddieset_metrics import average_landmark_points
from utils.golf_rules import STAGE_CONFIGS, analyze_stage_pose
from utils.guide_alignment import STAGE_KEYS
from utils.guide_skeleton import create_calibration_profile


REGRESSION_SCHEMA = "golf-coach-runtime-regression-v1"
STRICT_SCORE_THRESHOLD = 70.0
STRICT_PASS_RATE_THRESHOLD = 0.5
LENIENT_SCORE_THRESHOLD = 90.0
LENIENT_PASS_RATE_THRESHOLD = 0.9


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def deserialize_landmarks(frame_record):
    """저장된 MediaPipe 좌표를 앱 판정 함수가 받는 객체 목록으로 변환합니다."""
    records = frame_record.get("landmarks", [])
    if not records:
        return []

    size = max(int(record["index"]) for record in records) + 1
    landmarks = [
        SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=0.0, presence=0.0)
        for _ in range(size)
    ]
    for record in records:
        index = int(record["index"])
        landmarks[index] = SimpleNamespace(
            x=float(record.get("x", 0.0)),
            y=float(record.get("y", 0.0)),
            z=float(record.get("z", 0.0)),
            visibility=float(record.get("visibility", 1.0)),
            presence=float(record.get("presence", 1.0)),
        )
    return landmarks


def detected_frame_lookup(landmark_payload):
    return {
        int(frame["frame_index"]): frame
        for frame in landmark_payload.get("frames", [])
        if frame.get("detected") and frame.get("landmarks")
    }


def nearest_detected_frame(frame_lookup, requested_index):
    if not frame_lookup:
        raise ValueError("관절이 인식된 프레임이 없습니다.")
    requested_index = int(requested_index)
    actual_index = min(frame_lookup, key=lambda index: (abs(index - requested_index), index))
    return frame_lookup[actual_index], actual_index - requested_index


def select_stage_frames(manifest_video, automatic_events):
    """검수 완료 영상은 확정 프레임을, 나머지는 자동 검출 프레임을 사용합니다."""
    reviewed_events = manifest_video.get("events", {})
    if (
        manifest_video.get("review_status") == "reviewed"
        and all(isinstance(reviewed_events.get(stage), int) for stage in STAGE_KEYS)
    ):
        return dict(reviewed_events), "reviewed"

    automatic_stages = automatic_events.get("stages", {})
    missing = [stage for stage in STAGE_KEYS if stage not in automatic_stages]
    if missing:
        raise ValueError(f"자동 단계 검출 결과가 없습니다: {', '.join(missing)}")
    return {
        stage: int(automatic_stages[stage]["frame_index"])
        for stage in STAGE_KEYS
    }, "automatic"


def _stage_label(stage_key):
    for stage in STAGE_CONFIGS:
        if stage["key"] == stage_key:
            return stage["label"]
    return stage_key


def _score(metrics, key):
    value = metrics.get(key)
    if value is None:
        return None
    value = float(value)
    return round(value, 2) if math.isfinite(value) else None


def evaluate_cached_video(video_id, manifest_video, landmark_payload, automatic_events):
    """한 영상의 8개 대표 프레임을 현재 웹캠 통합 판정으로 다시 평가합니다."""
    video_metadata = landmark_payload.get("video", {})
    image_width = int(video_metadata.get("width", 0))
    image_height = int(video_metadata.get("height", 0))
    if image_width <= 0 or image_height <= 0:
        raise ValueError("영상 너비와 높이 정보가 없습니다.")

    requested_frames, event_source = select_stage_frames(manifest_video, automatic_events)
    frame_lookup = detected_frame_lookup(landmark_payload)
    selected = {}
    for stage_key in STAGE_KEYS:
        frame, offset = nearest_detected_frame(frame_lookup, requested_frames[stage_key])
        selected[stage_key] = {
            "frame": frame,
            "landmarks": deserialize_landmarks(frame),
            "requested_frame_index": requested_frames[stage_key],
            "frame_offset": offset,
        }

    address_landmarks = selected["address"]["landmarks"]
    calibration_profile = create_calibration_profile(
        [address_landmarks],
        image_width,
        image_height,
    )
    if calibration_profile is None:
        raise ValueError(
            "어드레스 프레임에서 전신 보정을 만들 수 없습니다. "
            "어깨와 양쪽 발목 visibility를 확인하세요."
        )
    calibration_profile["caddieset_address_points"] = average_landmark_points(
        [address_landmarks]
    )

    stages = {}
    for stage_key in STAGE_KEYS:
        selection = selected[stage_key]
        result = analyze_stage_pose(
            stage_key,
            [selection["landmarks"]],
            calibration_profile,
            image_width,
            image_height,
        )
        metrics = result.get("metrics", {})
        stages[stage_key] = {
            "label": _stage_label(stage_key),
            "requested_frame_index": selection["requested_frame_index"],
            "frame_index": int(selection["frame"]["frame_index"]),
            "frame_offset": int(selection["frame_offset"]),
            "timestamp_ms": int(selection["frame"].get("timestamp_ms", 0)),
            "passed": bool(result.get("passed")),
            "status": result.get("status", "warning"),
            "final_score": _score(metrics, "final_score"),
            "guide_score": _score(metrics, "guide_score"),
            "caddieset_score": _score(metrics, "caddieset_score"),
            "has_outer_warning": bool(metrics.get("has_outer_warning", False)),
            "messages": list(result.get("messages", [])),
        }

    return {
        "status": "ok",
        "source": manifest_video.get("source"),
        "review_status": manifest_video.get("review_status", "pending"),
        "event_source": event_source,
        "video": video_metadata,
        "calibration": {
            "body_ratio": round(float(calibration_profile["body_ratio"]), 4),
            "shoulder_width": round(float(calibration_profile["shoulder_width"]), 2),
        },
        "stages": stages,
    }


def diagnose_stage(mean_final_score, pass_rate):
    if mean_final_score is None or pass_rate is None:
        return "insufficient"
    if (
        mean_final_score < STRICT_SCORE_THRESHOLD
        or pass_rate < STRICT_PASS_RATE_THRESHOLD
    ):
        return "strict_candidate"
    if (
        mean_final_score >= LENIENT_SCORE_THRESHOLD
        and pass_rate >= LENIENT_PASS_RATE_THRESHOLD
    ):
        return "lenient_candidate"
    return "balanced_candidate"


def summarize_runtime_regression(videos):
    stage_summary = {}
    for stage_key in STAGE_KEYS:
        records = [
            video["stages"][stage_key]
            for video in videos.values()
            if video.get("status") == "ok"
            and video.get("stages", {}).get(stage_key, {}).get("final_score") is not None
        ]
        final_scores = [record["final_score"] for record in records]
        guide_scores = [
            record["guide_score"] for record in records
            if record.get("guide_score") is not None
        ]
        caddieset_scores = [
            record["caddieset_score"] for record in records
            if record.get("caddieset_score") is not None
        ]
        pass_rate = (
            sum(record["passed"] for record in records) / len(records)
            if records
            else None
        )
        mean_final_score = (
            round(sum(final_scores) / len(final_scores), 2)
            if final_scores
            else None
        )
        stage_summary[stage_key] = {
            "label": _stage_label(stage_key),
            "sample_count": len(records),
            "mean_final_score": mean_final_score,
            "min_final_score": min(final_scores) if final_scores else None,
            "max_final_score": max(final_scores) if final_scores else None,
            "mean_guide_score": (
                round(sum(guide_scores) / len(guide_scores), 2)
                if guide_scores
                else None
            ),
            "mean_caddieset_score": (
                round(sum(caddieset_scores) / len(caddieset_scores), 2)
                if caddieset_scores
                else None
            ),
            "pass_rate": round(pass_rate, 4) if pass_rate is not None else None,
            "diagnosis": diagnose_stage(mean_final_score, pass_rate),
        }

    ok_videos = [video for video in videos.values() if video.get("status") == "ok"]
    return {
        "video_count": len(videos),
        "evaluated_video_count": len(ok_videos),
        "failed_video_count": sum(
            video.get("status") == "failed" for video in videos.values()
        ),
        "reviewed_event_video_count": sum(
            video.get("event_source") == "reviewed" for video in ok_videos
        ),
        "automatic_event_video_count": sum(
            video.get("event_source") == "automatic" for video in ok_videos
        ),
        "stages": stage_summary,
    }


def build_runtime_regression(manifest, audit_root, video_ids=None):
    """기존 stage_audit 캐시를 읽어 현재 런타임 판정 회귀 보고서를 만듭니다."""
    audit_root = Path(audit_root)
    selected_ids = list(video_ids or manifest.get("videos", {}).keys())
    unknown = set(selected_ids) - set(manifest.get("videos", {}))
    if unknown:
        raise ValueError(f"매니페스트에 없는 영상 ID입니다: {', '.join(sorted(unknown))}")

    videos = {}
    for video_id in selected_ids:
        manifest_video = manifest["videos"][video_id]
        if manifest_video.get("review_status") == "excluded":
            videos[video_id] = {
                "status": "excluded",
                "source": manifest_video.get("source"),
                "review_status": "excluded",
                "error": manifest_video.get("note", "검수에서 제외된 영상입니다."),
            }
            continue

        session_dir = audit_root / video_id
        try:
            landmarks = load_json(session_dir / "frame_landmarks.json")
            events = load_json(session_dir / "stage_events.json")
            videos[video_id] = evaluate_cached_video(
                video_id,
                manifest_video,
                landmarks,
                events,
            )
        except Exception as error:
            videos[video_id] = {
                "status": "failed",
                "source": manifest_video.get("source"),
                "review_status": manifest_video.get("review_status", "pending"),
                "error_type": type(error).__name__,
                "error": str(error),
            }

    return {
        "schema": REGRESSION_SCHEMA,
        "stage_order": list(STAGE_KEYS),
        "scope": {
            "view": manifest.get("view", "FACEON"),
            "club_type": "I7",
            "swing_hand": "right",
            "scoring": "guide_55_percent_plus_caddieset_45_percent",
        },
        "limitations": [
            "정지 자세용 판정을 풀스윙의 단일 프레임에 적용한 진단 결과입니다.",
            "pending 영상의 단계 프레임은 자동 검출값이며 코치 검수 정답이 아닙니다.",
            "strict/lenient 후보는 기준 조정 대상을 찾기 위한 신호이며 자동 기준 변경에 사용하지 않습니다.",
        ],
        "summary": summarize_runtime_regression(videos),
        "videos": videos,
    }


def render_runtime_regression_markdown(report):
    summary = report["summary"]
    lines = [
        "# Runtime Regression Report",
        "",
        "현재 웹캠 통합 판정을 기존 프로 영상의 8단계 대표 프레임에 적용한 진단 보고서입니다.",
        "자동 검출 프레임은 정답 데이터가 아니며, 이 결과만으로 판정 기준을 자동 변경하지 않습니다.",
        "",
        "## Summary",
        "",
        f"- 분석 요청 영상: {summary['video_count']}",
        f"- 평가 완료 영상: {summary['evaluated_video_count']}",
        f"- 평가 실패 영상: {summary['failed_video_count']}",
        f"- 검수 확정 프레임 사용: {summary['reviewed_event_video_count']}",
        f"- 자동 검출 프레임 사용: {summary['automatic_event_video_count']}",
        "",
        "## Stage Diagnostics",
        "",
        "| Stage | Samples | Final | Guide | I7 | Pass rate | Diagnosis |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for stage_key in STAGE_KEYS:
        stage = summary["stages"][stage_key]
        pass_rate = (
            f"{stage['pass_rate'] * 100:.1f}%"
            if stage["pass_rate"] is not None
            else "-"
        )
        lines.append(
            "| {label} | {count} | {final} | {guide} | {i7} | {rate} | {diagnosis} |".format(
                label=stage["label"],
                count=stage["sample_count"],
                final=stage["mean_final_score"]
                if stage["mean_final_score"] is not None
                else "-",
                guide=stage["mean_guide_score"]
                if stage["mean_guide_score"] is not None
                else "-",
                i7=stage["mean_caddieset_score"]
                if stage["mean_caddieset_score"] is not None
                else "-",
                rate=pass_rate,
                diagnosis=stage["diagnosis"],
            )
        )

    failed = [
        (video_id, video)
        for video_id, video in report["videos"].items()
        if video.get("status") == "failed"
    ]
    if failed:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- `{video_id}`: {video['error']}" for video_id, video in failed)

    lines.extend(
        [
            "",
            "## Diagnosis Rules",
            "",
            f"- `strict_candidate`: 평균 최종 점수 < {STRICT_SCORE_THRESHOLD:g} 또는 통과율 < {STRICT_PASS_RATE_THRESHOLD * 100:g}%",
            f"- `lenient_candidate`: 평균 최종 점수 >= {LENIENT_SCORE_THRESHOLD:g} 이고 통과율 >= {LENIENT_PASS_RATE_THRESHOLD * 100:g}%",
            "- `balanced_candidate`: 위 두 조건 사이",
            "- `insufficient`: 평가 가능한 표본 없음",
            "",
        ]
    )
    return "\n".join(lines)
