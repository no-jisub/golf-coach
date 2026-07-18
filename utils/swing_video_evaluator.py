import math

from utils.caddieset_evaluator import (
    classify_stage_comparisons,
    compare_stage_metrics,
    load_evaluation_profiles,
    select_stage_evaluation_items,
)
from utils.caddieset_metrics import calculate_pose_metrics
from utils.golf_rules import build_caddieset_messages
from utils.guide_alignment import STAGE_KEYS
from utils.guide_skeleton import SWING_HAND


ANALYSIS_SCHEMA = "golf-coach-swing-video-analysis-v1"


def frame_points(frame_record, min_visibility=0.5):
    points = {}
    for landmark in frame_record.get("landmarks", []):
        if landmark.get("visibility", 1.0) < min_visibility:
            continue
        x = landmark.get("x")
        y = landmark.get("y")
        if x is None or y is None or not math.isfinite(float(x)) or not math.isfinite(float(y)):
            continue
        points[int(landmark["index"])] = (float(x), float(y))
    return points


def _frame_by_index(landmark_payload):
    return {
        int(frame["frame_index"]): frame
        for frame in landmark_payload.get("frames", [])
        if frame.get("detected")
    }


def _serialize_points(points):
    return {
        str(index): {"x": round(point[0], 7), "y": round(point[1], 7)}
        for index, point in sorted(points.items())
    }


def _serialize_metrics(metrics):
    return {
        metric_key: round(float(value), 6)
        if value is not None and math.isfinite(float(value))
        else None
        for metric_key, value in metrics.items()
    }


def evaluate_detected_stages(
    landmark_payload,
    stage_detection,
    *,
    swing_hand=SWING_HAND,
    view="FACEON",
    club_type=None,
    profile_data=None,
):
    """Calculate and classify CaddieSet metrics for the eight selected frames."""
    if profile_data is None:
        profile_data = load_evaluation_profiles()
    frame_lookup = _frame_by_index(landmark_payload)
    selected = {}
    for stage_key in STAGE_KEYS:
        stage = stage_detection.get("stages", {}).get(stage_key)
        if stage is None:
            raise ValueError(f"자동 추출 결과에 단계가 없습니다: {stage_key}")
        frame_index = int(stage["frame_index"])
        frame = frame_lookup.get(frame_index)
        if frame is None:
            raise ValueError(f"{stage_key} 대표 프레임의 관절 좌표가 없습니다: {frame_index}")
        selected[stage_key] = (frame, frame_points(frame))

    address_points = selected["address"][1]
    direction_multiplier = -1.0 if swing_hand == "right" else 1.0
    if swing_hand not in {"right", "left"}:
        raise ValueError(f"지원하지 않는 스윙 방향입니다: {swing_hand}")

    stages = {}
    total_summary = {
        "total_count": 0,
        "pass_count": 0,
        "warning_count": 0,
        "unavailable_count": 0,
        "passed_stage_count": 0,
    }
    profile_id = None
    for stage_key in STAGE_KEYS:
        frame, points = selected[stage_key]
        metrics = calculate_pose_metrics(
            points,
            address_points=address_points,
            direction_multiplier=direction_multiplier,
        )
        selection = select_stage_evaluation_items(
            stage_key,
            data=profile_data,
            view=view,
            club_type=club_type,
        )
        compared = compare_stage_metrics(metrics, selection)
        classified = classify_stage_comparisons(compared)
        messages = build_caddieset_messages(stage_key, classified)
        profile_id = profile_id or classified["profile_id"]
        summary = classified["summary"]
        for key in ("total_count", "pass_count", "warning_count", "unavailable_count"):
            total_summary[key] += summary[key]
        total_summary["passed_stage_count"] += int(classified["passed"])

        stages[stage_key] = {
            "frame_index": int(frame["frame_index"]),
            "timestamp_ms": int(frame["timestamp_ms"]),
            "confidence": stage_detection["stages"][stage_key].get("confidence"),
            "landmarks": _serialize_points(points),
            "metrics": _serialize_metrics(metrics),
            "evaluation": {
                "status": classified["overall_status"],
                "passed": classified["passed"],
                "summary": summary,
                "items": classified["comparisons"],
                "messages": messages,
            },
        }

    total_summary["stage_count"] = len(STAGE_KEYS)
    return {
        "schema": ANALYSIS_SCHEMA,
        "source_video": landmark_payload.get("source_video"),
        "stage_detection_method": stage_detection.get("method"),
        "profile_id": profile_id,
        "view": view,
        "club_type": club_type or "ALL",
        "swing_hand": swing_hand,
        "direction_multiplier": direction_multiplier,
        "stage_order": list(STAGE_KEYS),
        "summary": total_summary,
        "stages": stages,
    }
