import math

from utils.caddieset_evaluator import (
    classify_stage_comparisons,
    compare_stage_metrics,
    load_evaluation_profiles,
    select_stage_evaluation_items,
)
from utils.caddieset_metrics import calculate_pose_metrics


STAGE_KEYS = (
    "address",
    "takeaway",
    "backswing",
    "top",
    "downswing",
    "impact",
    "follow_through",
    "finish",
)


def validate_guide_poses(guide_poses):
    missing = [stage for stage in STAGE_KEYS if stage not in guide_poses]
    if missing:
        raise ValueError(f"가이드 스켈레톤 단계가 빠져 있습니다: {', '.join(missing)}")
    if not guide_poses.get("address"):
        raise ValueError("상대 지표 계산에 필요한 어드레스 가이드가 없습니다.")


def calculate_guide_stage_metrics(guide_poses, direction_multiplier=1.0):
    """런타임 8단계 스켈레톤을 CaddieSet 대응 지표로 변환합니다."""
    validate_guide_poses(guide_poses)
    address_points = guide_poses["address"]
    stage_metrics = {}
    for stage_key in STAGE_KEYS:
        metrics = calculate_pose_metrics(
            guide_poses[stage_key],
            address_points=address_points,
            direction_multiplier=direction_multiplier,
        )
        stage_metrics[stage_key] = {
            metric_key: round(float(value), 6)
            if value is not None and math.isfinite(value)
            else None
            for metric_key, value in metrics.items()
        }
    return stage_metrics


def audit_guide_stage_metrics(
    stage_metrics,
    *,
    profile_data=None,
    view="FACEON",
    club_type=None,
):
    """가이드 지표가 CaddieSet 단계별 관찰 범위에 들어오는지 검사합니다."""
    if profile_data is None:
        profile_data = load_evaluation_profiles()

    stage_results = {}
    total_summary = {
        "total_count": 0,
        "pass_count": 0,
        "warning_count": 0,
        "unavailable_count": 0,
        "outside_reference_count": 0,
        "outside_observed_count": 0,
    }
    for stage_key in STAGE_KEYS:
        selection = select_stage_evaluation_items(
            stage_key,
            data=profile_data,
            view=view,
            club_type=club_type,
        )
        comparison = compare_stage_metrics(stage_metrics[stage_key], selection)
        classified = classify_stage_comparisons(comparison)
        stage_results[stage_key] = classified

        summary = classified["summary"]
        for key in ("total_count", "pass_count", "warning_count", "unavailable_count"):
            total_summary[key] += summary[key]
        total_summary["outside_reference_count"] += sum(
            item["warning_level"] == "outside_reference"
            for item in classified["comparisons"].values()
        )
        total_summary["outside_observed_count"] += sum(
            item["warning_level"] == "outside_observed"
            for item in classified["comparisons"].values()
        )

    total_summary["aligned_stage_count"] = sum(
        result["overall_status"] == "pass" for result in stage_results.values()
    )
    total_summary["stage_count"] = len(stage_results)
    return {
        "view": view,
        "club_type": club_type or "ALL",
        "summary": total_summary,
        "stages": stage_results,
    }
