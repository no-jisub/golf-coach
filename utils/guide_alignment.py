import math

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

