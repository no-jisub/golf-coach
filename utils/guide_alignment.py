import math
from copy import deepcopy

from utils.caddieset_evaluator import (
    classify_stage_comparisons,
    compare_stage_metrics,
    load_evaluation_profiles,
    select_stage_evaluation_items,
)
from utils.caddieset_metrics import calculate_pose_metrics
from utils.guide_skeleton import (
    GUIDE_CONNECTIONS,
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)


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

METRIC_JOINTS = {
    "shoulder_angle": {LEFT_SHOULDER, RIGHT_SHOULDER},
    "spine_angle": {LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP},
    "stance_ratio": {LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ANKLE, RIGHT_ANKLE},
    "upper_tilt": {
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_HIP,
        RIGHT_HIP,
        LEFT_ANKLE,
        RIGHT_ANKLE,
    },
    "head_loc": {NOSE},
    "hip_line": {LEFT_HIP, RIGHT_HIP},
    "hip_rotation": {LEFT_HIP, RIGHT_HIP},
    "hip_shifted": {LEFT_HIP, RIGHT_HIP},
    "left_arm_angle": {LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST},
    "right_arm_angle": {RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST},
    "shoulder_loc": {LEFT_SHOULDER, LEFT_ANKLE, RIGHT_ANKLE},
    "hip_angle": {LEFT_HIP, RIGHT_HIP},
    "left_leg_angle": {LEFT_HIP, LEFT_KNEE, LEFT_ANKLE},
    "right_leg_angle": {RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE},
    "right_distance": {RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_HIP},
    "hip_hanging_back": {LEFT_HIP, LEFT_ANKLE, RIGHT_ANKLE},
    "right_armpit_angle": {RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP},
    "shoulder_hanging_back": {LEFT_SHOULDER, LEFT_ANKLE, RIGHT_ANKLE},
    "weight_shift": {LEFT_HIP, LEFT_ANKLE},
    "finish_angle": {RIGHT_HIP, LEFT_ANKLE},
}

DEFAULT_OPTIMIZATION_STEPS = (0.04, 0.02, 0.01, 0.005, 0.0025)
MAX_JOINT_DISPLACEMENT = 0.12
MIN_COORDINATE = 0.02
MAX_COORDINATE = 0.98


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


def _distance(point_a, point_b):
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def _range_violation(value, item):
    low, high = item["observed_reference_range"]
    span = max(high - low, 1e-9)
    if value is None:
        return 10.0
    if value < low:
        return (low - value) / span
    if value > high:
        return (value - high) / span
    return 0.0


def _pose_regularization(pose, original_pose):
    movement = sum(
        _distance(pose[index], original_pose[index]) ** 2 for index in pose
    )
    bone_change = 0.0
    for start_index, end_index in GUIDE_CONNECTIONS:
        original_length = _distance(original_pose[start_index], original_pose[end_index])
        if original_length <= 1e-9:
            continue
        current_length = _distance(pose[start_index], pose[end_index])
        bone_change += ((current_length - original_length) / original_length) ** 2
    return movement + bone_change * 0.015


def stage_alignment_objective(
    stage_key,
    pose,
    address_pose,
    evaluation_items,
    original_pose,
    direction_multiplier,
):
    metrics = calculate_pose_metrics(
        pose,
        address_points=address_pose,
        direction_multiplier=direction_multiplier,
    )
    violations = [
        _range_violation(metrics.get(metric_key), item)
        for metric_key, item in evaluation_items.items()
    ]
    warning_count = sum(violation > 1e-9 for violation in violations)
    violation_distance = sum(violation * violation for violation in violations)
    regularization = _pose_regularization(pose, original_pose)
    return (
        warning_count,
        round(violation_distance, 12),
        round(regularization, 12),
    )


def _candidate_coordinate(original_value, current_value, delta):
    lower = max(MIN_COORDINATE, original_value - MAX_JOINT_DISPLACEMENT)
    upper = min(MAX_COORDINATE, original_value + MAX_JOINT_DISPLACEMENT)
    return max(lower, min(upper, current_value + delta))


def optimize_stage_pose(
    stage_key,
    pose,
    address_pose,
    evaluation_items,
    direction_multiplier,
    steps=DEFAULT_OPTIMIZATION_STEPS,
):
    """CaddieSet 범위 이탈을 줄이면서 원본에 가까운 단계 자세를 찾습니다."""
    original_pose = deepcopy(pose)
    optimized = deepcopy(pose)
    adjustable_joints = sorted(
        set().union(
            *(METRIC_JOINTS.get(metric_key, set()) for metric_key in evaluation_items)
        )
    )
    best_objective = stage_alignment_objective(
        stage_key,
        optimized,
        address_pose,
        evaluation_items,
        original_pose,
        direction_multiplier,
    )

    for step in steps:
        improved = True
        pass_count = 0
        while improved and pass_count < 12:
            improved = False
            pass_count += 1
            for index in adjustable_joints:
                for axis in (0, 1):
                    current_point = optimized[index]
                    best_point = current_point
                    local_best = best_objective
                    for delta in (-step, step):
                        candidate_value = _candidate_coordinate(
                            original_pose[index][axis],
                            current_point[axis],
                            delta,
                        )
                        if abs(candidate_value - current_point[axis]) <= 1e-12:
                            continue
                        candidate = deepcopy(optimized)
                        point = list(candidate[index])
                        point[axis] = candidate_value
                        candidate[index] = tuple(point)
                        objective = stage_alignment_objective(
                            stage_key,
                            candidate,
                            address_pose,
                            evaluation_items,
                            original_pose,
                            direction_multiplier,
                        )
                        if objective < local_best:
                            local_best = objective
                            best_point = candidate[index]
                    if best_point != current_point:
                        optimized[index] = best_point
                        best_objective = local_best
                        improved = True

    return optimized, {
        "before_objective": list(
            stage_alignment_objective(
                stage_key,
                original_pose,
                address_pose if stage_key != "address" else original_pose,
                evaluation_items,
                original_pose,
                direction_multiplier,
            )
        ),
        "after_objective": list(best_objective),
        "adjustable_joints": adjustable_joints,
        "max_joint_displacement": max(
            _distance(optimized[index], original_pose[index]) for index in optimized
        ),
    }


def align_guide_poses_to_caddieset(
    guide_poses,
    *,
    profile_data=None,
    view="FACEON",
    club_type=None,
    direction_multiplier=1.0,
):
    """8단계 가이드 좌표를 CaddieSet 참조 범위에 최소 변형으로 정렬합니다."""
    validate_guide_poses(guide_poses)
    if profile_data is None:
        profile_data = load_evaluation_profiles()

    aligned = deepcopy(guide_poses)
    optimization = {}
    for stage_key in STAGE_KEYS:
        selection = select_stage_evaluation_items(
            stage_key,
            data=profile_data,
            view=view,
            club_type=club_type,
        )
        address_pose = aligned["address"]
        optimized, stage_report = optimize_stage_pose(
            stage_key,
            aligned[stage_key],
            address_pose,
            selection["evaluation_items"],
            direction_multiplier,
        )
        aligned[stage_key] = optimized
        optimization[stage_key] = stage_report

    before_metrics = calculate_guide_stage_metrics(
        guide_poses,
        direction_multiplier=direction_multiplier,
    )
    after_metrics = calculate_guide_stage_metrics(
        aligned,
        direction_multiplier=direction_multiplier,
    )
    before_audit = audit_guide_stage_metrics(
        before_metrics,
        profile_data=profile_data,
        view=view,
        club_type=club_type,
    )
    after_audit = audit_guide_stage_metrics(
        after_metrics,
        profile_data=profile_data,
        view=view,
        club_type=club_type,
    )
    return aligned, {
        "before": before_audit["summary"],
        "after": after_audit["summary"],
        "stages": optimization,
    }
