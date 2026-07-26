from copy import deepcopy
from functools import lru_cache

from utils.app_config import MVP_CLUB_TYPE, MVP_VIEW
from utils.caddieset_evaluator import (
    load_evaluation_profiles,
    select_stage_evaluation_items,
)
from utils.caddieset_metrics import calculate_pose_metrics
from utils.guide_alignment import METRIC_JOINTS, STAGE_KEYS
from utils.guide_skeleton import GUIDE_POSES, SWING_HAND


DEFAULT_MAX_OFFSET = 0.06
DEFAULT_SAMPLE_STEP = 0.0025


def _metrics_within_reference(metrics, evaluation_items, metric_keys):
    for metric_key in metric_keys:
        value = metrics.get(metric_key)
        low, high = evaluation_items[metric_key]["observed_reference_range"]
        if value is None or value < min(low, high) or value > max(low, high):
            return False
    return True


def _probe_axis_bound(
    stage_key,
    pose,
    address_pose,
    joint_index,
    axis,
    direction,
    metric_keys,
    evaluation_items,
    direction_multiplier,
    max_offset,
    sample_step,
):
    origin = pose[joint_index]
    last_valid = origin[axis]
    sample_count = max(1, round(max_offset / sample_step))
    for sample_index in range(1, sample_count + 1):
        offset = min(sample_index * sample_step, max_offset) * direction
        value = origin[axis] + offset
        if not 0.0 <= value <= 1.0:
            break

        candidate = deepcopy(pose)
        point = list(candidate[joint_index])
        point[axis] = value
        candidate[joint_index] = tuple(point)
        metrics = calculate_pose_metrics(
            candidate,
            address_points=address_pose,
            direction_multiplier=direction_multiplier,
        )
        if not _metrics_within_reference(metrics, evaluation_items, metric_keys):
            break
        last_valid = value
    return last_valid


def calculate_stage_tolerance_regions(
    stage_key,
    guide_poses=GUIDE_POSES,
    *,
    profile_data=None,
    view=MVP_VIEW,
    club_type=MVP_CLUB_TYPE,
    direction_multiplier=None,
    max_offset=DEFAULT_MAX_OFFSET,
    sample_step=DEFAULT_SAMPLE_STEP,
):
    """Estimate each assessed joint's local CaddieSet-pass region around the guide."""
    if stage_key not in STAGE_KEYS:
        raise ValueError(f"지원하지 않는 스윙 단계입니다: {stage_key}")
    if max_offset <= 0.0 or sample_step <= 0.0:
        raise ValueError("허용 범위 탐색 간격과 최대 거리는 0보다 커야 합니다.")
    if profile_data is None:
        profile_data = load_evaluation_profiles()
    if direction_multiplier is None:
        direction_multiplier = -1.0 if SWING_HAND == "right" else 1.0

    selection = select_stage_evaluation_items(
        stage_key,
        data=profile_data,
        view=view,
        club_type=club_type,
    )
    evaluation_items = selection["evaluation_items"]
    pose = guide_poses[stage_key]
    address_pose = guide_poses["address"]
    regions = {}

    for joint_index, point in pose.items():
        metric_keys = tuple(
            metric_key
            for metric_key in evaluation_items
            if joint_index in METRIC_JOINTS.get(metric_key, set())
        )
        if not metric_keys:
            continue

        bounds = []
        for axis in (0, 1):
            low = _probe_axis_bound(
                stage_key,
                pose,
                address_pose,
                joint_index,
                axis,
                -1,
                metric_keys,
                evaluation_items,
                direction_multiplier,
                max_offset,
                sample_step,
            )
            high = _probe_axis_bound(
                stage_key,
                pose,
                address_pose,
                joint_index,
                axis,
                1,
                metric_keys,
                evaluation_items,
                direction_multiplier,
                max_offset,
                sample_step,
            )
            bounds.append((min(low, point[axis]), max(high, point[axis])))

        center = (
            (bounds[0][0] + bounds[0][1]) / 2.0,
            (bounds[1][0] + bounds[1][1]) / 2.0,
        )
        radius = (
            (bounds[0][1] - bounds[0][0]) / 2.0,
            (bounds[1][1] - bounds[1][0]) / 2.0,
        )
        regions[joint_index] = {
            "center": center,
            "radius": radius,
            "bounds": {
                "x": bounds[0],
                "y": bounds[1],
            },
            "metric_keys": metric_keys,
            "sample_step": sample_step,
        }
    return regions


@lru_cache(maxsize=len(STAGE_KEYS))
def get_stage_tolerance_regions(stage_key):
    return calculate_stage_tolerance_regions(stage_key)
