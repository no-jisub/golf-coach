import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = (
    PROJECT_ROOT / "reference_data" / "caddieset" / "evaluation_profiles.json"
)
SUPPORTED_SCHEMA = "golf-coach-caddieset-evaluation-v1"
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


class CaddieSetProfileError(ValueError):
    pass


def load_evaluation_profiles(profile_path=DEFAULT_PROFILE_PATH):
    profile_path = Path(profile_path)
    if not profile_path.exists():
        raise CaddieSetProfileError(f"CaddieSet 평가 프로필이 없습니다: {profile_path}")

    data = json.loads(profile_path.read_text(encoding="utf-8"))
    if data.get("schema") != SUPPORTED_SCHEMA:
        raise CaddieSetProfileError(
            f"지원하지 않는 CaddieSet 프로필 스키마입니다: {data.get('schema')}"
        )
    if not isinstance(data.get("profiles"), dict) or not data["profiles"]:
        raise CaddieSetProfileError("CaddieSet 평가 프로필에 profiles가 없습니다.")
    return data


def make_profile_id(view="FACEON", club_type=None):
    normalized_view = str(view).strip().lower()
    normalized_club = str(club_type).strip().lower() if club_type else "all"
    return f"{normalized_view}_{normalized_club}"


def select_evaluation_profile(data, view="FACEON", club_type=None, allow_fallback=True):
    profiles = data.get("profiles", {})
    requested_id = make_profile_id(view, club_type)
    profile = profiles.get(requested_id)
    selected_id = requested_id
    used_fallback = False

    if profile is None and club_type and allow_fallback:
        selected_id = make_profile_id(view, None)
        profile = profiles.get(selected_id)
        used_fallback = profile is not None

    if profile is None:
        raise CaddieSetProfileError(
            f"요청한 CaddieSet 평가 프로필이 없습니다: {requested_id}"
        )

    missing_stages = [stage for stage in STAGE_KEYS if stage not in profile.get("stages", {})]
    if missing_stages:
        raise CaddieSetProfileError(
            f"CaddieSet 평가 프로필에 단계가 빠져 있습니다: {', '.join(missing_stages)}"
        )

    return {
        "profile_id": selected_id,
        "requested_profile_id": requested_id,
        "used_fallback": used_fallback,
        "profile": profile,
    }


def select_stage_evaluation_items(
    stage_key,
    *,
    data=None,
    profile_path=DEFAULT_PROFILE_PATH,
    view="FACEON",
    club_type=None,
    allow_fallback=True,
):
    if stage_key not in STAGE_KEYS:
        raise CaddieSetProfileError(f"지원하지 않는 스윙 단계입니다: {stage_key}")

    if data is None:
        data = load_evaluation_profiles(profile_path)
    selection = select_evaluation_profile(
        data,
        view=view,
        club_type=club_type,
        allow_fallback=allow_fallback,
    )
    stage = selection["profile"]["stages"][stage_key]
    items = stage.get("evaluation_items")
    if not isinstance(items, dict) or not items:
        raise CaddieSetProfileError(
            f"현재 단계에 CaddieSet 평가 항목이 없습니다: {stage_key}"
        )

    return {
        "stage_key": stage_key,
        "source_stage_index": stage.get("source_stage_index"),
        "profile_id": selection["profile_id"],
        "requested_profile_id": selection["requested_profile_id"],
        "used_fallback": selection["used_fallback"],
        "view": selection["profile"].get("view"),
        "club_type": selection["profile"].get("club_type"),
        "evaluation_items": items,
    }


def _valid_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validated_range(item, range_key):
    values = item.get(range_key)
    if (
        not isinstance(values, list)
        or len(values) != 2
        or not all(_valid_number(value) for value in values)
    ):
        raise CaddieSetProfileError(
            f"평가 항목의 {range_key}가 올바르지 않습니다: {item.get('source_column')}"
        )
    return min(values), max(values)


def compare_metric_value(metric_key, measured_value, item):
    """한 측정값과 CaddieSet 관찰 범위의 수치 관계를 계산합니다."""
    reference_low, reference_high = _validated_range(item, "observed_reference_range")
    outer_low, outer_high = _validated_range(item, "observed_outer_range")
    target = item.get("target")
    if not _valid_number(target):
        raise CaddieSetProfileError(
            f"평가 항목의 target이 올바르지 않습니다: {item.get('source_column')}"
        )

    base = {
        "metric_key": metric_key,
        "source_column": item.get("source_column"),
        "description": item.get("description", metric_key),
        "unit": item.get("unit", "unknown"),
        "target": target,
        "reference_range": [reference_low, reference_high],
        "outer_range": [outer_low, outer_high],
    }
    if not _valid_number(measured_value):
        return {
            **base,
            "measured_value": None,
            "relation": "unavailable",
            "delta_to_target": None,
            "normalized_delta": None,
        }

    if measured_value < outer_low:
        relation = "below_outer"
    elif measured_value < reference_low:
        relation = "below_reference"
    elif measured_value <= reference_high:
        relation = "within_reference"
    elif measured_value <= outer_high:
        relation = "above_reference"
    else:
        relation = "above_outer"

    delta_to_target = measured_value - target
    half_span = max((reference_high - reference_low) / 2.0, 1e-9)
    return {
        **base,
        "measured_value": measured_value,
        "relation": relation,
        "delta_to_target": delta_to_target,
        "normalized_delta": delta_to_target / half_span,
    }


def compare_stage_metrics(measured_metrics, stage_selection):
    """현재 단계에 선택된 항목만 사용자 지표와 비교합니다."""
    comparisons = {}
    for metric_key, item in stage_selection["evaluation_items"].items():
        comparisons[metric_key] = compare_metric_value(
            metric_key,
            measured_metrics.get(metric_key),
            item,
        )

    return {
        "stage_key": stage_selection["stage_key"],
        "profile_id": stage_selection["profile_id"],
        "view": stage_selection["view"],
        "club_type": stage_selection["club_type"],
        "used_profile_fallback": stage_selection["used_fallback"],
        "comparisons": comparisons,
    }


def classify_metric_comparison(comparison):
    """수치 관계를 항목별 통과·주의·측정 불가 상태로 변환합니다."""
    relation = comparison.get("relation")
    if relation == "within_reference":
        status = "pass"
        warning_level = None
    elif relation in {"below_reference", "above_reference"}:
        status = "warning"
        warning_level = "outside_reference"
    elif relation in {"below_outer", "above_outer"}:
        status = "warning"
        warning_level = "outside_observed"
    elif relation == "unavailable":
        status = "unavailable"
        warning_level = None
    else:
        raise CaddieSetProfileError(f"알 수 없는 지표 비교 관계입니다: {relation}")

    return {
        **comparison,
        "status": status,
        "warning_level": warning_level,
    }


def classify_stage_comparisons(stage_comparison, minimum_measurement_ratio=0.6):
    """항목별 상태와 단계 전체 상태를 계산합니다."""
    if not 0.0 <= minimum_measurement_ratio <= 1.0:
        raise ValueError("minimum_measurement_ratio는 0~1 사이여야 합니다.")

    classified = {
        metric_key: classify_metric_comparison(comparison)
        for metric_key, comparison in stage_comparison.get("comparisons", {}).items()
    }
    total_count = len(classified)
    pass_count = sum(item["status"] == "pass" for item in classified.values())
    warning_count = sum(item["status"] == "warning" for item in classified.values())
    unavailable_count = sum(item["status"] == "unavailable" for item in classified.values())
    measured_count = pass_count + warning_count
    measurement_ratio = measured_count / total_count if total_count else 0.0

    if total_count == 0 or measurement_ratio < minimum_measurement_ratio:
        overall_status = "unavailable"
    elif warning_count > 0 or unavailable_count > 0:
        overall_status = "warning"
    else:
        overall_status = "pass"

    return {
        **stage_comparison,
        "overall_status": overall_status,
        "passed": overall_status == "pass",
        "summary": {
            "total_count": total_count,
            "measured_count": measured_count,
            "pass_count": pass_count,
            "warning_count": warning_count,
            "unavailable_count": unavailable_count,
            "measurement_ratio": measurement_ratio,
            "minimum_measurement_ratio": minimum_measurement_ratio,
        },
        "comparisons": classified,
    }
