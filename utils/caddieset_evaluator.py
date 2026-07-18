import json
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

