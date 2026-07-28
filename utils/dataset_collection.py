"""비식별 웹캠 데이터 수집 세션과 로컬 전용 개인정보를 관리합니다."""

import json
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "reference_data" / "webcam_dataset"
COLLECTION_SESSION_SCHEMA = "golf-coach-collection-session-v1"
PRIVATE_PARTICIPANT_SCHEMA = "golf-coach-private-participant-v1"
ACTIVE_SESSION_SCHEMA = "golf-coach-active-collection-v1"

HEIGHT_BANDS = {"under_160", "160_169", "170_179", "180_189", "190_plus", "unspecified"}
BODY_BUILDS = {"slim", "average", "athletic", "broad", "unspecified"}
MOBILITY_LEVELS = {"limited", "typical", "high", "unspecified"}
EXPERIENCE_LEVELS = {"beginner", "intermediate", "advanced", "professional", "unspecified"}
HANDEDNESS = {"right", "left"}
VIEWS = {"FACEON", "DTL"}
CLUB_TYPES = {"I7", "W1", "OTHER"}
DISTANCE_BANDS = {"near", "recommended", "far", "unspecified"}
LIGHTING_LEVELS = {"low", "normal", "bright", "mixed", "unspecified"}
BACKGROUND_TYPES = {"plain", "indoor", "outdoor", "cluttered", "unspecified"}


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _slug(value, field_name):
    value = str(value or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,39}", value):
        raise ValueError(
            f"{field_name}는 영문 소문자·숫자·밑줄·하이픈으로 된 3~40자여야 합니다."
        )
    return value


def new_participant_id():
    return f"p_{uuid4().hex[:10]}"


def new_session_id(captured_at=None):
    captured_at = captured_at or datetime.now().astimezone()
    return f"s_{captured_at.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"


def validate_body_profile(profile):
    profile = dict(profile or {})
    validated = {
        "height_band": profile.get("height_band", "unspecified"),
        "body_build": profile.get("body_build", "unspecified"),
        "mobility": profile.get("mobility", "unspecified"),
        "experience_level": profile.get("experience_level", "unspecified"),
        "handedness": profile.get("handedness", "right"),
    }
    allowed = {
        "height_band": HEIGHT_BANDS,
        "body_build": BODY_BUILDS,
        "mobility": MOBILITY_LEVELS,
        "experience_level": EXPERIENCE_LEVELS,
        "handedness": HANDEDNESS,
    }
    for field, choices in allowed.items():
        if validated[field] not in choices:
            raise ValueError(f"지원하지 않는 {field} 값입니다: {validated[field]}")
    return validated


def validate_capture_conditions(conditions):
    conditions = dict(conditions or {})
    validated = {
        "view": str(conditions.get("view", "FACEON")).upper(),
        "club_type": str(conditions.get("club_type", "I7")).upper(),
        "camera_id": str(conditions.get("camera_id", "camera_0")),
        "distance_band": conditions.get("distance_band", "unspecified"),
        "lighting": conditions.get("lighting", "unspecified"),
        "background": conditions.get("background", "unspecified"),
        "resolution": conditions.get("resolution"),
        "notes": str(conditions.get("notes", "")).strip(),
    }
    allowed = {
        "view": VIEWS,
        "club_type": CLUB_TYPES,
        "distance_band": DISTANCE_BANDS,
        "lighting": LIGHTING_LEVELS,
        "background": BACKGROUND_TYPES,
    }
    for field, choices in allowed.items():
        if validated[field] not in choices:
            raise ValueError(f"지원하지 않는 {field} 값입니다: {validated[field]}")

    resolution = validated["resolution"]
    if resolution is not None:
        if (
            not isinstance(resolution, (list, tuple))
            or len(resolution) != 2
            or any(int(value) <= 0 for value in resolution)
        ):
            raise ValueError("resolution은 양의 정수 [width, height] 형식이어야 합니다.")
        validated["resolution"] = [int(resolution[0]), int(resolution[1])]
    return validated


def create_collection_session(
    *,
    participant_id,
    body_profile,
    capture_conditions,
    consent_confirmed,
    private_profile=None,
    dataset_root=DEFAULT_DATASET_ROOT,
    session_id=None,
    created_at=None,
    activate=True,
):
    """개인정보 파일과 비식별 세션 파일을 분리해 생성합니다."""
    if not consent_confirmed:
        raise ValueError("촬영 및 자세 데이터 이용 동의가 확인되어야 합니다.")

    dataset_root = Path(dataset_root)
    participant_id = _slug(participant_id, "participant_id")
    session_id = _slug(session_id or new_session_id(created_at), "session_id")
    created_at = created_at or datetime.now().astimezone()
    body_profile = validate_body_profile(body_profile)
    capture_conditions = validate_capture_conditions(capture_conditions)

    private_payload = {
        "schema": PRIVATE_PARTICIPANT_SCHEMA,
        "participant_id": participant_id,
        "consent": {
            "confirmed": True,
            "confirmed_at": created_at.isoformat(),
        },
        "private_profile": dict(private_profile or {}),
        "storage_policy": "local_only_never_commit",
    }
    private_path = (
        dataset_root / "private" / "participants" / f"{participant_id}.json"
    )
    _write_json(private_path, private_payload)

    session_payload = {
        "schema": COLLECTION_SESSION_SCHEMA,
        "session_id": session_id,
        "participant_id": participant_id,
        "created_at": created_at.isoformat(),
        "body_profile": body_profile,
        "capture_conditions": capture_conditions,
        "privacy": {
            "contains_direct_identifiers": False,
            "raw_media_local_only": True,
            "private_profile": str(private_path.relative_to(dataset_root)),
        },
    }
    session_path = dataset_root / "sessions" / f"{session_id}.json"
    _write_json(session_path, session_payload)

    if activate:
        active_path = dataset_root / "active_session.json"
        _write_json(
            active_path,
            {
                "schema": ACTIVE_SESSION_SCHEMA,
                "session": str(session_path.relative_to(dataset_root)),
            },
        )

    return {
        **session_payload,
        "dataset_root": str(dataset_root),
        "session_path": str(session_path),
        "private_path": str(private_path),
        "capture_root": str(
            dataset_root
            / "captures"
            / participant_id
            / session_id
        ),
    }


def load_collection_session(path, *, dataset_root=None):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != COLLECTION_SESSION_SCHEMA:
        raise ValueError(f"지원하지 않는 수집 세션 형식입니다: {path}")
    payload["body_profile"] = validate_body_profile(payload.get("body_profile"))
    payload["capture_conditions"] = validate_capture_conditions(
        payload.get("capture_conditions")
    )
    root = Path(dataset_root) if dataset_root else path.parent.parent
    return {
        **payload,
        "dataset_root": str(root),
        "session_path": str(path),
        "capture_root": str(
            root
            / "captures"
            / payload["participant_id"]
            / payload["session_id"]
        ),
    }


def load_active_collection_context(
    dataset_root=DEFAULT_DATASET_ROOT,
    *,
    required=False,
):
    dataset_root = Path(dataset_root)
    active_path = dataset_root / "active_session.json"
    if not active_path.exists():
        if required:
            raise FileNotFoundError(
                "활성 웹캠 수집 세션이 없습니다. configure_webcam_collection.py를 먼저 실행하세요."
            )
        return None

    active = json.loads(active_path.read_text(encoding="utf-8"))
    if active.get("schema") != ACTIVE_SESSION_SCHEMA:
        raise ValueError(f"지원하지 않는 활성 세션 형식입니다: {active_path}")
    session_path = dataset_root / active["session"]
    return load_collection_session(session_path, dataset_root=dataset_root)


def deactivate_collection_session(dataset_root=DEFAULT_DATASET_ROOT):
    active_path = Path(dataset_root) / "active_session.json"
    if active_path.exists():
        active_path.unlink()
        return True
    return False
