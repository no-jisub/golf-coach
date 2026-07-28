"""웹캠 진단 프레임과 판정 근거를 로컬 전용 샘플로 저장합니다."""

import json
import math
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE_ROOT = PROJECT_ROOT / "reference_data" / "runtime_samples"
CAPTURE_SCHEMA = "golf-coach-runtime-sample-v1"
COLLECTION_CAPTURE_SCHEMA = "golf-coach-runtime-sample-v2"
EXPECTED_LABELS = {"pending", "expected_pass", "expected_fail"}


def serialize_landmarks(landmarks):
    serialized = []
    for index, landmark in enumerate(landmarks or []):
        item = {
            "index": index,
            "x": round(float(getattr(landmark, "x", 0.0)), 7),
            "y": round(float(getattr(landmark, "y", 0.0)), 7),
            "z": round(float(getattr(landmark, "z", 0.0)), 7),
        }
        for field in ("visibility", "presence"):
            value = getattr(landmark, field, None)
            if value is not None and math.isfinite(float(value)):
                item[field] = round(float(value), 7)
        serialized.append(item)
    return serialized


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def classify_discrepancy(expected_label, feedback):
    actual_passed = None if feedback is None else bool(feedback.get("passed"))
    if expected_label == "expected_pass" and actual_passed is False:
        return "false_reject"
    if expected_label == "expected_fail" and actual_passed is True:
        return "false_accept"
    if expected_label == "pending" or actual_passed is None:
        return "unreviewed"
    return "match"


def save_runtime_sample(
    *,
    raw_frame,
    overlay_frame,
    stage_key,
    landmarks,
    diagnostics,
    feedback,
    expected_label="pending",
    output_root=None,
    captured_at=None,
    collection_context=None,
):
    """화면과 구조화된 판정 정보를 충돌 없는 로컬 폴더에 저장합니다."""
    if expected_label not in EXPECTED_LABELS:
        raise ValueError(f"지원하지 않는 기대 라벨입니다: {expected_label}")
    if raw_frame is None or overlay_frame is None:
        raise ValueError("저장할 원본 프레임과 오버레이 프레임이 필요합니다.")

    if collection_context is None and output_root is None:
        from utils.dataset_collection import load_active_collection_context

        collection_context = load_active_collection_context()

    captured_at = captured_at or datetime.now().astimezone()
    sample_id = (
        captured_at.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        + "_"
        + uuid4().hex[:6]
    )
    if collection_context:
        required_context = {
            "participant_id",
            "session_id",
            "body_profile",
            "capture_conditions",
            "capture_root",
        }
        missing_context = required_context - set(collection_context)
        if missing_context:
            raise ValueError(
                "수집 세션 정보가 부족합니다: " + ", ".join(sorted(missing_context))
            )
        sample_dir = Path(collection_context["capture_root"]) / stage_key / sample_id
        capture_schema = COLLECTION_CAPTURE_SCHEMA
    else:
        sample_dir = Path(output_root or DEFAULT_CAPTURE_ROOT) / stage_key / sample_id
        capture_schema = CAPTURE_SCHEMA
    sample_dir.mkdir(parents=True, exist_ok=False)

    raw_path = sample_dir / "raw.jpg"
    overlay_path = sample_dir / "overlay.jpg"
    metadata_path = sample_dir / "sample.json"
    if not cv2.imwrite(str(raw_path), raw_frame):
        raise OSError(f"원본 프레임을 저장하지 못했습니다: {raw_path}")
    if not cv2.imwrite(str(overlay_path), overlay_frame):
        raise OSError(f"오버레이 프레임을 저장하지 못했습니다: {overlay_path}")

    discrepancy = classify_discrepancy(expected_label, feedback)
    metadata = {
        "schema": capture_schema,
        "sample_id": sample_id,
        "captured_at": captured_at.isoformat(),
        "local_only": True,
        "stage_key": stage_key,
        "expected_label": expected_label,
        "actual_passed": None if feedback is None else bool(feedback.get("passed")),
        "discrepancy": discrepancy,
        "landmarks": serialize_landmarks(landmarks),
        "diagnostics": _json_safe(diagnostics),
        "feedback": _json_safe(feedback),
        "artifacts": {
            "raw_frame": raw_path.name,
            "overlay_frame": overlay_path.name,
        },
    }
    if collection_context:
        metadata["collection"] = {
            "participant_id": collection_context["participant_id"],
            "session_id": collection_context["session_id"],
            "body_profile": _json_safe(collection_context["body_profile"]),
            "capture_conditions": _json_safe(
                collection_context["capture_conditions"]
            ),
            "contains_direct_identifiers": False,
        }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        **metadata,
        "sample_dir": str(sample_dir),
        "metadata_path": str(metadata_path),
    }
