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
REPRODUCIBLE_CAPTURE_SCHEMA = "golf-coach-runtime-sample-v3"
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


def serialize_timed_landmark_samples(landmark_samples, fallback_landmarks=None):
    """monotonic 시간 기반 판정 구간을 마지막 프레임 기준 상대 시간으로 직렬화합니다."""
    samples = list(landmark_samples or [])
    if not samples and fallback_landmarks is not None:
        samples = [(0.0, fallback_landmarks)]
    if not samples:
        return []

    reference_time = float(samples[-1][0])
    return [
        {
            "sample_index": index,
            "offset_ms": round((float(sample[0]) - reference_time) * 1000.0, 3),
            "landmarks": serialize_landmarks(sample[1]),
        }
        for index, sample in enumerate(samples)
    ]


def average_serialized_landmarks(serialized_samples, min_visibility=0.5):
    """실제 판정과 같은 visibility 기준으로 시간창의 평균 관절 좌표를 만듭니다."""
    by_index = {}
    for sample in serialized_samples:
        for landmark in sample.get("landmarks", []):
            if float(landmark.get("visibility", 1.0)) < min_visibility:
                continue
            by_index.setdefault(int(landmark["index"]), []).append(landmark)

    averaged = []
    for index in sorted(by_index):
        records = by_index[index]
        item = {"index": index}
        for field in ("x", "y", "z", "visibility", "presence"):
            item[field] = round(
                sum(float(record.get(field, 0.0)) for record in records)
                / len(records),
                7,
            )
        averaged.append(item)
    return averaged


def _write_decision_frames(sample_dir, decision_frames):
    """메모리에 보관한 JPEG bytes 또는 ndarray 대표 프레임을 로컬에 저장합니다."""
    records = list(decision_frames or [])
    if not records:
        return []
    reference_time = float(records[-1][0])
    frame_dir = sample_dir / "decision_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for index, record in enumerate(records):
        timestamp = float(record[0])
        frame = record[1]
        path = frame_dir / f"frame_{index:03d}.jpg"
        if isinstance(frame, (bytes, bytearray, memoryview)):
            path.write_bytes(bytes(frame))
        elif not cv2.imwrite(str(path), frame):
            raise OSError(f"판정 프레임을 저장하지 못했습니다: {path}")
        artifacts.append(
            {
                "sample_index": index,
                "offset_ms": round((timestamp - reference_time) * 1000.0, 3),
                "path": str(Path("decision_frames") / path.name),
            }
        )
    return artifacts


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
    landmark_samples=None,
    decision_frames=None,
    calibration_profile=None,
    runtime_provenance=None,
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
    reproducible = (
        landmark_samples is not None
        and calibration_profile is not None
        and runtime_provenance is not None
    )
    if reproducible:
        capture_schema = REPRODUCIBLE_CAPTURE_SCHEMA
    sample_dir.mkdir(parents=True, exist_ok=False)

    raw_path = sample_dir / "raw.jpg"
    overlay_path = sample_dir / "overlay.jpg"
    metadata_path = sample_dir / "sample.json"
    if not cv2.imwrite(str(raw_path), raw_frame):
        raise OSError(f"원본 프레임을 저장하지 못했습니다: {raw_path}")
    if not cv2.imwrite(str(overlay_path), overlay_frame):
        raise OSError(f"오버레이 프레임을 저장하지 못했습니다: {overlay_path}")

    decision_frame_artifacts = _write_decision_frames(
        sample_dir,
        decision_frames,
    )
    serialized_window = serialize_timed_landmark_samples(
        landmark_samples,
        fallback_landmarks=landmarks,
    )

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
            "decision_frames": decision_frame_artifacts,
        },
    }
    if reproducible:
        duration_ms = (
            round(
                serialized_window[-1]["offset_ms"]
                - serialized_window[0]["offset_ms"],
                3,
            )
            if serialized_window
            else 0.0
        )
        metadata["reproducibility"] = {
            "replay_supported": True,
            "image_size": {
                "width": int(raw_frame.shape[1]),
                "height": int(raw_frame.shape[0]),
            },
            "decision_window": {
                "sample_count": len(serialized_window),
                "duration_ms": duration_ms,
                "samples": serialized_window,
                "average_landmarks": average_serialized_landmarks(
                    serialized_window
                ),
            },
            "calibration_profile": _json_safe(calibration_profile),
            "runtime_provenance": _json_safe(runtime_provenance),
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
