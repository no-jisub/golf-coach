"""저장된 v3 웹캠 판정 입력을 현재 코드로 다시 평가합니다."""

import json
from pathlib import Path

from utils.diagnostic_capture import REPRODUCIBLE_CAPTURE_SCHEMA
from utils.golf_rules import analyze_stage_pose
from utils.runtime_regression import deserialize_landmarks


REPLAY_SCHEMA = "golf-coach-runtime-sample-replay-v1"


def load_sample(sample_or_path):
    if isinstance(sample_or_path, (str, Path)):
        return json.loads(Path(sample_or_path).read_text(encoding="utf-8"))
    return dict(sample_or_path)


def restore_calibration_profile(payload):
    profile = dict(payload or {})
    if "shoulder_mid" in profile:
        profile["shoulder_mid"] = tuple(profile["shoulder_mid"])
    address_points = profile.get("caddieset_address_points")
    if isinstance(address_points, dict):
        profile["caddieset_address_points"] = {
            int(index): tuple(point)
            for index, point in address_points.items()
        }
    return profile


def _score(feedback, key):
    value = ((feedback or {}).get("metrics") or {}).get(key)
    return None if value is None else float(value)


def compare_feedback(stored, replayed):
    score_keys = ("final_score", "guide_score", "caddieset_score")
    score_deltas = {}
    for key in score_keys:
        old = _score(stored, key)
        new = _score(replayed, key)
        score_deltas[key] = (
            round(new - old, 4)
            if old is not None and new is not None
            else None
        )
    stored_passed = None if stored is None else bool(stored.get("passed"))
    replayed_passed = bool(replayed.get("passed"))
    changed = stored_passed is not None and stored_passed != replayed_passed
    return {
        "passed_changed": changed,
        "stored_passed": stored_passed,
        "replayed_passed": replayed_passed,
        "status_changed": (
            stored is not None
            and stored.get("status") != replayed.get("status")
        ),
        "score_deltas": score_deltas,
        "has_score_drift": any(
            delta is not None and abs(delta) > 0.01
            for delta in score_deltas.values()
        ),
    }


def compare_provenance(stored, current):
    if current is None:
        return None
    stored = stored or {}
    fields = {
        "git_commit": (
            stored.get("git_commit"),
            current.get("git_commit"),
        ),
        "model_sha256": (
            stored.get("model", {}).get("sha256"),
            current.get("model", {}).get("sha256"),
        ),
    }
    for name in ("caddieset_profile", "aligned_guide", "generated_guide"):
        fields[f"{name}_sha256"] = (
            stored.get("reference_data", {}).get(name, {}).get("sha256"),
            current.get("reference_data", {}).get(name, {}).get("sha256"),
        )
    return {
        key: {
            "stored": old,
            "current": new,
            "changed": old != new,
        }
        for key, (old, new) in fields.items()
    }


def replay_runtime_sample(
    sample_or_path,
    *,
    analyzer=analyze_stage_pose,
    current_provenance=None,
):
    sample = load_sample(sample_or_path)
    if sample.get("schema") != REPRODUCIBLE_CAPTURE_SCHEMA:
        raise ValueError(
            "replay는 golf-coach-runtime-sample-v3 샘플만 지원합니다."
        )
    reproducibility = sample.get("reproducibility") or {}
    if not reproducibility.get("replay_supported"):
        raise ValueError("샘플에 replay 가능한 판정 입력이 없습니다.")

    timed_samples = reproducibility.get("decision_window", {}).get("samples", [])
    landmark_samples = [
        deserialize_landmarks({"landmarks": record.get("landmarks", [])})
        for record in timed_samples
    ]
    if not landmark_samples or any(not landmarks for landmarks in landmark_samples):
        raise ValueError("replay에 필요한 관절 프레임이 없습니다.")
    image_size = reproducibility.get("image_size") or {}
    width = int(image_size.get("width", 0))
    height = int(image_size.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("replay에 필요한 이미지 크기가 없습니다.")

    replayed = analyzer(
        sample["stage_key"],
        landmark_samples,
        restore_calibration_profile(
            reproducibility.get("calibration_profile")
        ),
        width,
        height,
    )
    differences = compare_feedback(sample.get("feedback"), replayed)
    return {
        "schema": REPLAY_SCHEMA,
        "sample_id": sample.get("sample_id"),
        "stage_key": sample.get("stage_key"),
        "sample_count": len(landmark_samples),
        "stored_feedback": sample.get("feedback"),
        "replayed_feedback": replayed,
        "differences": differences,
        "provenance_comparison": compare_provenance(
            reproducibility.get("runtime_provenance"),
            current_provenance,
        ),
    }
