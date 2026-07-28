"""웹캠 샘플 검수 큐, 예비 라벨, 우선순위와 정답 내보내기를 관리합니다."""

import json
from datetime import datetime
from pathlib import Path


REVIEW_SCHEMA = "golf-coach-webcam-review-v1"
GROUND_TRUTH_SCHEMA = "golf-coach-webcam-ground-truth-v1"
COACH_LABELS = {"good", "bad", "uncertain", "pending"}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def discover_runtime_samples(sample_roots):
    """v1/v2 sample.json을 찾아 sample_id 기준으로 반환합니다."""
    discovered = {}
    for root in sample_roots:
        root = Path(root)
        if not root.exists():
            continue
        for metadata_path in sorted(root.rglob("sample.json")):
            sample = load_json(metadata_path)
            sample_id = sample.get("sample_id")
            if not sample_id:
                continue
            if sample_id in discovered:
                raise ValueError(f"중복 sample_id입니다: {sample_id}")
            discovered[sample_id] = {
                "metadata_path": str(metadata_path.resolve()),
                "sample": sample,
            }
    return discovered


def preliminary_label(sample):
    """수집자 기대 라벨을 우선하고, 없으면 런타임 판정을 낮은 신뢰도로 제안합니다."""
    expected = sample.get("expected_label", "pending")
    actual = sample.get("actual_passed")
    if expected == "expected_pass":
        return {"label": "good", "confidence": 0.9, "source": "collector"}
    if expected == "expected_fail":
        return {"label": "bad", "confidence": 0.9, "source": "collector"}
    if actual is True:
        return {"label": "good", "confidence": 0.55, "source": "runtime_model"}
    if actual is False:
        return {"label": "bad", "confidence": 0.55, "source": "runtime_model"}
    return {"label": "uncertain", "confidence": 0.0, "source": "unavailable"}


def review_priority(sample):
    """오판 가능성과 임계값 근접도를 이용해 먼저 볼 샘플을 정렬합니다."""
    score = 0
    reasons = []
    discrepancy = sample.get("discrepancy")
    if discrepancy in {"false_accept", "false_reject"}:
        score += 100
        reasons.append(discrepancy)
    if sample.get("expected_label", "pending") == "pending":
        score += 35
        reasons.append("collector_label_missing")

    feedback = sample.get("feedback") or {}
    status = feedback.get("status")
    if status == "unavailable" or sample.get("actual_passed") is None:
        score += 30
        reasons.append("model_unavailable")

    final_score = (feedback.get("metrics") or {}).get("final_score")
    if final_score is not None:
        margin = abs(float(final_score) - 70.0)
        if margin <= 5:
            score += 30
            reasons.append("near_threshold_5")
        elif margin <= 10:
            score += 15
            reasons.append("near_threshold_10")

    preliminary = preliminary_label(sample)
    if preliminary["confidence"] < 0.6:
        score += 10
        reasons.append("low_preliminary_confidence")
    return {"score": score, "reasons": reasons}


def _artifact_paths(metadata_path, sample):
    metadata_path = Path(metadata_path)
    artifacts = sample.get("artifacts", {})
    return {
        key: str((metadata_path.parent / value).resolve())
        for key, value in artifacts.items()
        if value
    }


def build_review_manifest(sample_roots, existing_manifest=None):
    existing_manifest = existing_manifest or {}
    existing_samples = existing_manifest.get("samples", {})
    discovered = discover_runtime_samples(sample_roots)
    samples = {}
    for sample_id, record in discovered.items():
        sample = record["sample"]
        existing = existing_samples.get(sample_id, {})
        samples[sample_id] = {
            "sample_id": sample_id,
            "stage_key": sample.get("stage_key"),
            "metadata_path": record["metadata_path"],
            "artifacts": _artifact_paths(record["metadata_path"], sample),
            "participant_id": (
                sample.get("collection", {}).get("participant_id", "legacy_anonymous")
            ),
            "session_id": (
                sample.get("collection", {}).get("session_id", "legacy")
            ),
            "preliminary": preliminary_label(sample),
            "priority": review_priority(sample),
            "coach_review": existing.get(
                "coach_review",
                {
                    "label": "pending",
                    "reviewed_by": None,
                    "reviewed_at": None,
                    "note": "",
                },
            ),
        }

    ordered = sorted(
        samples,
        key=lambda sample_id: (
            -samples[sample_id]["priority"]["score"],
            samples[sample_id]["stage_key"] or "",
            sample_id,
        ),
    )
    summary = {
        "sample_count": len(samples),
        "reviewed_count": sum(
            samples[sample_id]["coach_review"]["label"] != "pending"
            for sample_id in samples
        ),
        "pending_count": sum(
            samples[sample_id]["coach_review"]["label"] == "pending"
            for sample_id in samples
        ),
        "uncertain_count": sum(
            samples[sample_id]["coach_review"]["label"] == "uncertain"
            for sample_id in samples
        ),
    }
    return {
        "schema": REVIEW_SCHEMA,
        "updated_at": datetime.now().astimezone().isoformat(),
        "sample_roots": [str(Path(root).resolve()) for root in sample_roots],
        "summary": summary,
        "review_order": ordered,
        "samples": samples,
    }


def apply_coach_review(
    manifest,
    sample_id,
    label,
    *,
    reviewed_by,
    note="",
    reviewed_at=None,
):
    if manifest.get("schema") != REVIEW_SCHEMA:
        raise ValueError("지원하지 않는 웹캠 검수 매니페스트입니다.")
    if sample_id not in manifest.get("samples", {}):
        raise KeyError(f"검수 큐에 없는 sample_id입니다: {sample_id}")
    if label not in COACH_LABELS:
        raise ValueError(f"지원하지 않는 코치 라벨입니다: {label}")
    if label != "pending" and not str(reviewed_by or "").strip():
        raise ValueError("확정 검수에는 reviewed_by가 필요합니다.")

    manifest["samples"][sample_id]["coach_review"] = {
        "label": label,
        "reviewed_by": str(reviewed_by).strip() if reviewed_by else None,
        "reviewed_at": (
            (reviewed_at or datetime.now().astimezone()).isoformat()
            if label != "pending"
            else None
        ),
        "note": str(note or "").strip(),
    }
    manifest["updated_at"] = datetime.now().astimezone().isoformat()
    manifest["summary"]["reviewed_count"] = sum(
        sample["coach_review"]["label"] != "pending"
        for sample in manifest["samples"].values()
    )
    manifest["summary"]["pending_count"] = sum(
        sample["coach_review"]["label"] == "pending"
        for sample in manifest["samples"].values()
    )
    manifest["summary"]["uncertain_count"] = sum(
        sample["coach_review"]["label"] == "uncertain"
        for sample in manifest["samples"].values()
    )
    return manifest["samples"][sample_id]


def export_coach_ground_truth(review_manifest, *, include_uncertain=False):
    """코치 검수 완료 샘플을 비식별 분석용 정답 레코드로 변환합니다."""
    if review_manifest.get("schema") != REVIEW_SCHEMA:
        raise ValueError("지원하지 않는 웹캠 검수 매니페스트입니다.")

    allowed = {"good", "bad"}
    if include_uncertain:
        allowed.add("uncertain")
    records = []
    for sample_id in review_manifest.get("review_order", []):
        review_item = review_manifest["samples"][sample_id]
        coach_review = review_item["coach_review"]
        if coach_review["label"] not in allowed:
            continue
        sample = load_json(review_item["metadata_path"])
        feedback = sample.get("feedback") or {}
        metrics = feedback.get("metrics") or {}
        collection = sample.get("collection") or {}
        records.append(
            {
                "sample_id": sample_id,
                "stage_key": sample.get("stage_key"),
                "label": coach_review["label"],
                "review": coach_review,
                "participant_id": collection.get(
                    "participant_id", "legacy_anonymous"
                ),
                "session_id": collection.get("session_id", "legacy"),
                "body_profile": collection.get("body_profile", {}),
                "capture_conditions": collection.get("capture_conditions", {}),
                "model": {
                    "passed": sample.get("actual_passed"),
                    "status": feedback.get("status"),
                    "final_score": metrics.get("final_score"),
                    "guide_score": metrics.get("guide_score"),
                    "caddieset_score": metrics.get("caddieset_score"),
                    "messages": feedback.get("messages", []),
                },
                "landmarks": sample.get("landmarks", []),
                "artifacts": review_item.get("artifacts", {}),
                "source_metadata": review_item["metadata_path"],
            }
        )

    return {
        "schema": GROUND_TRUTH_SCHEMA,
        "generated_at": datetime.now().astimezone().isoformat(),
        "privacy": {
            "contains_direct_identifiers": False,
            "participant_ids_are_pseudonymous": True,
            "local_review_required_before_distribution": True,
        },
        "summary": {
            "record_count": len(records),
            "good_count": sum(record["label"] == "good" for record in records),
            "bad_count": sum(record["label"] == "bad" for record in records),
            "uncertain_count": sum(
                record["label"] == "uncertain" for record in records
            ),
        },
        "records": records,
    }
