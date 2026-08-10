"""개인정보 없는 합성 입력으로 런타임 판정 회귀를 감시합니다."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from utils.golf_rules import STAGE_CONFIGS, analyze_stage_pose
from utils.guide_skeleton import GUIDE_POSES, create_calibration_profile


BASELINE_SCHEMA = "golf-coach-runtime-scoring-baseline-v1"
SCORE_FIELDS = ("final_score", "guide_score", "caddieset_score")
EXACT_FIELDS = ("passed", "status", "profile_id", "measured_count")


def _landmarks_from_points(points):
    landmarks = [
        SimpleNamespace(x=0.0, y=0.0, visibility=0.0) for _ in range(33)
    ]
    for index, point in points.items():
        landmarks[index] = SimpleNamespace(
            x=float(point[0]),
            y=float(point[1]),
            visibility=1.0,
        )
    return landmarks


def build_synthetic_runtime_snapshot():
    """현재 런타임 가이드 자체를 입력해 판정 파이프라인의 결정적 스냅샷을 만듭니다."""
    address_landmarks = _landmarks_from_points(GUIDE_POSES["address"])
    calibration = create_calibration_profile([address_landmarks], 1000, 1000)
    calibration["caddieset_address_points"] = GUIDE_POSES["address"]

    stages = {}
    for stage in STAGE_CONFIGS:
        stage_key = stage["key"]
        result = analyze_stage_pose(
            stage_key,
            [_landmarks_from_points(GUIDE_POSES[stage_key])],
            calibration,
            1000,
            1000,
        )
        metrics = result.get("metrics", {})
        stages[stage_key] = {
            "passed": bool(result.get("passed")),
            "status": result.get("status"),
            "final_score": int(metrics.get("final_score", 0)),
            "guide_score": int(metrics.get("guide_score", 0)),
            "caddieset_score": int(metrics.get("caddieset_score", 0)),
            "profile_id": metrics.get("profile_id"),
            "measured_count": int(metrics.get("measured_count", 0)),
        }

    return {
        "schema": BASELINE_SCHEMA,
        "kind": "synthetic_runtime_scoring",
        "scope": {
            "view": "FACEON",
            "club_type": "I7",
            "swing_hand": "right",
            "image_size": [1000, 1000],
        },
        "limitations": [
            "개인정보 없는 고정 가이드 좌표로 코드 회귀만 감시합니다.",
            "실제 사용자 정확도나 임계값 조정 근거로 사용할 수 없습니다.",
        ],
        "stage_order": [stage["key"] for stage in STAGE_CONFIGS],
        "stages": stages,
    }


def validate_baseline(snapshot):
    if snapshot.get("schema") != BASELINE_SCHEMA:
        raise ValueError(f"지원하지 않는 회귀 기준선 스키마입니다: {snapshot.get('schema')}")
    stage_order = snapshot.get("stage_order")
    expected = [stage["key"] for stage in STAGE_CONFIGS]
    if stage_order != expected:
        raise ValueError("회귀 기준선의 8단계 순서가 현재 런타임과 다릅니다.")
    if set(snapshot.get("stages", {})) != set(expected):
        raise ValueError("회귀 기준선에 정확히 8단계 결과가 필요합니다.")
    return snapshot


def compare_runtime_snapshots(baseline, current, *, score_tolerance=0):
    validate_baseline(baseline)
    validate_baseline(current)
    changes = []

    if baseline.get("scope") != current.get("scope"):
        changes.append(
            {
                "stage": None,
                "field": "scope",
                "baseline": baseline.get("scope"),
                "current": current.get("scope"),
            }
        )

    for stage_key in baseline["stage_order"]:
        before = baseline["stages"][stage_key]
        after = current["stages"][stage_key]
        for field in EXACT_FIELDS:
            if before.get(field) != after.get(field):
                changes.append(
                    {
                        "stage": stage_key,
                        "field": field,
                        "baseline": before.get(field),
                        "current": after.get(field),
                    }
                )
        for field in SCORE_FIELDS:
            before_value = float(before.get(field, 0))
            after_value = float(after.get(field, 0))
            if abs(after_value - before_value) > float(score_tolerance):
                changes.append(
                    {
                        "stage": stage_key,
                        "field": field,
                        "baseline": before.get(field),
                        "current": after.get(field),
                        "delta": after_value - before_value,
                    }
                )

    return {
        "schema": "golf-coach-runtime-scoring-comparison-v1",
        "passed": not changes,
        "score_tolerance": float(score_tolerance),
        "change_count": len(changes),
        "changes": changes,
    }


def render_comparison_markdown(comparison):
    lines = [
        "# Runtime Scoring Baseline Comparison",
        "",
        f"- 결과: {'pass' if comparison['passed'] else 'fail'}",
        f"- 허용 점수 변화: ±{comparison['score_tolerance']:g}",
        f"- 변경 항목: {comparison['change_count']}",
        "",
    ]
    if comparison["passed"]:
        lines.append("합성 8단계 판정이 저장된 기준선과 일치합니다.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| Stage | Field | Baseline | Current |",
            "|---|---|---|---|",
        ]
    )
    for item in comparison["changes"]:
        lines.append(
            f"| {item.get('stage') or '-'} | {item['field']} | "
            f"{item.get('baseline')} | {item.get('current')} |"
        )
    return "\n".join(lines) + "\n"


def load_snapshot(path):
    return validate_baseline(json.loads(Path(path).read_text(encoding="utf-8")))


def write_snapshot(path, snapshot):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(validate_baseline(deepcopy(snapshot)), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return destination
