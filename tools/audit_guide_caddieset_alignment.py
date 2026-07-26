import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.app_config import MVP_CLUB_TYPE, MVP_VIEW
from utils.guide_alignment import (
    STAGE_KEYS,
    audit_guide_stage_metrics,
    calculate_guide_stage_metrics,
)
from utils.guide_skeleton import GUIDE_POSES, SWING_HAND


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "reference_data"
    / "guide_poses"
    / "caddieset_alignment_report.json"
)
SCHEMA = "golf-coach-guide-caddieset-alignment-v1"


def build_alignment_report(guide_poses=GUIDE_POSES, swing_hand=SWING_HAND):
    direction_multiplier = -1.0 if swing_hand == "right" else 1.0
    metrics = calculate_guide_stage_metrics(
        guide_poses,
        direction_multiplier=direction_multiplier,
    )
    audit = audit_guide_stage_metrics(
        metrics,
        view=MVP_VIEW,
        club_type=MVP_CLUB_TYPE,
    )
    return {
        "schema": SCHEMA,
        "swing_hand": swing_hand,
        "coordinate_space": "runtime_guide_after_swing_hand",
        "profile_id": f"{MVP_VIEW.lower()}_{MVP_CLUB_TYPE.lower()}",
        "summary": audit["summary"],
        "stages": {
            stage_key: {
                "overall_status": audit["stages"][stage_key]["overall_status"],
                "summary": audit["stages"][stage_key]["summary"],
                "items": audit["stages"][stage_key]["comparisons"],
            }
            for stage_key in STAGE_KEYS
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="현재 예시 스켈레톤과 CaddieSet 단계별 범위의 정렬 상태를 검사합니다."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output.resolve()
    report = build_alignment_report()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for stage_key in STAGE_KEYS:
        stage = report["stages"][stage_key]
        summary = stage["summary"]
        print(
            f"[{stage_key}] {stage['overall_status']} "
            f"pass={summary['pass_count']} warning={summary['warning_count']} "
            f"unavailable={summary['unavailable_count']}"
        )
    summary = report["summary"]
    print(
        f"전체: pass={summary['pass_count']} warning={summary['warning_count']} "
        f"outside_observed={summary['outside_observed_count']}"
    )
    print(f"저장: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
