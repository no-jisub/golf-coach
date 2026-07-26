import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.app_config import MVP_CLUB_TYPE, MVP_VIEW
from utils.guide_alignment import STAGE_KEYS, calculate_guide_stage_metrics
from utils.guide_skeleton import GUIDE_POSES, SWING_HAND


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "reference_data"
    / "guide_poses"
    / "caddieset_guide_metrics.json"
)
SCHEMA = "golf-coach-guide-caddieset-metrics-v1"


def build_metric_snapshot(guide_poses=GUIDE_POSES, swing_hand=SWING_HAND):
    direction_multiplier = -1.0 if swing_hand == "right" else 1.0
    return {
        "schema": SCHEMA,
        "swing_hand": swing_hand,
        "coordinate_space": "runtime_guide_after_swing_hand",
        "direction_multiplier": direction_multiplier,
        "profile_id": f"{MVP_VIEW.lower()}_{MVP_CLUB_TYPE.lower()}",
        "stage_order": list(STAGE_KEYS),
        "stages": calculate_guide_stage_metrics(
            guide_poses,
            direction_multiplier=direction_multiplier,
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="현재 런타임 가이드 스켈레톤의 CaddieSet 대응 지표를 계산합니다."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output.resolve()
    snapshot = build_metric_snapshot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for stage_key in STAGE_KEYS:
        measured_count = sum(
            value is not None for value in snapshot["stages"][stage_key].values()
        )
        print(f"[{stage_key}] 계산 지표 {measured_count}개")
    print(f"저장: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
