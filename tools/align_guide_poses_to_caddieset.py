import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.guide_alignment import STAGE_KEYS, align_guide_poses_to_caddieset
from utils.guide_skeleton import REFERENCE_GUIDE_POSES, SWING_HAND


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "reference_data"
    / "guide_poses"
    / "caddieset_aligned_guide_poses.json"
)
SCHEMA = "golf-coach-caddieset-aligned-guide-v1"


def serialize_pose(pose):
    return {
        str(index): [round(point[0], 6), round(point[1], 6)]
        for index, point in pose.items()
    }


def build_aligned_guide(guide_poses=REFERENCE_GUIDE_POSES, swing_hand=SWING_HAND):
    direction_multiplier = -1.0 if swing_hand == "right" else 1.0
    aligned, report = align_guide_poses_to_caddieset(
        guide_poses,
        view="FACEON",
        direction_multiplier=direction_multiplier,
    )
    return {
        "schema": SCHEMA,
        "profile_id": "faceon_all",
        "swing_hand": swing_hand,
        "coordinate_space": "runtime_guide_after_swing_hand",
        "max_joint_displacement_limit": 0.12,
        "stage_order": list(STAGE_KEYS),
        "stages": {
            stage_key: serialize_pose(aligned[stage_key]) for stage_key in STAGE_KEYS
        },
        "alignment": report,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="현재 예시 스켈레톤을 CaddieSet 참조 범위에 최소 변형으로 정렬합니다."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output.resolve()
    output = build_aligned_guide()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    before = output["alignment"]["before"]
    after = output["alignment"]["after"]
    print(
        f"정렬 전: pass={before['pass_count']} warning={before['warning_count']} "
        f"outside_observed={before['outside_observed_count']}"
    )
    print(
        f"정렬 후: pass={after['pass_count']} warning={after['warning_count']} "
        f"outside_observed={after['outside_observed_count']}"
    )
    for stage_key in STAGE_KEYS:
        stage = output["alignment"]["stages"][stage_key]
        print(
            f"[{stage_key}] warning {stage['before_objective'][0]} -> "
            f"{stage['after_objective'][0]}, max_move={stage['max_joint_displacement']:.4f}"
        )
    print(f"저장: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
