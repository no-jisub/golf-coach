import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.sample_review import (  # noqa: E402
    export_coach_ground_truth,
    load_json,
    write_json,
)


DEFAULT_REVIEW_MANIFEST = (
    PROJECT_ROOT
    / "reference_data"
    / "webcam_dataset"
    / "reviews"
    / "coach_reviews.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reference_data"
    / "webcam_dataset"
    / "exports"
    / "coach_ground_truth.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="코치의 good/bad 검수 결과를 비식별 정답 데이터로 변환합니다."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-uncertain", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    report = export_coach_ground_truth(
        load_json(args.manifest),
        include_uncertain=args.include_uncertain,
    )
    write_json(args.output, report)
    summary = report["summary"]
    print(
        f"정답={summary['record_count']} good={summary['good_count']} "
        f"bad={summary['bad_count']} uncertain={summary['uncertain_count']}"
    )
    print(f"출력: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
