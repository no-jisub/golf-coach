import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.regression_baseline import (  # noqa: E402
    build_synthetic_runtime_snapshot,
    compare_runtime_snapshots,
    load_snapshot,
    render_comparison_markdown,
    write_snapshot,
)


DEFAULT_BASELINE = PROJECT_ROOT / "tests" / "fixtures" / "runtime_scoring_baseline.json"
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "analysis_sessions"
    / "regression_baseline"
    / "runtime_scoring_comparison.md"
)


def main():
    parser = argparse.ArgumentParser(
        description="합성 8단계 판정 결과를 비식별 회귀 기준선과 비교합니다."
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--score-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--update",
        action="store_true",
        help="현재 합성 결과로 기준선을 명시적으로 갱신합니다.",
    )
    args = parser.parse_args()
    if args.score_tolerance < 0:
        parser.error("--score-tolerance은 0 이상이어야 합니다.")

    current = build_synthetic_runtime_snapshot()
    if args.update:
        destination = write_snapshot(args.baseline, current)
        print(f"회귀 기준선 갱신: {destination}")
        return 0

    baseline = load_snapshot(args.baseline)
    comparison = compare_runtime_snapshots(
        baseline,
        current,
        score_tolerance=args.score_tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_comparison_markdown(comparison), encoding="utf-8")

    if comparison["passed"]:
        print("회귀 기준선 검사 통과: 합성 8단계 판정이 기준선과 일치합니다.")
        return 0
    print(f"회귀 기준선 검사 실패: {comparison['change_count']}개 항목 변경")
    for item in comparison["changes"]:
        print(
            f"- {item.get('stage') or 'global'}:{item['field']} "
            f"{item.get('baseline')} -> {item.get('current')}"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
