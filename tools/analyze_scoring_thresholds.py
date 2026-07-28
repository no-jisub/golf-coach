import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.scoring_analysis import (  # noqa: E402
    build_scoring_analysis,
    render_scoring_analysis_markdown,
)


DEFAULT_REGRESSION = (
    PROJECT_ROOT
    / "analysis_sessions"
    / "runtime_regression"
    / "runtime_regression.json"
)
DEFAULT_GROUND_TRUTH = (
    PROJECT_ROOT
    / "reference_data"
    / "webcam_dataset"
    / "exports"
    / "coach_ground_truth.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis_sessions" / "scoring_analysis"


def parse_args():
    parser = argparse.ArgumentParser(
        description="단계·관절 병목과 코치 정답 기준 임계값별 오탐·미탐을 분석합니다."
    )
    parser.add_argument("--regression", type=Path, default=DEFAULT_REGRESSION)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    args = parse_args()
    ground_truth = load_json(args.ground_truth) if args.ground_truth.exists() else None
    report = build_scoring_analysis(load_json(args.regression), ground_truth)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "scoring_analysis.json"
    markdown_path = args.output_dir / "scoring_analysis.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_scoring_analysis_markdown(report),
        encoding="utf-8",
    )
    print(f"JSON: {json_path}")
    print(f"보고서: {markdown_path}")
    best = (report.get("threshold_analysis") or {}).get("best_f1_threshold")
    if best:
        print(
            f"최고 F1 임계값={best['threshold']} "
            f"precision={best['precision']} recall={best['recall']} f1={best['f1']}"
        )
    else:
        print("코치 good/bad 정답이 없어 임계값 성능은 계산하지 않았습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
