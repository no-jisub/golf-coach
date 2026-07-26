import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.runtime_regression import (  # noqa: E402
    build_runtime_regression,
    load_json,
    render_runtime_regression_markdown,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="기존 프로 영상 좌표 캐시를 현재 웹캠 판정 기준으로 회귀 분석합니다."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reference_data" / "swing_stage_ground_truth.json",
        help="8단계 검수 매니페스트 JSON 경로",
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=PROJECT_ROOT / "analysis_sessions" / "stage_audit",
        help="영상별 frame_landmarks.json과 stage_events.json이 있는 폴더",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis_sessions" / "runtime_regression",
        help="JSON과 Markdown 보고서를 저장할 폴더",
    )
    parser.add_argument(
        "--video",
        action="append",
        dest="video_ids",
        help="분석할 영상 ID. 여러 번 지정할 수 있습니다. 생략하면 전체를 분석합니다.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = load_json(args.manifest)
    report = build_runtime_regression(
        manifest,
        args.audit_root,
        video_ids=args.video_ids,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "runtime_regression.json"
    markdown_path = args.output_dir / "runtime_regression.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_runtime_regression_markdown(report),
        encoding="utf-8",
    )

    summary = report["summary"]
    print(
        "회귀 분석 완료: "
        f"{summary['evaluated_video_count']}/{summary['video_count']}개 영상 평가, "
        f"{summary['failed_video_count']}개 실패"
    )
    print(f"JSON: {json_path}")
    print(f"보고서: {markdown_path}")
    for stage_key, stage in summary["stages"].items():
        score = stage["mean_final_score"]
        pass_rate = stage["pass_rate"]
        score_text = "-" if score is None else f"{score:.1f}"
        pass_text = "-" if pass_rate is None else f"{pass_rate * 100:.1f}%"
        print(
            f"- {stage_key:14s} 평균 {score_text:>5s}, "
            f"통과 {pass_text:>6s}, {stage['diagnosis']}"
        )

    return 1 if summary["failed_video_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
