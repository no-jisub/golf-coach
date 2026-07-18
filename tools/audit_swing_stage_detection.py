import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.swing_stage_audit import audit_manifest_videos, load_ground_truth_manifest
from utils.swing_stage_accuracy import (
    DEFAULT_TOLERANCE_MS,
    build_stage_accuracy_report,
)
from utils.swing_stage_contact_sheet import generate_audit_contact_sheets
from utils.swing_video import write_json


DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "reference_data" / "swing_stage_ground_truth.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis_sessions" / "stage_audit"
MODEL_PATH = PROJECT_ROOT / "pose_landmarker_full.task"


def parse_args():
    parser = argparse.ArgumentParser(
        description="정답 매니페스트의 정면 스윙 영상을 일괄 분석해 자동 8단계 결과를 저장합니다."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--video",
        action="append",
        dest="video_ids",
        help="분석할 영상 ID. 여러 번 지정할 수 있으며 생략하면 전체를 분석합니다.",
    )
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=DEFAULT_TOLERANCE_MS,
        help="정답 프레임 대비 자동 감지 성공으로 인정할 최대 시간 오차(ms).",
    )
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="기존 frame_landmarks.json 캐시를 사용하지 않고 다시 추출합니다.",
    )
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    manifest = load_ground_truth_manifest(
        manifest_path,
        project_root=PROJECT_ROOT,
        require_files=False,
    )
    audit = audit_manifest_videos(
        manifest,
        project_root=PROJECT_ROOT,
        output_root=output_dir,
        model_path=MODEL_PATH,
        video_ids=args.video_ids,
        sample_step=args.sample_step,
        reuse_landmarks=not args.no_reuse,
        progress=args.progress,
    )
    generate_audit_contact_sheets(
        audit,
        manifest,
        project_root=PROJECT_ROOT,
        output_root=output_dir,
    )
    accuracy = build_stage_accuracy_report(
        audit,
        manifest,
        tolerance_ms=args.tolerance_ms,
    )
    output_path = write_json(output_dir / "stage_detection_audit.json", audit)
    accuracy_path = write_json(output_dir / "stage_accuracy_report.json", accuracy)
    summary = audit["summary"]
    print(
        f"완료={summary['processed_count']} 실패={summary['failed_count']} "
        f"제외={summary['excluded_count']} 캐시생성={summary['created_cache_count']} "
        f"캐시재사용={summary['reused_cache_count']} "
        f"검수시트={summary['contact_sheet_count']}"
    )
    print(f"감사 결과: {output_path}")
    accuracy_summary = accuracy["summary"]
    rate = accuracy_summary["within_tolerance_rate"]
    rate_text = "검수 대기" if rate is None else f"{rate * 100:.1f}%"
    print(
        f"정답 검수완료={accuracy_summary['reviewed_count']} "
        f"검수대기={accuracy_summary['pending_review_count']} "
        f"평가단계={accuracy_summary['evaluated_count']} "
        f"허용오차내={rate_text}"
    )
    print(f"정확도 보고서: {accuracy_path}")
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
