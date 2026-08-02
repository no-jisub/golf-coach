import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.regression_comparison import build_stage_comparison_sheet  # noqa: E402
from utils.runtime_regression import (  # noqa: E402
    build_reviewed_runtime_regression,
    build_runtime_regression,
    render_runtime_regression_markdown,
)
from utils.scoring_analysis import (  # noqa: E402
    build_scoring_analysis,
    render_scoring_analysis_markdown,
)
from utils.stage_candidate_extractor import extract_stage_candidates  # noqa: E402
from utils.swing_stage_accuracy import (  # noqa: E402
    DEFAULT_TOLERANCE_MS,
    build_stage_accuracy_report,
)
from utils.swing_stage_audit import (  # noqa: E402
    audit_manifest_videos,
    load_ground_truth_manifest,
)
from utils.swing_stage_contact_sheet import generate_audit_contact_sheets  # noqa: E402
from utils.swing_video import write_json  # noqa: E402


DEFAULT_MANIFEST = PROJECT_ROOT / "reference_data" / "swing_stage_ground_truth.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "analysis_sessions" / "reviewed_pipeline"
DEFAULT_WEBCAM_GROUND_TRUTH = (
    PROJECT_ROOT
    / "reference_data"
    / "webcam_dataset"
    / "exports"
    / "coach_ground_truth.json"
)
MODEL_PATH = PROJECT_ROOT / "pose_landmarker_full.task"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "검수 매니페스트가 갱신될 때 단계 감사, 정확도, 런타임 회귀, "
            "병목 보고서와 자동/정답 비교 이미지를 한 번에 재생성합니다."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--webcam-ground-truth", type=Path, default=DEFAULT_WEBCAM_GROUND_TRUTH)
    parser.add_argument("--video", action="append", dest="video_ids")
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument("--tolerance-ms", type=float, default=DEFAULT_TOLERANCE_MS)
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--no-candidates", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def manifest_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    args = parse_args()
    output_root = args.output_root.resolve()
    audit_root = output_root / "stage_audit"
    regression_root = output_root / "runtime_regression"
    scoring_root = output_root / "scoring_analysis"
    reviewed_root = output_root / "reviewed_only"
    reviewed_regression_root = reviewed_root / "runtime_regression"
    reviewed_scoring_root = reviewed_root / "scoring_analysis"
    candidate_root = output_root / "stage_candidates"
    comparison_root = output_root / "comparisons"

    manifest = load_ground_truth_manifest(
        args.manifest,
        project_root=PROJECT_ROOT,
        require_files=False,
    )
    selected_ids = args.video_ids or list(manifest["videos"])
    audit = audit_manifest_videos(
        manifest,
        project_root=PROJECT_ROOT,
        output_root=audit_root,
        model_path=MODEL_PATH,
        video_ids=selected_ids,
        sample_step=args.sample_step,
        reuse_landmarks=not args.no_reuse,
        progress=args.progress,
    )
    generate_audit_contact_sheets(
        audit,
        manifest,
        project_root=PROJECT_ROOT,
        output_root=audit_root,
    )
    accuracy = build_stage_accuracy_report(
        audit,
        manifest,
        tolerance_ms=args.tolerance_ms,
    )
    write_json(audit_root / "stage_detection_audit.json", audit)
    write_json(audit_root / "stage_accuracy_report.json", accuracy)

    regression = build_runtime_regression(
        manifest,
        audit_root,
        video_ids=selected_ids,
    )
    write_json(regression_root / "runtime_regression.json", regression)
    write_text(
        regression_root / "runtime_regression.md",
        render_runtime_regression_markdown(regression),
    )
    reviewed_regression = build_reviewed_runtime_regression(
        manifest,
        audit_root,
        video_ids=selected_ids,
    )
    write_json(
        reviewed_regression_root / "runtime_regression.json",
        reviewed_regression,
    )
    write_text(
        reviewed_regression_root / "runtime_regression.md",
        render_runtime_regression_markdown(reviewed_regression),
    )

    candidate_results = {}
    comparison_results = {}
    artifact_failures = []
    for video_id in selected_ids:
        video = manifest["videos"][video_id]
        if video.get("review_status") == "excluded":
            continue
        source_path = (PROJECT_ROOT / video["source"]).resolve()
        events_path = audit_root / video_id / "stage_events.json"
        if not source_path.exists() or not events_path.exists():
            artifact_failures.append(
                {
                    "video_id": video_id,
                    "type": "missing_source_or_events",
                    "message": f"{source_path} / {events_path}",
                }
            )
            continue
        automatic_events = load_json(events_path)
        if not args.no_candidates:
            try:
                result = extract_stage_candidates(
                    source_path,
                    automatic_events,
                    candidate_root / video_id,
                )
                candidate_results[video_id] = {
                    "status": "ok",
                    "contact_sheet": str(
                        candidate_root / video_id / result["contact_sheet"]
                    ),
                }
            except Exception as error:
                artifact_failures.append(
                    {
                        "video_id": video_id,
                        "type": "candidate_generation",
                        "message": str(error),
                    }
                )
        if video.get("review_status") == "reviewed":
            try:
                comparison_results[video_id] = build_stage_comparison_sheet(
                    source_path,
                    video["events"],
                    automatic_events,
                    comparison_root / f"{video_id}_automatic_vs_reviewed.jpg",
                )
            except Exception as error:
                artifact_failures.append(
                    {
                        "video_id": video_id,
                        "type": "comparison_generation",
                        "message": str(error),
                    }
                )

    webcam_ground_truth = (
        load_json(args.webcam_ground_truth)
        if args.webcam_ground_truth.exists()
        else None
    )
    scoring = build_scoring_analysis(regression, webcam_ground_truth)
    write_json(scoring_root / "scoring_analysis.json", scoring)
    write_text(
        scoring_root / "scoring_analysis.md",
        render_scoring_analysis_markdown(scoring),
    )
    reviewed_scoring = build_scoring_analysis(
        reviewed_regression,
        webcam_ground_truth,
    )
    write_json(
        reviewed_scoring_root / "scoring_analysis.json",
        reviewed_scoring,
    )
    write_text(
        reviewed_scoring_root / "scoring_analysis.md",
        render_scoring_analysis_markdown(reviewed_scoring),
    )

    state = {
        "schema": "golf-coach-reviewed-regression-pipeline-v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha256(args.manifest),
        "selected_video_ids": selected_ids,
        "reviewed_video_ids": [
            video_id
            for video_id in selected_ids
            if manifest["videos"][video_id].get("review_status") == "reviewed"
        ],
        "summary": {
            "audit": audit["summary"],
            "accuracy": accuracy["summary"],
            "regression": regression["summary"],
            "reviewed_only_regression": reviewed_regression["summary"],
            "reviewed_only_dataset_quality": reviewed_regression[
                "dataset_quality"
            ],
            "candidate_count": len(candidate_results),
            "comparison_count": len(comparison_results),
            "artifact_failure_count": len(artifact_failures),
        },
        "artifacts": {
            "audit": str(audit_root / "stage_detection_audit.json"),
            "accuracy": str(audit_root / "stage_accuracy_report.json"),
            "runtime_regression": str(
                regression_root / "runtime_regression.json"
            ),
            "scoring_analysis": str(scoring_root / "scoring_analysis.json"),
            "reviewed_only_runtime_regression": str(
                reviewed_regression_root / "runtime_regression.json"
            ),
            "reviewed_only_scoring_analysis": str(
                reviewed_scoring_root / "scoring_analysis.json"
            ),
            "candidates": candidate_results,
            "comparisons": comparison_results,
        },
        "artifact_failures": artifact_failures,
    }
    write_json(output_root / "pipeline_state.json", state)
    print(
        f"감사={audit['summary']['processed_count']} "
        f"검수완료={accuracy['summary']['reviewed_count']} "
        f"회귀={regression['summary']['evaluated_video_count']} "
        f"검수전용회귀={reviewed_regression['summary']['evaluated_video_count']} "
        f"후보={len(candidate_results)} 비교={len(comparison_results)} "
        f"아티팩트실패={len(artifact_failures)}"
    )
    print(f"파이프라인 상태: {output_root / 'pipeline_state.json'}")
    return 1 if audit["summary"]["failed_count"] or artifact_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
