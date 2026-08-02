import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.diagnostic_capture import REPRODUCIBLE_CAPTURE_SCHEMA  # noqa: E402
from utils.runtime_provenance import build_runtime_provenance  # noqa: E402
from utils.runtime_sample_replay import replay_runtime_sample  # noqa: E402


DEFAULT_ROOTS = [
    PROJECT_ROOT / "reference_data" / "runtime_samples",
    PROJECT_ROOT / "reference_data" / "webcam_dataset" / "captures",
]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "analysis_sessions"
    / "runtime_sample_replay"
    / "replay_report.json"
)
MODEL_PATH = PROJECT_ROOT / "pose_landmarker_full.task"


def parse_args():
    parser = argparse.ArgumentParser(
        description="저장된 v3 웹캠 판정 입력을 현재 코드로 다시 평가합니다."
    )
    parser.add_argument("--sample-root", action="append", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    args = parse_args()
    roots = args.sample_root or DEFAULT_ROOTS
    paths = sorted(
        path
        for root in roots
        if root.exists()
        for path in root.rglob("sample.json")
    )
    records = []
    for path in paths:
        sample = load_json(path)
        if sample.get("schema") != REPRODUCIBLE_CAPTURE_SCHEMA:
            records.append(
                {
                    "path": str(path),
                    "status": "skipped",
                    "reason": f"unsupported_schema:{sample.get('schema')}",
                }
            )
            continue
        stored_provenance = sample["reproducibility"]["runtime_provenance"]
        current_provenance = build_runtime_provenance(
            model_path=MODEL_PATH,
            runtime_settings=stored_provenance.get("runtime_settings", {}),
        )
        try:
            result = replay_runtime_sample(
                sample,
                current_provenance=current_provenance,
            )
            records.append({"path": str(path), "status": "ok", **result})
        except Exception as error:
            records.append(
                {
                    "path": str(path),
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    report = {
        "schema": "golf-coach-runtime-sample-replay-report-v1",
        "summary": {
            "sample_count": len(records),
            "replayed_count": sum(record["status"] == "ok" for record in records),
            "skipped_count": sum(record["status"] == "skipped" for record in records),
            "failed_count": sum(record["status"] == "failed" for record in records),
            "decision_changed_count": sum(
                record.get("differences", {}).get("passed_changed", False)
                for record in records
            ),
            "score_drift_count": sum(
                record.get("differences", {}).get("has_score_drift", False)
                for record in records
            ),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = report["summary"]
    print(
        f"replay={summary['replayed_count']} skip={summary['skipped_count']} "
        f"fail={summary['failed_count']} decision_change={summary['decision_changed_count']} "
        f"score_drift={summary['score_drift_count']}"
    )
    print(f"보고서: {args.output}")
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
