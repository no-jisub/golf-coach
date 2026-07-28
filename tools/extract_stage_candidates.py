import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stage_candidate_extractor import extract_stage_candidates  # noqa: E402


DEFAULT_MANIFEST = PROJECT_ROOT / "reference_data" / "swing_stage_ground_truth.json"
DEFAULT_AUDIT_ROOT = PROJECT_ROOT / "analysis_sessions" / "stage_audit"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "analysis_sessions" / "stage_candidates"


def parse_args():
    parser = argparse.ArgumentParser(
        description="자동 검출 단계 주변 프레임을 코치 검수 후보로 일괄 추출합니다."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--video", action="append", dest="video_ids")
    parser.add_argument("--radius-frames", type=int, default=8)
    parser.add_argument("--candidate-count", type=int, default=5)
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    args = parse_args()
    manifest = load_json(args.manifest)
    selected = args.video_ids or list(manifest.get("videos", {}))
    failures = []
    for video_id in selected:
        if video_id not in manifest.get("videos", {}):
            failures.append((video_id, "매니페스트에 없는 영상"))
            continue
        video = manifest["videos"][video_id]
        if video.get("review_status") == "excluded":
            continue
        source = PROJECT_ROOT / video["source"]
        events_path = args.audit_root / video_id / "stage_events.json"
        try:
            result = extract_stage_candidates(
                source,
                load_json(events_path),
                args.output_root / video_id,
                radius_frames=args.radius_frames,
                candidate_count=args.candidate_count,
            )
            print(
                f"[{video_id}] {len(result['stages'])}단계 후보 생성: "
                f"{args.output_root / video_id}"
            )
        except Exception as error:
            failures.append((video_id, str(error)))
            print(f"[{video_id}] 실패: {error}")
    print(f"완료={len(selected) - len(failures)} 실패={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
