import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.swing_video import extract_video_landmarks, write_json


MODEL_PATH = PROJECT_ROOT / "pose_landmarker_full.task"
DEFAULT_SESSIONS_DIR = PROJECT_ROOT / "analysis_sessions"


def resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(
        description="저장된 골프 스윙 영상의 프레임별 MediaPipe 관절 좌표를 저장합니다."
    )
    parser.add_argument("video", help="분석할 정면 스윙 영상 경로")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="분석 결과 디렉터리. 기본값은 analysis_sessions/<영상 이름>입니다.",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=1,
        help="관절을 분석할 프레임 간격. 기본값 1은 모든 프레임을 분석합니다.",
    )
    parser.add_argument("--progress", action="store_true", help="처리 진행 상황을 표시합니다.")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = resolve_path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"영상 파일이 없습니다: {video_path}")
    output_dir = (
        resolve_path(args.output_dir).resolve()
        if args.output_dir
        else DEFAULT_SESSIONS_DIR / video_path.stem
    )
    output_path = output_dir / "frame_landmarks.json"
    payload = extract_video_landmarks(
        video_path,
        MODEL_PATH,
        sample_step=args.sample_step,
        progress=args.progress,
    )
    write_json(output_path, payload)
    print(f"영상: {video_path}")
    print(
        f"프레임: {payload['video']['decoded_frames']}, "
        f"자세 감지: {payload['sampling']['detected_frames']}/"
        f"{payload['sampling']['sampled_frames']}"
    )
    print(f"저장: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
