"""Rotate a looped swing video so its address frame starts the timeline."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.loop_video_normalizer import frame_from_seconds, normalize_loop_video


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "반복 스윙 영상을 지정 위치부터 끝까지, 이어서 처음부터 지정 위치 "
            "직전까지 기록해 선형 영상으로 만듭니다."
        )
    )
    parser.add_argument("input", type=Path, help="원본 반복 영상")
    parser.add_argument("output", type=Path, help="새로 만들 선형 영상")
    split_group = parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument(
        "--split-frame",
        type=int,
        help="새 영상의 0번이 될 원본 프레임 번호",
    )
    split_group.add_argument(
        "--split-sec",
        type=float,
        help="새 영상의 시작점이 될 원본 시간(초)",
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        help="원본 프레임에서 새 프레임으로의 대응표를 저장할 JSON 경로",
    )
    parser.add_argument("--progress", action="store_true", help="진행률 표시")
    return parser.parse_args(argv)


def _resolve(path):
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_fps(video_path):
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"영상을 열 수 없습니다: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise ValueError(f"영상 FPS를 확인할 수 없습니다: {video_path}")
        return fps
    finally:
        capture.release()


def main(argv=None):
    args = parse_args(argv)
    input_path = _resolve(args.input)
    output_path = _resolve(args.output)
    if args.split_frame is not None:
        split_frame = args.split_frame
    else:
        split_frame = frame_from_seconds(args.split_sec, _read_fps(input_path))

    result = normalize_loop_video(
        input_path,
        output_path,
        split_frame,
        progress=args.progress,
    )
    if args.mapping_json:
        mapping_path = _resolve(args.mapping_json)
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        mapping_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"프레임 대응표: {mapping_path}")

    print(f"원본: {result['source_video']}")
    print(f"결과: {result['normalized_video']}")
    print(
        f"분할점: {result['split']['source_frame']}번 "
        f"({result['split']['source_sec']:.3f}초)"
    )
    print(
        f"영상: {result['video']['total_frames']}프레임, "
        f"{result['video']['fps']:.3f}FPS, "
        f"{result['video']['width']}x{result['video']['height']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
