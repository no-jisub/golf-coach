import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.guide_alignment import STAGE_KEYS  # noqa: E402
from utils.stage_candidate_review import finalize_stage_frame_review  # noqa: E402
from utils.stage_video_review import (  # noqa: E402
    FrameNavigator,
    build_review_progress,
    load_review_progress,
    save_review_progress,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "reference_data" / "swing_stage_ground_truth.json"
DEFAULT_CANDIDATE_ROOT = PROJECT_ROOT / "analysis_sessions" / "stage_candidates"
DEFAULT_PROGRESS_ROOT = PROJECT_ROOT / "analysis_sessions" / "stage_review_progress"
LEFT_KEYS = {81, 2424832}
RIGHT_KEYS = {83, 2555904}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "프로 영상 전체를 재생·탐색하면서 현재 프레임을 8단계 정답으로 지정합니다."
        )
    )
    parser.add_argument("video_id")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--progress-root", type=Path, default=DEFAULT_PROGRESS_ROOT)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--note", default="")
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_frame(capture, frame_index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    if not ok:
        raise OSError(f"영상 프레임을 읽지 못했습니다: {frame_index}")
    return frame


def zoom_frame(frame, zoom):
    zoom = max(1.0, min(float(zoom), 3.0))
    if zoom <= 1.0:
        return frame
    height, width = frame.shape[:2]
    crop_width = max(1, int(width / zoom))
    crop_height = max(1, int(height / zoom))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return frame[top : top + crop_height, left : left + crop_width]


def fit_frame(frame, width=1280, height=720):
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    scale = min(width / frame.shape[1], (height - 115) / frame.shape[0])
    resized = cv2.resize(
        frame,
        (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))),
    )
    left = (width - resized.shape[1]) // 2
    top = 75 + (height - 115 - resized.shape[0]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def render_review_frame(
    frame,
    *,
    navigator,
    stage_index,
    selections,
    automatic_frames,
    playing,
    zoom,
):
    stage_key = STAGE_KEYS[stage_index]
    canvas = fit_frame(zoom_frame(frame, zoom))
    selected = selections.get(stage_key)
    previous_stage = STAGE_KEYS[stage_index - 1] if stage_index > 0 else None
    next_stage = (
        STAGE_KEYS[stage_index + 1]
        if stage_index + 1 < len(STAGE_KEYS)
        else None
    )
    timestamp = (
        "-" if navigator.timestamp_ms is None else f"{navigator.timestamp_ms / 1000:.3f}s"
    )
    lines = [
        (
            f"Stage {stage_index + 1}/8 {stage_key} | frame "
            f"{navigator.current_frame}/{navigator.total_frames - 1} | {timestamp} | "
            f"{'PLAY' if playing else 'PAUSE'} | zoom={zoom:.1f}x"
        ),
        (
            f"selected={selected if selected is not None else '-'} | "
            f"auto={automatic_frames.get(stage_key, '-')} | "
            f"prev={selections.get(previous_stage, '-') if previous_stage else '-'} | "
            f"next={selections.get(next_stage, '-') if next_stage else '-'}"
        ),
        (
            "Space play/pause | arrows/a,d 1f | j,l 5f | u,o 10f | "
            "[ ] stage | 1-8 jump | c candidate | m mark | +/- zoom | q save"
        ),
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (16, 24 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (80, 255, 120) if index == 0 else (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    completed = len(selections)
    cv2.putText(
        canvas,
        f"reviewed stages: {completed}/8",
        (16, canvas.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 255, 120) if completed == 8 else (80, 210, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def load_candidate_context(candidate_path):
    if not candidate_path.exists():
        return {}, {}
    manifest = load_json(candidate_path)
    automatic = {
        stage: int(record["automatic_frame"])
        for stage, record in manifest.get("stages", {}).items()
        if record.get("automatic_frame") is not None
    }
    candidates = {
        stage: [int(item["frame_index"]) for item in record.get("candidates", [])]
        for stage, record in manifest.get("stages", {}).items()
    }
    return automatic, candidates


def main():
    args = parse_args()
    manifest = load_json(args.manifest)
    if args.video_id not in manifest.get("videos", {}):
        raise SystemExit(f"정답 매니페스트에 없는 영상입니다: {args.video_id}")
    video = manifest["videos"][args.video_id]
    source_path = Path(video["source"])
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {source_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    navigator = FrameNavigator(total_frames, fps=fps)

    candidate_path = args.candidate_root / args.video_id / "candidate_manifest.json"
    automatic_frames, candidates = load_candidate_context(candidate_path)
    progress_path = args.progress_root / f"{args.video_id}.json"
    progress = load_review_progress(progress_path, video_id=args.video_id)
    existing_events = video.get("events", {})
    selections = {
        stage: int(frame)
        for stage, frame in existing_events.items()
        if isinstance(frame, int)
    }
    if progress:
        selections.update(progress.get("selections", {}))
        stage_index = int(progress.get("current_stage_index", 0))
        navigator.seek(progress.get("current_frame", 0))
    else:
        stage_index = 0
        navigator.seek(
            selections.get(STAGE_KEYS[0], automatic_frames.get(STAGE_KEYS[0], 0))
        )

    candidate_positions = {stage: 0 for stage in STAGE_KEYS}
    playing = False
    zoom = 1.0
    window = "Golf Coach - Full Timeline Stage Review"
    trackbar_name = "Frame"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 820)
    cv2.createTrackbar(trackbar_name, window, navigator.current_frame, total_frames - 1, lambda _: None)
    try:
        while True:
            trackbar_frame = cv2.getTrackbarPos(trackbar_name, window)
            if not playing and trackbar_frame != navigator.current_frame:
                navigator.seek(trackbar_frame)
            frame = read_frame(capture, navigator.current_frame)
            cv2.imshow(
                window,
                render_review_frame(
                    frame,
                    navigator=navigator,
                    stage_index=stage_index,
                    selections=selections,
                    automatic_frames=automatic_frames,
                    playing=playing,
                    zoom=zoom,
                ),
            )
            cv2.setTrackbarPos(trackbar_name, window, navigator.current_frame)
            delay = max(1, round(1000 / fps)) if playing and fps > 0 else 30
            key = cv2.waitKeyEx(delay)
            if key == -1:
                if playing:
                    previous = navigator.current_frame
                    navigator.step(1)
                    if navigator.current_frame == previous:
                        playing = False
                continue
            if key == ord("q"):
                break
            if key == ord(" "):
                playing = not playing
                continue
            if key in LEFT_KEYS or key == ord("a"):
                playing = False
                navigator.step(-1)
                continue
            if key in RIGHT_KEYS or key == ord("d"):
                playing = False
                navigator.step(1)
                continue
            if key == ord("j"):
                playing = False
                navigator.step(-5)
                continue
            if key == ord("l"):
                playing = False
                navigator.step(5)
                continue
            if key == ord("u"):
                playing = False
                navigator.step(-10)
                continue
            if key == ord("o"):
                playing = False
                navigator.step(10)
                continue
            if key == ord("["):
                stage_index = max(0, stage_index - 1)
            elif key == ord("]"):
                stage_index = min(len(STAGE_KEYS) - 1, stage_index + 1)
            elif ord("1") <= key <= ord("8"):
                stage_index = key - ord("1")
            elif key == ord("c"):
                stage_key = STAGE_KEYS[stage_index]
                stage_candidates = candidates.get(stage_key, [])
                if stage_candidates:
                    position = candidate_positions[stage_key] % len(stage_candidates)
                    navigator.seek(stage_candidates[position])
                    candidate_positions[stage_key] = position + 1
                continue
            elif key == ord("m"):
                selections[STAGE_KEYS[stage_index]] = navigator.current_frame
                if stage_index + 1 < len(STAGE_KEYS):
                    stage_index += 1
            elif key in {ord("+"), ord("=")}:
                zoom = min(3.0, zoom + 0.25)
                continue
            elif key in {ord("-"), ord("_")}:
                zoom = max(1.0, zoom - 0.25)
                continue
            else:
                continue

            stage_key = STAGE_KEYS[stage_index]
            navigator.seek(
                selections.get(stage_key, automatic_frames.get(stage_key, navigator.current_frame))
            )
            save_review_progress(
                progress_path,
                build_review_progress(
                    video_id=args.video_id,
                    source_video=source_path,
                    selections=selections,
                    current_stage_index=stage_index,
                    current_frame=navigator.current_frame,
                ),
            )
    finally:
        capture.release()
        cv2.destroyAllWindows()

    save_review_progress(
        progress_path,
        build_review_progress(
            video_id=args.video_id,
            source_video=source_path,
            selections=selections,
            current_stage_index=stage_index,
            current_frame=navigator.current_frame,
        ),
    )
    if set(selections) != set(STAGE_KEYS):
        missing = [stage for stage in STAGE_KEYS if stage not in selections]
        print(f"진행 상태 저장: {progress_path}")
        print(f"아직 선택하지 않은 단계: {', '.join(missing)}")
        return 0

    updated = finalize_stage_frame_review(
        manifest,
        video_id=args.video_id,
        selections=selections,
        reviewed_by=args.reviewer,
        note=args.note,
        project_root=PROJECT_ROOT,
        total_frames=total_frames,
        review_source={
            "type": "full_timeline_frame_selection",
            "candidate_manifest": str(candidate_path) if candidate_path.exists() else None,
            "progress_file": str(progress_path),
        },
    )
    write_json(args.manifest, updated)
    if progress_path.exists():
        progress_path.unlink()
    print(f"{args.video_id} 검수 완료: {selections}")
    print(f"정답 매니페스트: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
