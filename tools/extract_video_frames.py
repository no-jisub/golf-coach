import argparse
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_IMAGES_DIR = PROJECT_ROOT / "reference_data" / "raw_images"
RAW_VIDEOS_DIR = PROJECT_ROOT / "reference_data" / "raw_videos"

STAGES = [
    ("1", "address", "Address"),
    ("2", "takeaway", "Takeaway"),
    ("3", "backswing", "Backswing"),
    ("4", "top", "Top of Swing"),
    ("5", "downswing", "Downswing"),
    ("6", "impact", "Impact"),
    ("7", "follow_through", "Follow-through"),
    ("8", "finish", "Finish"),
]

KEY_TO_STAGE = {ord(key): stage for key, stage, _ in STAGES}
STAGE_LABELS = {stage: label for _, stage, label in STAGES}


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def draw_help(frame, frame_index, total_frames, paused, range_start, range_end, samples):
    # OpenCV 기본 글꼴은 한글 출력이 약해서 영어 안내로 표시합니다.
    range_text = "Range: none"
    if range_start is not None or range_end is not None:
        start_text = "-" if range_start is None else str(range_start + 1)
        end_text = "-" if range_end is None else str(range_end + 1)
        range_text = f"Range: {start_text} ~ {end_text} | samples: {samples}"

    lines = [
        f"Frame: {frame_index + 1}/{total_frames} | {'PAUSE' if paused else 'PLAY'}",
        range_text,
        "1 Address | 2 Takeaway | 3 Backswing | 4 Top",
        "5 Downswing | 6 Impact | 7 Follow-through | 8 Finish",
        "B: range start | E: range end | stage key saves range if set",
        "Space/P: play-pause | A/D or arrows: move | C: clear range | Q: quit",
    ]

    x, y = 16, 28
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        y += 28


def save_frame(frame, video_path, frame_index, stage, suffix=None):
    output_dir = RAW_IMAGES_DIR / stage
    output_dir.mkdir(parents=True, exist_ok=True)

    extra = "" if suffix is None else f"_{suffix}"
    output_path = output_dir / f"{video_path.stem}_frame_{frame_index:06d}{extra}.jpg"
    cv2.imwrite(str(output_path), frame)
    return output_path


def sample_frame_indexes(start_frame, end_frame, sample_count):
    start_frame, end_frame = sorted((start_frame, end_frame))
    if sample_count <= 1 or start_frame == end_frame:
        return [start_frame]

    step = (end_frame - start_frame) / (sample_count - 1)
    return [round(start_frame + step * index) for index in range(sample_count)]


def save_range_frames(capture, video_path, stage, start_frame, end_frame, sample_count):
    saved_paths = []
    frame_indexes = sample_frame_indexes(start_frame, end_frame, sample_count)

    for sample_number, sample_frame_index in enumerate(frame_indexes, start=1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, sample_frame_index)
        success, frame = capture.read()
        if not success:
            continue

        output_path = save_frame(
            frame,
            video_path,
            sample_frame_index,
            stage,
            suffix=f"sample_{sample_number:02d}",
        )
        saved_paths.append(output_path)

    return saved_paths


def open_video(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"동영상을 열 수 없습니다: {video_path}")
    return capture


def main():
    parser = argparse.ArgumentParser(
        description="풀스윙 동영상에서 8단계 기준 프레임을 골라 raw_images 폴더에 저장합니다."
    )
    parser.add_argument(
        "video",
        nargs="?",
        help="분석할 동영상 경로입니다. 생략하면 reference_data/raw_videos 안의 첫 번째 동영상을 사용합니다.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=15,
        help="구간 저장 시 뽑을 프레임 수입니다. 기본값은 15장입니다.",
    )
    args = parser.parse_args()

    if args.video:
        video_path = Path(args.video)
    else:
        videos = sorted(
            path
            for path in RAW_VIDEOS_DIR.rglob("*")
            if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        )
        if not videos:
            print("reference_data\\raw_videos 폴더에 동영상 파일이 없습니다.")
            print("예: reference_data\\raw_videos\\pro01_full_swing.mp4")
            return
        video_path = videos[0]

    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path

    capture = open_video(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    frame_index = 0
    paused = True
    range_start = None
    range_end = None
    samples = max(1, args.samples)

    print("동영상 프레임 선택을 시작합니다.")
    print("B로 구간 시작, E로 구간 끝을 지정한 뒤 1~8 키를 누르면 해당 단계에 여러 프레임을 저장합니다.")
    print("구간을 지정하지 않은 상태에서 1~8 키를 누르면 현재 프레임 1장만 저장합니다.")
    print("저장 후에는 py -3.12 tools\\extract_reference_poses.py 를 실행하세요.")

    while True:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success:
            break

        display = frame.copy()
        draw_help(display, frame_index, total_frames, paused, range_start, range_end, samples)
        cv2.imshow("Extract Golf Reference Frames", display)

        delay = 0 if paused else max(1, int(1000 / fps))
        key = cv2.waitKey(delay) & 0xFF

        if key in (ord("q"), 27):
            break
        if key in (ord(" "), ord("p")):
            paused = not paused
            continue
        if key == ord("b"):
            range_start = frame_index
            paused = True
            print(f"[RANGE START] frame {frame_index + 1}")
            continue
        if key == ord("e"):
            range_end = frame_index
            paused = True
            print(f"[RANGE END] frame {frame_index + 1}")
            continue
        if key == ord("c"):
            range_start = None
            range_end = None
            print("[RANGE CLEAR]")
            continue
        if key in (ord("a"), 81):
            paused = True
            frame_index = clamp(frame_index - 1, 0, total_frames - 1)
            continue
        if key in (ord("d"), 83):
            paused = True
            frame_index = clamp(frame_index + 1, 0, total_frames - 1)
            continue
        if key in KEY_TO_STAGE:
            stage = KEY_TO_STAGE[key]
            if range_start is not None and range_end is not None:
                saved_paths = save_range_frames(capture, video_path, stage, range_start, range_end, samples)
                print(f"[SAVE RANGE] {STAGE_LABELS[stage]}: {len(saved_paths)} frames")
                for output_path in saved_paths:
                    print(f"  - {output_path.relative_to(PROJECT_ROOT)}")
                range_start = None
                range_end = None
            else:
                output_path = save_frame(frame, video_path, frame_index, stage)
                print(f"[SAVE] {stage}: {output_path.relative_to(PROJECT_ROOT)}")
            continue

        if not paused:
            frame_index = clamp(frame_index + 1, 0, total_frames - 1)

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
