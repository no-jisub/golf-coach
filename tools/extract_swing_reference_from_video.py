import argparse
from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
RAW_IMAGES_DIR = PROJECT_ROOT / "reference_data" / "raw_images"
RAW_VIDEOS_DIR = PROJECT_ROOT / "reference_data" / "raw_videos"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from detect_swing_events_mediapipe import detect_swing_events  # noqa: E402
from extract_reference_poses import create_landmarker, extract_image, save_result  # noqa: E402
from extract_reference_shafts import process_json  # noqa: E402
from visualize_reference_poses import visualize_file  # noqa: E402


STAGE_EVENTS = [
    ("address", "Address", 0.08),
    ("takeaway", "Takeaway", 0.22),
    ("backswing", "Backswing", 0.36),
    ("top", "Top of Swing", 0.48),
    ("downswing", "Downswing", 0.60),
    ("impact", "Impact", 0.70),
    ("follow_through", "Follow-through", 0.82),
    ("finish", "Finish", 0.94),
]

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def find_default_video():
    videos = sorted(
        path
        for path in RAW_VIDEOS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    return videos[0] if videos else None


def resolve_video_path(video_arg):
    if video_arg:
        video_path = Path(video_arg)
        if not video_path.is_absolute():
            video_path = PROJECT_ROOT / video_path
        return video_path

    return find_default_video()


def open_video(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"동영상을 열 수 없습니다: {video_path}")
    return capture


def get_video_info(capture):
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if total_frames <= 0:
        raise ValueError("동영상 프레임 수를 읽을 수 없습니다.")
    return total_frames, fps


def event_frame_index(total_frames, ratio):
    return max(0, min(total_frames - 1, round((total_frames - 1) * ratio)))


def get_ratio_event_frames(total_frames):
    return {
        stage: event_frame_index(total_frames, ratio)
        for stage, _, ratio in STAGE_EVENTS
    }, {"source": "ratio"}


def get_mediapipe_event_frames(video_path, total_frames, sample_step):
    events, diagnostics, _, _ = detect_swing_events(video_path, sample_step=sample_step, progress=False)
    if events is None:
        print(f"[WARN] MediaPipe 이벤트 탐지 실패: {diagnostics}. 비율 기반 후보로 fallback합니다.")
        return get_ratio_event_frames(total_frames)

    diagnostics = dict(diagnostics)
    diagnostics["source"] = "mediapipe"
    return events, diagnostics


def save_stage_frame(capture, video_path, stage, frame_index, prefix, overwrite):
    output_dir = RAW_IMAGES_DIR / stage
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{prefix}_{stage}_auto.jpg"

    if output_path.exists() and not overwrite:
        return output_path, False

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = capture.read()
    if not success:
        raise RuntimeError(f"프레임을 읽을 수 없습니다: {frame_index}")

    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"프레임 저장 실패: {output_path}")

    return output_path, True


def extract_pose_json(landmarker, stage, image_path):
    result = extract_image(landmarker, stage, image_path)
    output_path = save_result(result, image_path)
    return output_path, result.get("detected", False)


def process_video(video_path, prefix, overwrite, dry_run, event_source, sample_step):
    capture = open_video(video_path)
    total_frames, fps = get_video_info(capture)
    duration_sec = total_frames / fps

    print(f"[VIDEO] {video_path.relative_to(PROJECT_ROOT) if video_path.is_relative_to(PROJECT_ROOT) else video_path}")
    print(f"[INFO] frames={total_frames}, fps={fps:.2f}, duration={duration_sec:.2f}s")

    if event_source == "mediapipe":
        print("[INFO] MediaPipe 관절 움직임 기반으로 8단계 후보 프레임을 탐지합니다.")
        event_frames, diagnostics = get_mediapipe_event_frames(video_path, total_frames, sample_step)
    else:
        print("[INFO] 영상 진행률 기반 후보 프레임을 사용합니다.")
        event_frames, diagnostics = get_ratio_event_frames(total_frames)
    print(f"[EVENT] {diagnostics}")

    saved_images = []
    try:
        for stage, label, ratio in STAGE_EVENTS:
            frame_index = max(0, min(total_frames - 1, int(event_frames.get(stage, event_frame_index(total_frames, ratio)))))
            image_path, saved = save_stage_frame(capture, video_path, stage, frame_index, prefix, overwrite)
            saved_images.append((stage, image_path))
            action = "SAVE" if saved else "KEEP"
            print(f"[{action}] {label:15} frame={frame_index + 1} -> {image_path.relative_to(PROJECT_ROOT)}")
    finally:
        capture.release()

    if dry_run:
        print("[DRY RUN] 프레임만 확인하고 JSON 추출은 건너뜁니다.")
        return

    detected_count = 0
    with create_landmarker() as landmarker:
        for stage, image_path in saved_images:
            json_path, detected = extract_pose_json(landmarker, stage, image_path)
            if detected:
                detected_count += 1
                print(f"[POSE OK] {json_path.relative_to(PROJECT_ROOT)}")
            else:
                print(f"[POSE FAIL] {json_path.relative_to(PROJECT_ROOT)}")

            process_json(json_path, write=True)
            visualize_file(json_path)

    print()
    print(f"처리 단계: {len(saved_images)}")
    print(f"포즈 인식 성공: {detected_count}")
    print("검수 위치:")
    print("- reference_data\\debug_overlay")
    print("- reference_data\\debug_shaft_overlay")
    print("주의: 자동 후보 프레임이므로 앱 기준으로 반영하기 전 반드시 오버레이를 검수하세요.")


def main():
    parser = argparse.ArgumentParser(
        description="로컬 풀스윙 영상에서 8단계 후보 프레임, 관절 JSON, 샤프트 후보를 자동 추출합니다."
    )
    parser.add_argument(
        "video",
        nargs="?",
        help="분석할 동영상 경로입니다. 생략하면 reference_data/raw_videos 안의 첫 번째 동영상을 사용합니다.",
    )
    parser.add_argument(
        "--prefix",
        help="저장 파일명 접두어입니다. 기본값은 동영상 파일명입니다.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 저장된 같은 이름의 자동 프레임/JPEG을 덮어씁니다.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="프레임 캡처만 수행하고 Pose/shaft JSON 추출은 하지 않습니다.",
    )
    parser.add_argument(
        "--event-source",
        choices=("mediapipe", "ratio"),
        default="mediapipe",
        help="8단계 후보 프레임 선택 방식입니다. 기본값은 mediapipe입니다.",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=2,
        help="MediaPipe 이벤트 탐지 시 몇 프레임마다 분석할지 설정합니다. 기본값은 2입니다.",
    )
    args = parser.parse_args()

    video_path = resolve_video_path(args.video)
    if video_path is None:
        print("reference_data\\raw_videos 폴더에 동영상 파일이 없습니다.")
        print("예: reference_data\\raw_videos\\pro03_full_swing.mp4")
        return
    if not video_path.exists():
        raise FileNotFoundError(f"동영상 파일이 없습니다: {video_path}")

    prefix = args.prefix or video_path.stem
    process_video(video_path, prefix, args.overwrite, args.dry_run, args.event_source, max(1, args.sample_step))


if __name__ == "__main__":
    main()
