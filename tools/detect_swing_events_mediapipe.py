import argparse
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "pose_landmarker_full.task"
RAW_VIDEOS_DIR = PROJECT_ROOT / "reference_data" / "raw_videos"

STAGE_ORDER = [
    "address",
    "takeaway",
    "backswing",
    "top",
    "downswing",
    "impact",
    "follow_through",
    "finish",
]

STAGE_LABELS = {
    "address": "Address",
    "takeaway": "Takeaway",
    "backswing": "Backswing",
    "top": "Top of Swing",
    "downswing": "Downswing",
    "impact": "Impact",
    "follow_through": "Follow-through",
    "finish": "Finish",
}

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def create_video_landmarker():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 필요합니다: {MODEL_PATH}")

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.45,
        min_pose_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    return vision.PoseLandmarker.create_from_options(options)


def resolve_video_path(video_arg):
    if video_arg:
        video_path = Path(video_arg)
        if not video_path.is_absolute():
            video_path = PROJECT_ROOT / video_path
        return video_path

    videos = sorted(
        path
        for path in RAW_VIDEOS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    return videos[0] if videos else None


def landmark_visible(landmark, min_visibility=0.35):
    return getattr(landmark, "visibility", 1.0) >= min_visibility


def midpoint(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def extract_frame_features(frame_index, landmarks):
    required = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ANKLE, RIGHT_ANKLE]
    if any(not landmark_visible(landmarks[index]) for index in required):
        return None

    wrist_points = []
    wrist_visibility = []
    for index in (LEFT_WRIST, RIGHT_WRIST):
        landmark = landmarks[index]
        if landmark_visible(landmark, 0.25):
            wrist_points.append((landmark.x, landmark.y))
            wrist_visibility.append(getattr(landmark, "visibility", 1.0))

    if not wrist_points:
        return None

    left_shoulder = (landmarks[LEFT_SHOULDER].x, landmarks[LEFT_SHOULDER].y)
    right_shoulder = (landmarks[RIGHT_SHOULDER].x, landmarks[RIGHT_SHOULDER].y)
    left_ankle = (landmarks[LEFT_ANKLE].x, landmarks[LEFT_ANKLE].y)
    right_ankle = (landmarks[RIGHT_ANKLE].x, landmarks[RIGHT_ANKLE].y)

    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    ankle_mid = midpoint(left_ankle, right_ankle)
    body_height = abs(ankle_mid[1] - shoulder_mid[1])
    if body_height < 0.10:
        return None

    wrist = tuple(np.mean(wrist_points, axis=0))
    return {
        "frame": frame_index,
        "wrist_x": float(wrist[0]),
        "wrist_y": float(wrist[1]),
        "shoulder_x": float(shoulder_mid[0]),
        "shoulder_y": float(shoulder_mid[1]),
        "body_height": float(body_height),
        "wrist_visibility": float(np.mean(wrist_visibility)),
    }


def read_pose_features(video_path, sample_step=2, progress=False):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"동영상을 열 수 없습니다: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    features = []
    start_time = time.monotonic()

    with create_video_landmarker() as landmarker:
        frame_index = 0
        while frame_index < total_frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(frame_index / fps * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                frame_features = extract_frame_features(frame_index, result.pose_landmarks[0])
                if frame_features is not None:
                    features.append(frame_features)

            if progress and frame_index % max(sample_step * 30, 1) == 0:
                elapsed = time.monotonic() - start_time
                print(f"[SCAN] frame={frame_index + 1}/{total_frames}, detected={len(features)}, elapsed={elapsed:.1f}s")

            frame_index += max(1, sample_step)

    capture.release()
    return features, total_frames, fps


def moving_average(values, window=5):
    if len(values) == 0:
        return values
    window = max(1, min(window, len(values)))
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def nearest_feature_index(features, frame_index):
    return min(range(len(features)), key=lambda index: abs(features[index]["frame"] - frame_index))


def nearest_frame(features, target_index):
    target_index = max(0, min(len(features) - 1, target_index))
    return int(features[target_index]["frame"])


def index_at_fraction(start_index, end_index, fraction):
    if end_index <= start_index:
        return start_index
    return round(start_index + (end_index - start_index) * fraction)


def choose_low_motion_index(features, speeds, start_index, end_index, prefer_low_hands=False):
    start_index = max(0, start_index)
    end_index = min(len(features) - 1, end_index)
    if end_index <= start_index:
        return start_index

    best_index = start_index
    best_score = float("inf")
    max_speed = max(float(np.max(speeds)), 1e-6) if len(speeds) else 1.0
    for index in range(start_index, end_index + 1):
        speed = speeds[index] if index < len(speeds) else speeds[-1]
        score = speed / max_speed
        if prefer_low_hands:
            # y가 클수록 손이 아래쪽입니다. address 후보에서는 손이 낮은 프레임을 선호합니다.
            score -= features[index]["wrist_y"] * 0.35
        if score < best_score:
            best_score = score
            best_index = index
    return best_index


def detect_active_segment(features, speeds):
    if len(features) < 8 or len(speeds) == 0:
        return 0, max(0, len(features) - 1)

    max_speed = float(np.max(speeds))
    if max_speed <= 1e-6:
        return 0, len(features) - 1

    threshold = max(max_speed * 0.18, float(np.percentile(speeds, 70)) * 0.55)
    active = [index for index, speed in enumerate(speeds) if speed >= threshold]
    if not active:
        return 0, len(features) - 1

    start_index = max(0, active[0] - 3)
    end_index = min(len(features) - 1, active[-1] + 5)
    if end_index - start_index < 8:
        return 0, len(features) - 1
    return start_index, end_index


def frame_index_at_ratio(features, total_frames, ratio):
    target_frame = round((total_frames - 1) * ratio)
    return nearest_feature_index(features, target_frame)


def choose_top_index(features, wrist_y, start_index, end_index):
    start_index = max(0, start_index)
    end_index = min(len(features) - 1, end_index)
    if end_index <= start_index:
        return start_index
    return min(range(start_index, end_index + 1), key=lambda index: wrist_y[index])


def choose_impact_index(features, wrist_x, wrist_y, speeds, top_index, address_index, total_frames):
    # 임팩트는 탑 이후 짧은 구간에서 빠르게 내려오며 address 손 높이에 가까워지는 순간입니다.
    min_index = max(top_index + 1, frame_index_at_ratio(features, total_frames, 0.50))
    max_index = max(min_index, min(len(features) - 1, frame_index_at_ratio(features, total_frames, 0.78)))
    address_x = wrist_x[address_index]
    address_y = wrist_y[address_index]
    max_speed = max(float(np.max(speeds)), 1e-6)

    best_index = min_index
    best_score = float("inf")
    for index in range(min_index, max_index + 1):
        y_score = abs(wrist_y[index] - address_y)
        x_score = abs(wrist_x[index] - address_x)
        speed_bonus = speeds[index] / max_speed
        progress_penalty = (index - min_index) / max(max_index - min_index, 1) * 0.10
        score = y_score * 1.2 + x_score * 0.35 + progress_penalty - speed_bonus * 0.18
        if score < best_score:
            best_score = score
            best_index = index

    return best_index


def detect_events_from_features(features, total_frames):
    if len(features) < 8:
        return None, {"reason": "not_enough_pose_frames", "detected_frames": len(features)}

    wrist_x = moving_average(np.array([feature["wrist_x"] for feature in features], dtype=float), 5)
    wrist_y = moving_average(np.array([feature["wrist_y"] for feature in features], dtype=float), 5)
    body_height = moving_average(np.array([feature["body_height"] for feature in features], dtype=float), 5)

    dx = np.diff(wrist_x, prepend=wrist_x[0])
    dy = np.diff(wrist_y, prepend=wrist_y[0])
    speeds = np.sqrt(dx * dx + dy * dy) / np.maximum(body_height, 1e-6)
    speeds = moving_average(speeds, 5)

    # 영상마다 앞뒤 대기 시간이 달라서 전체 active segment를 믿기보다,
    # 골프 스윙의 일반적인 시간 순서 안에서 손목 궤적을 고릅니다.
    address_start = 0
    address_end = frame_index_at_ratio(features, total_frames, 0.22)
    address_index = choose_low_motion_index(features, speeds, address_start, address_end, prefer_low_hands=True)

    top_start = max(address_index + 1, frame_index_at_ratio(features, total_frames, 0.18))
    top_end = frame_index_at_ratio(features, total_frames, 0.68)
    top_index = choose_top_index(features, wrist_y, top_start, top_end)

    impact_index = choose_impact_index(features, wrist_x, wrist_y, speeds, top_index, address_index, total_frames)
    if impact_index <= top_index:
        impact_index = min(len(features) - 1, top_index + 1)

    finish_start = max(impact_index + 1, frame_index_at_ratio(features, total_frames, 0.72))
    finish_index = choose_low_motion_index(features, speeds, finish_start, len(features) - 1, prefer_low_hands=False)
    if finish_index <= impact_index:
        finish_index = min(len(features) - 1, impact_index + 1)

    takeaway_index = index_at_fraction(address_index, top_index, 0.32)
    backswing_index = index_at_fraction(address_index, top_index, 0.68)
    downswing_index = index_at_fraction(top_index, impact_index, 0.48)
    follow_index = index_at_fraction(impact_index, finish_index, 0.45)

    ordered_indexes = [
        address_index,
        takeaway_index,
        backswing_index,
        top_index,
        downswing_index,
        impact_index,
        follow_index,
        finish_index,
    ]

    for index in range(1, len(ordered_indexes)):
        if ordered_indexes[index] <= ordered_indexes[index - 1]:
            ordered_indexes[index] = min(len(features) - 1, ordered_indexes[index - 1] + 1)

    events = {
        stage: nearest_frame(features, feature_index)
        for stage, feature_index in zip(STAGE_ORDER, ordered_indexes)
    }
    diagnostics = {
        "pose_frames": len(features),
        "total_frames": total_frames,
        "address_frame": int(features[address_index]["frame"]),
        "top_frame": int(features[top_index]["frame"]),
        "impact_frame": int(features[impact_index]["frame"]),
        "finish_frame": int(features[finish_index]["frame"]),
        "max_speed": round(max(float(np.max(speeds)), 1e-6), 5),
    }
    return events, diagnostics

def detect_swing_events(video_path, sample_step=2, progress=False):
    features, total_frames, fps = read_pose_features(video_path, sample_step=sample_step, progress=progress)
    events, diagnostics = detect_events_from_features(features, total_frames)
    return events, diagnostics, total_frames, fps


def main():
    parser = argparse.ArgumentParser(description="MediaPipe 관절 움직임 기반으로 골프 스윙 8단계 후보 프레임을 탐지합니다.")
    parser.add_argument("video", nargs="?", help="분석할 동영상 경로입니다.")
    parser.add_argument("--sample-step", type=int, default=2, help="몇 프레임마다 분석할지 설정합니다. 기본값은 2입니다.")
    parser.add_argument("--progress", action="store_true", help="분석 진행 상황을 출력합니다.")
    args = parser.parse_args()

    video_path = resolve_video_path(args.video)
    if video_path is None:
        print("reference_data\\raw_videos 폴더에 동영상 파일이 없습니다.")
        return
    if not video_path.exists():
        raise FileNotFoundError(f"동영상 파일이 없습니다: {video_path}")

    events, diagnostics, total_frames, fps = detect_swing_events(video_path, args.sample_step, args.progress)
    print(f"[VIDEO] {video_path.relative_to(PROJECT_ROOT) if video_path.is_relative_to(PROJECT_ROOT) else video_path}")
    print(f"[INFO] frames={total_frames}, fps={fps:.2f}")
    print(f"[DIAG] {diagnostics}")
    if events is None:
        print("[FAIL] 이벤트를 탐지하지 못했습니다.")
        return

    for stage in STAGE_ORDER:
        print(f"{stage:15} frame={events[stage] + 1}")


if __name__ == "__main__":
    main()
