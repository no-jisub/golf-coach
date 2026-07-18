import json
from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


LANDMARK_SCHEMA = "golf-coach-swing-video-landmarks-v1"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def create_video_pose_landmarker(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"MediaPipe 모델 파일이 없습니다: {model_path}")
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.45,
        min_pose_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    return vision.PoseLandmarker.create_from_options(options)


def serialize_landmarks(landmarks):
    serialized = []
    for index, landmark in enumerate(landmarks or []):
        point = {
            "index": index,
            "x": round(float(landmark.x), 7),
            "y": round(float(landmark.y), 7),
            "z": round(float(landmark.z), 7),
        }
        for name in ("visibility", "presence"):
            value = getattr(landmark, name, None)
            if value is not None:
                point[name] = round(float(value), 7)
        serialized.append(point)
    return serialized


def make_frame_record(frame_index, timestamp_ms, result):
    detected = bool(result.pose_landmarks)
    pose = result.pose_landmarks[0] if detected else []
    world = (
        result.pose_world_landmarks[0]
        if detected and getattr(result, "pose_world_landmarks", None)
        else []
    )
    return {
        "frame_index": int(frame_index),
        "timestamp_ms": int(timestamp_ms),
        "detected": detected,
        "landmarks": serialize_landmarks(pose),
        "world_landmarks": serialize_landmarks(world),
    }


def extract_video_landmarks(
    video_path,
    model_path,
    *,
    sample_step=1,
    progress=False,
    landmarker_factory=create_video_pose_landmarker,
):
    """Read a saved video and retain pose data for every sampled timeline frame."""
    video_path = Path(video_path).resolve()
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"지원하지 않는 영상 확장자입니다: {video_path.suffix}")
    sample_step = int(sample_step)
    if sample_step < 1:
        raise ValueError("sample_step은 1 이상이어야 합니다.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    declared_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    records = []
    decoded_frames = 0
    detected_frames = 0
    last_timestamp_ms = -1

    try:
        with landmarker_factory(model_path) as landmarker:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frame_index = decoded_frames
                decoded_frames += 1
                if frame_index % sample_step != 0:
                    continue

                timestamp_ms = int(round(frame_index / fps * 1000.0))
                timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                record = make_frame_record(frame_index, timestamp_ms, result)
                records.append(record)
                detected_frames += int(record["detected"])

                if progress and len(records) % 30 == 0:
                    print(
                        f"[POSE] decoded={decoded_frames}/{declared_total or '?'} "
                        f"sampled={len(records)} detected={detected_frames}"
                    )
    finally:
        capture.release()

    if decoded_frames == 0:
        raise ValueError(f"영상에서 프레임을 읽지 못했습니다: {video_path}")

    total_frames = declared_total if declared_total > 0 else decoded_frames
    return {
        "schema": LANDMARK_SCHEMA,
        "source_video": str(video_path),
        "video": {
            "fps": round(fps, 6),
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "decoded_frames": decoded_frames,
            "duration_sec": round(decoded_frames / fps, 6),
        },
        "sampling": {
            "sample_step": sample_step,
            "sampled_frames": len(records),
            "detected_frames": detected_frames,
            "detection_ratio": round(detected_frames / len(records), 6) if records else 0.0,
        },
        "frames": records,
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
