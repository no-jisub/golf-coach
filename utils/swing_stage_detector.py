import math
from pathlib import Path

import cv2
import numpy as np

from utils.guide_alignment import STAGE_KEYS


STAGE_SCHEMA = "golf-coach-swing-stage-events-v1"
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def _point_map(frame_record, min_visibility=0.25):
    points = {}
    for point in frame_record.get("landmarks", []):
        if point.get("visibility", 1.0) >= min_visibility:
            points[int(point["index"])] = (float(point["x"]), float(point["y"]))
    return points


def _midpoint(point_a, point_b):
    return ((point_a[0] + point_b[0]) / 2.0, (point_a[1] + point_b[1]) / 2.0)


def _distance(point_a, point_b):
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def extract_motion_features(landmark_payload):
    required = {
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_WRIST,
        RIGHT_WRIST,
        LEFT_HIP,
        RIGHT_HIP,
        LEFT_ANKLE,
        RIGHT_ANKLE,
    }
    features = []
    for frame in landmark_payload.get("frames", []):
        if not frame.get("detected"):
            continue
        points = _point_map(frame)
        if not required.issubset(points):
            continue

        shoulder_mid = _midpoint(points[LEFT_SHOULDER], points[RIGHT_SHOULDER])
        hip_mid = _midpoint(points[LEFT_HIP], points[RIGHT_HIP])
        ankle_mid = _midpoint(points[LEFT_ANKLE], points[RIGHT_ANKLE])
        wrist_mid = _midpoint(points[LEFT_WRIST], points[RIGHT_WRIST])
        body_height = _distance(shoulder_mid, ankle_mid)
        shoulder_width = _distance(points[LEFT_SHOULDER], points[RIGHT_SHOULDER])
        if body_height < 0.1 or shoulder_width < 0.02:
            continue

        shoulder_delta = (
            points[RIGHT_SHOULDER][0] - points[LEFT_SHOULDER][0],
            points[RIGHT_SHOULDER][1] - points[LEFT_SHOULDER][1],
        )
        features.append(
            {
                "frame_index": int(frame["frame_index"]),
                "timestamp_ms": int(frame["timestamp_ms"]),
                "wrist_x": wrist_mid[0],
                "wrist_y": wrist_mid[1],
                "shoulder_angle": math.degrees(
                    math.atan2(shoulder_delta[1], shoulder_delta[0])
                ),
                "shoulder_width_ratio": shoulder_width / body_height,
                "hip_x": hip_mid[0],
                "body_height": body_height,
            }
        )
    return features


def _smooth(values, window=5):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return values
    window = max(1, min(int(window), len(values)))
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _argmin_range(values, start, end):
    start = max(0, int(start))
    end = min(len(values) - 1, int(end))
    if end <= start:
        return start
    return min(range(start, end + 1), key=lambda index: float(values[index]))


def _choose_path_fraction(wrist_x, wrist_y, start, end, fraction):
    if end <= start:
        return start
    distances = [0.0]
    for index in range(start + 1, end + 1):
        distances.append(
            distances[-1]
            + math.hypot(
                wrist_x[index] - wrist_x[index - 1],
                wrist_y[index] - wrist_y[index - 1],
            )
        )
    target = distances[-1] * fraction
    offset = min(range(len(distances)), key=lambda index: abs(distances[index] - target))
    return start + offset


def _force_strict_order(indexes, feature_count):
    ordered = []
    count = len(indexes)
    for position, value in enumerate(indexes):
        minimum = position if position == 0 else ordered[-1] + 1
        maximum = feature_count - (count - position)
        ordered.append(max(minimum, min(int(value), maximum)))
    return ordered


def detect_stage_events(landmark_payload):
    """Detect eight ordered face-on swing events from wrist and shoulder motion."""
    features = extract_motion_features(landmark_payload)
    if len(features) < len(STAGE_KEYS):
        raise ValueError(
            f"8단계를 찾으려면 유효 자세 프레임이 최소 8개 필요합니다: {len(features)}개"
        )

    wrist_x = _smooth([feature["wrist_x"] for feature in features])
    wrist_y = _smooth([feature["wrist_y"] for feature in features])
    body_height = _smooth([feature["body_height"] for feature in features])
    shoulder_angle = _smooth([feature["shoulder_angle"] for feature in features])
    shoulder_width = _smooth([feature["shoulder_width_ratio"] for feature in features])
    timestamps = np.asarray([feature["timestamp_ms"] for feature in features], dtype=float)

    dt = np.diff(timestamps, prepend=timestamps[0] - 33.0) / 1000.0
    dt = np.maximum(dt, 1e-3)
    dx = np.diff(wrist_x, prepend=wrist_x[0])
    dy = np.diff(wrist_y, prepend=wrist_y[0])
    speed = _smooth(np.hypot(dx, dy) / np.maximum(body_height * dt, 1e-6))
    max_speed = max(float(np.max(speed)), 1e-6)

    early_end = max(1, round((len(features) - 1) * 0.2))
    early_y = wrist_y[: early_end + 1]
    early_speed = speed[: early_end + 1] / max_speed
    y_span = max(float(np.ptp(early_y)), 1e-6)
    address_score = early_speed - (early_y - float(np.min(early_y))) / y_span * 0.2
    address_index = int(np.argmin(address_score))

    address_angle = float(shoulder_angle[address_index])
    address_width = max(float(shoulder_width[address_index]), 1e-6)
    shoulder_turn = np.abs(shoulder_angle - address_angle)
    shoulder_turn += np.maximum(0.0, (address_width - shoulder_width) / address_width) * 25.0
    top_start = min(len(features) - 2, address_index + 1)
    top_end = max(top_start, round((len(features) - 1) * 0.68))
    top_wrist = wrist_y[top_start : top_end + 1] / np.maximum(
        body_height[top_start : top_end + 1], 1e-6
    )
    top_turn = shoulder_turn[top_start : top_end + 1]
    turn_scale = max(float(np.max(top_turn)), 1e-6)
    top_score = top_wrist - top_turn / turn_scale * 0.12
    top_index = top_start + int(np.argmin(top_score))

    impact_start = min(len(features) - 1, top_index + 1)
    impact_end = max(impact_start, round((len(features) - 1) * 0.88))
    address_wrist = (wrist_x[address_index], wrist_y[address_index])
    impact_scores = []
    for index in range(impact_start, impact_end + 1):
        position_distance = math.hypot(
            wrist_x[index] - address_wrist[0],
            wrist_y[index] - address_wrist[1],
        ) / max(body_height[index], 1e-6)
        impact_scores.append(position_distance - speed[index] / max_speed * 0.3)
    impact_index = impact_start + int(np.argmin(impact_scores))

    finish_start = min(len(features) - 1, impact_index + 1)
    finish_scores = []
    finish_span = max(len(features) - 1 - finish_start, 1)
    for index in range(finish_start, len(features)):
        progress = (index - finish_start) / finish_span
        finish_scores.append(speed[index] / max_speed - progress * 0.3)
    finish_index = finish_start + int(np.argmin(finish_scores))

    takeaway_index = _choose_path_fraction(
        wrist_x, wrist_y, address_index, top_index, 0.32
    )
    backswing_index = _choose_path_fraction(
        wrist_x, wrist_y, address_index, top_index, 0.68
    )
    downswing_index = _choose_path_fraction(
        wrist_x, wrist_y, top_index, impact_index, 0.5
    )
    follow_index = _choose_path_fraction(
        wrist_x, wrist_y, impact_index, finish_index, 0.45
    )
    indexes = _force_strict_order(
        [
            address_index,
            takeaway_index,
            backswing_index,
            top_index,
            downswing_index,
            impact_index,
            follow_index,
            finish_index,
        ],
        len(features),
    )

    upward_ratio = float(np.mean(np.diff(wrist_y[indexes[0] : indexes[3] + 1]) < 0))
    downward_ratio = float(np.mean(np.diff(wrist_y[indexes[3] : indexes[5] + 1]) > 0))
    follow_up_ratio = float(np.mean(np.diff(wrist_y[indexes[5] : indexes[7] + 1]) < 0))
    direction_quality = (upward_ratio + downward_ratio + follow_up_ratio) / 3.0
    detection_ratio = landmark_payload.get("sampling", {}).get("detection_ratio", 1.0)
    base_confidence = max(0.0, min(1.0, detection_ratio * 0.55 + direction_quality * 0.45))

    stages = {}
    for stage_key, feature_index in zip(STAGE_KEYS, indexes):
        feature = features[feature_index]
        stages[stage_key] = {
            "frame_index": feature["frame_index"],
            "timestamp_ms": feature["timestamp_ms"],
            "confidence": round(base_confidence, 4),
            "motion": {
                "wrist_x": round(float(wrist_x[feature_index]), 6),
                "wrist_y": round(float(wrist_y[feature_index]), 6),
                "wrist_speed": round(float(speed[feature_index]), 6),
                "shoulder_angle": round(float(shoulder_angle[feature_index]), 6),
                "shoulder_turn": round(float(shoulder_turn[feature_index]), 6),
            },
        }

    return {
        "schema": STAGE_SCHEMA,
        "source_video": landmark_payload.get("source_video"),
        "method": "faceon_wrist_shoulder_motion_v1",
        "stage_order": list(STAGE_KEYS),
        "stages": stages,
        "diagnostics": {
            "usable_pose_frames": len(features),
            "max_normalized_wrist_speed": round(max_speed, 6),
            "backswing_upward_ratio": round(upward_ratio, 4),
            "downswing_downward_ratio": round(downward_ratio, 4),
            "follow_through_upward_ratio": round(follow_up_ratio, 4),
            "direction_quality": round(direction_quality, 4),
        },
    }


def save_representative_frames(video_path, stage_detection, output_dir):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

    saved = {}
    try:
        for order, stage_key in enumerate(STAGE_KEYS, start=1):
            frame_index = stage_detection["stages"][stage_key]["frame_index"]
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"대표 프레임을 읽지 못했습니다: {frame_index}")
            output_path = output_dir / f"{order:02d}_{stage_key}.jpg"
            if not cv2.imwrite(str(output_path), frame):
                raise OSError(f"대표 프레임을 저장하지 못했습니다: {output_path}")
            saved[stage_key] = str(output_path)
    finally:
        capture.release()
    return saved
