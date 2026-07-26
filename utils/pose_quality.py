"""보정과 자세 분석 전에 입력 품질과 정지 상태를 검사합니다."""

import math

import numpy as np

from utils.guide_skeleton import (
    GUIDE_POSES,
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)


FULL_BODY_LANDMARKS = {
    NOSE: "머리",
    LEFT_SHOULDER: "왼쪽 어깨",
    RIGHT_SHOULDER: "오른쪽 어깨",
    LEFT_ELBOW: "왼쪽 팔꿈치",
    RIGHT_ELBOW: "오른쪽 팔꿈치",
    LEFT_WRIST: "왼쪽 손목",
    RIGHT_WRIST: "오른쪽 손목",
    LEFT_HIP: "왼쪽 골반",
    RIGHT_HIP: "오른쪽 골반",
    LEFT_KNEE: "왼쪽 무릎",
    RIGHT_KNEE: "오른쪽 무릎",
    LEFT_ANKLE: "왼쪽 발목",
    RIGHT_ANKLE: "오른쪽 발목",
}

STABILITY_LANDMARKS = tuple(FULL_BODY_LANDMARKS)
DEFAULT_MIN_VISIBILITY = 0.55
DEFAULT_FRAME_MARGIN = 0.015
DEFAULT_STABILITY_DURATION_SEC = 1.5
DEFAULT_MAX_MEAN_JITTER = 0.012
DEFAULT_MAX_JOINT_JITTER = 0.035
CALIBRATION_MIN_ADDRESS_SCORE = 55


def _valid_landmark(landmark, min_visibility):
    if landmark is None:
        return False
    if getattr(landmark, "visibility", 1.0) < min_visibility:
        return False
    return all(
        math.isfinite(float(getattr(landmark, axis, float("nan"))))
        for axis in ("x", "y")
    )


def check_full_body_visibility(
    landmarks,
    *,
    min_visibility=DEFAULT_MIN_VISIBILITY,
    frame_margin=DEFAULT_FRAME_MARGIN,
):
    """주요 관절이 신뢰도 있게 화면 안에 들어왔는지 확인합니다."""
    missing = []
    clipped = []
    for index, label in FULL_BODY_LANDMARKS.items():
        if landmarks is None or len(landmarks) <= index:
            missing.append(label)
            continue

        landmark = landmarks[index]
        if not _valid_landmark(landmark, min_visibility):
            missing.append(label)
            continue

        if not (
            frame_margin <= float(landmark.x) <= 1.0 - frame_margin
            and frame_margin <= float(landmark.y) <= 1.0 - frame_margin
        ):
            clipped.append(label)

    messages = []
    if missing:
        messages.append(f"{', '.join(missing[:3])} 관절이 잘 보이지 않습니다.")
    if clipped:
        messages.append(f"{', '.join(clipped[:3])} 부분이 화면 가장자리에 너무 가깝습니다.")
    if messages:
        messages.append("머리부터 양쪽 발목까지 화면 안에 나오도록 위치를 조정해주세요.")

    return {
        "passed": not missing and not clipped,
        "missing": missing,
        "clipped": clipped,
        "messages": messages,
    }


def _points_from_landmarks(landmarks):
    points = {}
    for index in FULL_BODY_LANDMARKS:
        landmark = landmarks[index]
        points[index] = (float(landmark.x), float(landmark.y))
    return points


def _midpoint(point_a, point_b):
    return (
        (point_a[0] + point_b[0]) / 2.0,
        (point_a[1] + point_b[1]) / 2.0,
    )


def _normalize_pose(points):
    shoulder_mid = _midpoint(points[LEFT_SHOULDER], points[RIGHT_SHOULDER])
    ankle_mid = _midpoint(points[LEFT_ANKLE], points[RIGHT_ANKLE])
    shoulder_width = abs(points[RIGHT_SHOULDER][0] - points[LEFT_SHOULDER][0])
    body_height = abs(ankle_mid[1] - shoulder_mid[1])
    if shoulder_width < 0.01 or body_height < 0.1:
        return None
    return {
        index: (
            (point[0] - shoulder_mid[0]) / shoulder_width,
            (point[1] - shoulder_mid[1]) / body_height,
        )
        for index, point in points.items()
    }


def check_address_similarity(landmarks, min_score=CALIBRATION_MIN_ADDRESS_SCORE):
    """보정 자세가 최소한 어드레스 형태인지 가이드와 비교합니다."""
    points = _points_from_landmarks(landmarks)
    user_pose = _normalize_pose(points)
    guide_pose = _normalize_pose(GUIDE_POSES["address"])
    if user_pose is None or guide_pose is None:
        return {
            "passed": False,
            "score": 0,
            "messages": ["몸 크기를 계산할 수 없습니다. 카메라와 거리를 조정해주세요."],
        }

    distances = {
        index: math.dist(user_pose[index], guide_pose[index])
        for index in FULL_BODY_LANDMARKS
    }
    average_distance = float(np.mean(list(distances.values())))
    arm_distance = float(
        np.mean(
            [
                distances[LEFT_ELBOW],
                distances[RIGHT_ELBOW],
                distances[LEFT_WRIST],
                distances[RIGHT_WRIST],
            ]
        )
    )
    lower_distance = float(
        np.mean(
            [
                distances[LEFT_KNEE],
                distances[RIGHT_KNEE],
                distances[LEFT_ANKLE],
                distances[RIGHT_ANKLE],
            ]
        )
    )
    score = max(0, min(100, round(100 - average_distance * 85)))
    passed = score >= min_score and arm_distance <= 0.95 and lower_distance <= 0.75

    messages = []
    if not passed:
        if arm_distance > 0.95:
            messages.append("양손과 팔을 어드레스 가이드 위치에 맞춰주세요.")
        if lower_distance > 0.75:
            messages.append("발 간격과 무릎 위치를 어드레스 가이드에 맞춰주세요.")
        if not messages:
            messages.append("보조 스켈레톤에 맞춰 어드레스 자세를 잡아주세요.")

    return {
        "passed": passed,
        "score": score,
        "average_distance": average_distance,
        "arm_distance": arm_distance,
        "lower_distance": lower_distance,
        "messages": messages,
    }


def evaluate_calibration_frame(landmarks):
    """한 프레임이 사용자 체형 보정에 사용할 수 있는지 확인합니다."""
    visibility = check_full_body_visibility(landmarks)
    if not visibility["passed"]:
        return {
            "passed": False,
            "reason": "visibility",
            "messages": visibility["messages"],
            "visibility": visibility,
        }

    address = check_address_similarity(landmarks)
    if not address["passed"]:
        return {
            "passed": False,
            "reason": "address",
            "messages": address["messages"],
            "visibility": visibility,
            "address": address,
        }

    return {
        "passed": True,
        "reason": "ready",
        "messages": ["어드레스 자세를 유지해주세요."],
        "visibility": visibility,
        "address": address,
    }


def trim_timed_samples(samples, now, window_sec):
    """deque에 저장된 시간 기반 샘플에서 오래된 프레임을 제거합니다."""
    while samples and now - samples[0][0] > window_sec:
        samples.popleft()


def evaluate_pose_stability(
    timed_samples,
    *,
    min_duration_sec=DEFAULT_STABILITY_DURATION_SEC,
    max_mean_jitter=DEFAULT_MAX_MEAN_JITTER,
    max_joint_jitter=DEFAULT_MAX_JOINT_JITTER,
    min_visibility=DEFAULT_MIN_VISIBILITY,
):
    """주요 관절의 최근 흔들림이 분석 가능한 수준인지 계산합니다."""
    if not timed_samples:
        return {
            "ready": False,
            "stable": False,
            "duration_sec": 0.0,
            "progress": 0.0,
            "message": "자세를 잡고 잠시 멈춰주세요.",
        }

    duration = max(0.0, timed_samples[-1][0] - timed_samples[0][0])
    progress = min(duration / min_duration_sec, 1.0)
    if duration < min_duration_sec or len(timed_samples) < 8:
        return {
            "ready": False,
            "stable": False,
            "duration_sec": duration,
            "progress": progress,
            "message": f"자세를 움직이지 말고 {min_duration_sec:.1f}초 유지해주세요.",
        }

    joint_jitters = {}
    for index in STABILITY_LANDMARKS:
        coordinates = []
        for _, landmarks in timed_samples:
            if len(landmarks) <= index or not _valid_landmark(
                landmarks[index],
                min_visibility,
            ):
                continue
            coordinates.append((landmarks[index].x, landmarks[index].y))

        if len(coordinates) < max(4, len(timed_samples) * 0.6):
            return {
                "ready": True,
                "stable": False,
                "duration_sec": duration,
                "progress": progress,
                "message": f"{FULL_BODY_LANDMARKS[index]} 관절을 안정적으로 인식하지 못했습니다.",
            }

        values = np.asarray(coordinates, dtype=float)
        jitter = float(np.hypot(np.std(values[:, 0]), np.std(values[:, 1])))
        joint_jitters[index] = jitter

    mean_jitter = float(np.mean(list(joint_jitters.values())))
    max_index = max(joint_jitters, key=joint_jitters.get)
    max_jitter = joint_jitters[max_index]
    stable = mean_jitter <= max_mean_jitter and max_jitter <= max_joint_jitter
    if stable:
        message = "자세가 안정되었습니다. 분석 중입니다."
    else:
        message = (
            f"{FULL_BODY_LANDMARKS[max_index]} 움직임이 큽니다. "
            "현재 자세에서 잠시 멈춰주세요."
        )

    return {
        "ready": True,
        "stable": stable,
        "duration_sec": duration,
        "progress": progress,
        "mean_jitter": mean_jitter,
        "max_joint_jitter": max_jitter,
        "max_joint_index": max_index,
        "joint_jitters": joint_jitters,
        "message": message,
    }
