import math

import numpy as np

from utils.angle_calculator import calculate_angle


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

REQUIRED_LANDMARKS = [
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
]


def get_visibility(landmark):
    """Tasks API 랜드마크의 visibility 값을 안전하게 읽습니다."""
    return getattr(landmark, "visibility", 1.0)


def average_point(landmark_samples, index):
    """여러 프레임의 같은 관절 좌표를 평균냅니다."""
    points = []
    for landmarks in landmark_samples:
        if len(landmarks) <= index:
            continue

        landmark = landmarks[index]
        if get_visibility(landmark) < 0.5:
            continue

        points.append((landmark.x, landmark.y))

    if not points:
        return None

    return tuple(np.mean(points, axis=0))


def get_average_points(landmark_samples):
    """어드레스 분석에 필요한 주요 관절 평균 좌표를 만듭니다."""
    points = {}
    for index in REQUIRED_LANDMARKS:
        point = average_point(landmark_samples, index)
        if point is None:
            return None
        points[index] = point
    return points


def midpoint(point_a, point_b):
    """두 점의 중간점을 반환합니다."""
    return ((point_a[0] + point_b[0]) / 2, (point_a[1] + point_b[1]) / 2)


def calculate_torso_tilt_degrees(points):
    """골반-어깨 중심선이 수직선에서 얼마나 기울었는지 계산합니다."""
    shoulder_mid = midpoint(points[LEFT_SHOULDER], points[RIGHT_SHOULDER])
    hip_mid = midpoint(points[LEFT_HIP], points[RIGHT_HIP])

    dx = shoulder_mid[0] - hip_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    if dy == 0:
        return 90.0

    return abs(math.degrees(math.atan2(dx, dy)))


def analyze_address_pose(landmark_samples):
    """최근 프레임 평균값으로 어드레스 자세를 1차 판단합니다."""
    if not landmark_samples:
        return {
            "passed": False,
            "messages": ["자세를 인식하지 못했습니다. 카메라 앞에서 어드레스 자세를 잡아주세요."],
            "metrics": {},
        }

    points = get_average_points(landmark_samples)
    if points is None:
        return {
            "passed": False,
            "messages": ["주요 관절이 충분히 보이지 않습니다. 전신이 화면에 나오도록 서주세요."],
            "metrics": {},
        }

    left_knee_angle = calculate_angle(
        points[LEFT_HIP],
        points[LEFT_KNEE],
        points[LEFT_ANKLE],
    )
    right_knee_angle = calculate_angle(
        points[RIGHT_HIP],
        points[RIGHT_KNEE],
        points[RIGHT_ANKLE],
    )
    knee_angle = (left_knee_angle + right_knee_angle) / 2

    torso_tilt = calculate_torso_tilt_degrees(points)

    shoulder_width = abs(points[LEFT_SHOULDER][0] - points[RIGHT_SHOULDER][0])
    foot_width = abs(points[LEFT_ANKLE][0] - points[RIGHT_ANKLE][0])
    foot_to_shoulder_ratio = foot_width / shoulder_width if shoulder_width else 0

    shoulder_tilt = (
        abs(points[LEFT_SHOULDER][1] - points[RIGHT_SHOULDER][1]) / shoulder_width
        if shoulder_width
        else 0
    )

    messages = []

    if knee_angle > 170:
        messages.append("무릎이 너무 펴져 있습니다. 조금 더 굽혀주세요.")
    elif knee_angle < 125:
        messages.append("무릎이 너무 많이 굽혀져 있습니다. 힘을 빼고 조금만 펴주세요.")

    if torso_tilt < 8:
        messages.append("상체가 너무 세워져 있습니다. 골반을 기준으로 조금 더 숙여주세요.")
    elif torso_tilt > 45:
        messages.append("상체가 너무 많이 숙여져 있습니다. 허리를 조금 세워주세요.")

    if foot_to_shoulder_ratio < 0.9:
        messages.append("발 간격이 너무 좁습니다.")
    elif foot_to_shoulder_ratio > 2.2:
        messages.append("발 간격이 너무 넓습니다.")

    if shoulder_tilt > 0.12:
        messages.append("어깨 높이가 많이 기울어져 있습니다.")

    if not messages:
        messages.append("어드레스 자세가 좋습니다. 다음 단계로 넘어가도 됩니다.")

    return {
        "passed": len(messages) == 1 and messages[0].startswith("어드레스 자세가 좋습니다"),
        "messages": messages,
        "metrics": {
            "knee_angle": knee_angle,
            "torso_tilt": torso_tilt,
            "foot_to_shoulder_ratio": foot_to_shoulder_ratio,
            "shoulder_tilt": shoulder_tilt,
        },
    }
