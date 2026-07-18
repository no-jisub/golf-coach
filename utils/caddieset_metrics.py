"""MediaPipe 2D 관절에서 CaddieSet 의미에 대응하는 자세 지표를 계산합니다.

CaddieSet은 지표 설명을 공개하지만 원본 계산 코드는 공개하지 않습니다. 이 모듈은
논문에 적힌 관절과 단위를 기준으로 만든 프로젝트 내부의 재현 가능한 근사식입니다.
따라서 계산 결과는 다음 단계에서 관찰 범위와 비교하되, 원본 데이터와 완전히 같은
측정 장비 또는 생체역학적 정답으로 취급하지 않습니다.
"""

import math

from utils.guide_skeleton import (
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


METRIC_LANDMARKS = {
    NOSE,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
}


def _finite_point(point):
    return (
        point is not None
        and len(point) >= 2
        and math.isfinite(float(point[0]))
        and math.isfinite(float(point[1]))
    )


def _point(points, index):
    point = points.get(index)
    if not _finite_point(point):
        return None
    return float(point[0]), float(point[1])


def midpoint(point_a, point_b):
    if not _finite_point(point_a) or not _finite_point(point_b):
        return None
    return (
        (float(point_a[0]) + float(point_b[0])) / 2.0,
        (float(point_a[1]) + float(point_b[1])) / 2.0,
    )


def distance(point_a, point_b):
    if not _finite_point(point_a) or not _finite_point(point_b):
        return None
    return math.hypot(
        float(point_a[0]) - float(point_b[0]),
        float(point_a[1]) - float(point_b[1]),
    )


def joint_angle(point_a, point_b, point_c):
    """세 점 A-B-C에서 B를 중심으로 0~180도 각도를 계산합니다."""
    if not all(_finite_point(point) for point in (point_a, point_b, point_c)):
        return None

    vector_a = (
        float(point_a[0]) - float(point_b[0]),
        float(point_a[1]) - float(point_b[1]),
    )
    vector_c = (
        float(point_c[0]) - float(point_b[0]),
        float(point_c[1]) - float(point_b[1]),
    )
    denominator = math.hypot(*vector_a) * math.hypot(*vector_c)
    if denominator <= 1e-9:
        return None
    cosine = (vector_a[0] * vector_c[0] + vector_a[1] * vector_c[1]) / denominator
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def line_angle_to_horizontal(point_a, point_b, acute=True):
    """두 점을 잇는 선과 화면 수평선 사이 각도를 계산합니다."""
    if not _finite_point(point_a) or not _finite_point(point_b):
        return None
    dx = float(point_b[0]) - float(point_a[0])
    dy = float(point_b[1]) - float(point_a[1])
    if math.hypot(dx, dy) <= 1e-9:
        return None
    angle = abs(math.degrees(math.atan2(dy, dx)))
    if acute and angle > 90.0:
        angle = 180.0 - angle
    return angle


def point_to_line_distance(point, line_start, line_end):
    if not all(_finite_point(item) for item in (point, line_start, line_end)):
        return None
    line_dx = float(line_end[0]) - float(line_start[0])
    line_dy = float(line_end[1]) - float(line_start[1])
    line_length = math.hypot(line_dx, line_dy)
    if line_length <= 1e-9:
        return None
    numerator = abs(
        line_dy * float(point[0])
        - line_dx * float(point[1])
        + float(line_end[0]) * float(line_start[1])
        - float(line_end[1]) * float(line_start[0])
    )
    return numerator / line_length


def average_landmark_points(landmark_samples, min_visibility=0.5):
    """여러 MediaPipe 프레임에서 지표 계산용 관절의 평균 2D 좌표를 만듭니다."""
    collected = {index: [] for index in METRIC_LANDMARKS}
    for landmarks in landmark_samples:
        for index in METRIC_LANDMARKS:
            if len(landmarks) <= index:
                continue
            landmark = landmarks[index]
            if getattr(landmark, "visibility", 1.0) < min_visibility:
                continue
            point = (getattr(landmark, "x", None), getattr(landmark, "y", None))
            if _finite_point(point):
                collected[index].append((float(point[0]), float(point[1])))

    return {
        index: (
            sum(point[0] for point in samples) / len(samples),
            sum(point[1] for point in samples) / len(samples),
        )
        for index, samples in collected.items()
        if samples
    }


def _safe_ratio(numerator, denominator):
    if numerator is None or denominator is None or abs(denominator) <= 1e-9:
        return None
    return numerator / denominator


def _signed_horizontal_delta(current, reference, scale, direction_multiplier):
    if not _finite_point(current) or not _finite_point(reference):
        return None
    ratio = _safe_ratio(float(current[0]) - float(reference[0]), scale)
    if ratio is None:
        return None
    return ratio * direction_multiplier


def calculate_pose_metrics(points, address_points=None, direction_multiplier=1.0):
    """한 자세의 CaddieSet 대응 2D 지표를 계산합니다.

    ``direction_multiplier``는 좌우 반전된 영상의 부호를 기준 데이터 방향과 맞출 때
    -1로 줄 수 있습니다. 어드레스 대비 이동 지표는 ``address_points``가 없으면
    ``None``을 반환합니다.
    """
    left_shoulder = _point(points, LEFT_SHOULDER)
    right_shoulder = _point(points, RIGHT_SHOULDER)
    left_elbow = _point(points, LEFT_ELBOW)
    right_elbow = _point(points, RIGHT_ELBOW)
    left_wrist = _point(points, LEFT_WRIST)
    right_wrist = _point(points, RIGHT_WRIST)
    left_hip = _point(points, LEFT_HIP)
    right_hip = _point(points, RIGHT_HIP)
    left_knee = _point(points, LEFT_KNEE)
    right_knee = _point(points, RIGHT_KNEE)
    left_ankle = _point(points, LEFT_ANKLE)
    right_ankle = _point(points, RIGHT_ANKLE)
    nose = _point(points, NOSE)

    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)
    ankle_mid = midpoint(left_ankle, right_ankle)
    shoulder_width = distance(left_shoulder, right_shoulder)
    stance_width = distance(left_ankle, right_ankle)
    torso_length = distance(shoulder_mid, hip_mid)
    lower_body_length = distance(hip_mid, ankle_mid)

    address_nose = _point(address_points or {}, NOSE)
    address_left_hip = _point(address_points or {}, LEFT_HIP)
    address_right_hip = _point(address_points or {}, RIGHT_HIP)
    address_hip_mid = midpoint(address_left_hip, address_right_hip)
    address_left_ankle = _point(address_points or {}, LEFT_ANKLE)
    address_right_ankle = _point(address_points or {}, RIGHT_ANKLE)
    address_stance_width = distance(address_left_ankle, address_right_ankle)
    movement_scale = address_stance_width or stance_width

    current_hip_angle = line_angle_to_horizontal(left_hip, right_hip, acute=False)
    address_hip_angle = line_angle_to_horizontal(address_left_hip, address_right_hip, acute=False)
    hip_rotation = None
    if current_hip_angle is not None and address_hip_angle is not None:
        delta = abs(current_hip_angle - address_hip_angle)
        hip_rotation = min(delta, 180.0 - delta)

    shoulder_location = None
    if left_shoulder is not None and left_ankle is not None and right_ankle is not None:
        left_edge = min(left_ankle[0], right_ankle[0])
        shoulder_location = _safe_ratio(left_shoulder[0] - left_edge, stance_width)
        if shoulder_location is not None:
            shoulder_location *= 100.0

    right_distance = point_to_line_distance(right_elbow, right_shoulder, right_hip)
    right_distance = _safe_ratio(right_distance, shoulder_width)

    return {
        "shoulder_angle": line_angle_to_horizontal(left_shoulder, right_shoulder),
        "spine_angle": line_angle_to_horizontal(hip_mid, shoulder_mid),
        "stance_ratio": _safe_ratio(stance_width, shoulder_width),
        "upper_tilt": _safe_ratio(torso_length, lower_body_length),
        "head_loc": _signed_horizontal_delta(
            nose,
            address_nose,
            movement_scale,
            direction_multiplier,
        ),
        "hip_line": _signed_horizontal_delta(
            hip_mid,
            address_hip_mid,
            movement_scale,
            direction_multiplier,
        ),
        "hip_rotation": hip_rotation,
        "hip_shifted": _signed_horizontal_delta(
            hip_mid,
            address_hip_mid,
            movement_scale,
            direction_multiplier,
        ),
        "left_arm_angle": joint_angle(left_shoulder, left_elbow, left_wrist),
        "right_arm_angle": joint_angle(right_shoulder, right_elbow, right_wrist),
        "shoulder_loc": shoulder_location,
        "hip_angle": current_hip_angle,
        "left_leg_angle": joint_angle(left_hip, left_knee, left_ankle),
        "right_leg_angle": joint_angle(right_hip, right_knee, right_ankle),
        "right_distance": right_distance,
        "hip_hanging_back": _safe_ratio(
            abs(left_hip[0] - left_ankle[0]) if left_hip and left_ankle else None,
            stance_width,
        ),
        "right_armpit_angle": joint_angle(right_elbow, right_shoulder, right_hip),
        "shoulder_hanging_back": _safe_ratio(
            abs(left_shoulder[0] - left_ankle[0])
            if left_shoulder and left_ankle
            else None,
            stance_width,
        ),
        "weight_shift": line_angle_to_horizontal(left_ankle, left_hip, acute=False),
        "finish_angle": line_angle_to_horizontal(left_ankle, right_hip, acute=False),
    }

