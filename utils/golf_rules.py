import math

import numpy as np

from utils.angle_calculator import calculate_angle


NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

STAGE_CONFIGS = [
    {
        "key": "address",
        "label": "Address",
        "korean": "어드레스",
        "description": "스윙을 시작하기 위한 준비 자세",
    },
    {
        "key": "takeaway",
        "label": "Takeaway",
        "korean": "테이크백",
        "description": "어드레스 자세에서 클럽을 뒤로 빼기 시작하는 구간",
    },
    {
        "key": "backswing",
        "label": "Backswing",
        "korean": "백스윙",
        "description": "테이크백 이후 클럽을 위로 들어 올리는 과정",
    },
    {
        "key": "top",
        "label": "Top of Swing",
        "korean": "백스윙 탑",
        "description": "백스윙의 정점에 도달한 상태",
    },
    {
        "key": "downswing",
        "label": "Downswing",
        "korean": "다운스윙",
        "description": "탑에서 임팩트 존으로 클럽을 끌어내리는 전환 동작",
    },
    {
        "key": "impact",
        "label": "Impact",
        "korean": "임팩트",
        "description": "클럽 헤드가 골프공과 정확하게 만나는 순간",
    },
    {
        "key": "follow_through",
        "label": "Follow-through",
        "korean": "팔로우스루",
        "description": "임팩트 직후 클럽이 타겟 방향으로 자연스럽게 뻗어나가는 과정",
    },
    {
        "key": "finish",
        "label": "Finish",
        "korean": "피니쉬",
        "description": "스윙의 최종 마무리 단계",
    },
]

REQUIRED_LANDMARKS = [
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
]


def get_stage_config(stage_key):
    """단계 key로 단계 설정을 찾습니다."""
    for stage in STAGE_CONFIGS:
        if stage["key"] == stage_key:
            return stage
    return STAGE_CONFIGS[0]


def make_result(stage_key, passed, messages, metrics=None):
    """화면 출력에 쓰기 쉬운 공통 결과 형식입니다."""
    stage = get_stage_config(stage_key)
    return {
        "stage_key": stage["key"],
        "stage_label": stage["label"],
        "stage_korean": stage["korean"],
        "passed": passed,
        "messages": messages,
        "metrics": metrics or {},
    }


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
    """분석에 필요한 주요 관절 평균 좌표를 만듭니다."""
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


def vertical_distance(point_a, point_b):
    """화면 y 좌표 기준 세로 거리입니다. 양수면 point_a가 더 아래에 있습니다."""
    return point_a[1] - point_b[1]


def horizontal_distance(point_a, point_b):
    """화면 x 좌표 기준 가로 거리입니다."""
    return abs(point_a[0] - point_b[0])


def calculate_torso_tilt_degrees(points):
    """골반-어깨 중심선이 수직선에서 얼마나 기울었는지 계산합니다."""
    shoulder_mid = midpoint(points[LEFT_SHOULDER], points[RIGHT_SHOULDER])
    hip_mid = midpoint(points[LEFT_HIP], points[RIGHT_HIP])

    dx = shoulder_mid[0] - hip_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    if dy == 0:
        return 90.0

    return abs(math.degrees(math.atan2(dx, dy)))


def get_common_metrics(points):
    """여러 단계에서 같이 쓰는 기본 수치를 계산합니다."""
    shoulder_mid = midpoint(points[LEFT_SHOULDER], points[RIGHT_SHOULDER])
    hip_mid = midpoint(points[LEFT_HIP], points[RIGHT_HIP])
    wrist_mid = midpoint(points[LEFT_WRIST], points[RIGHT_WRIST])
    ankle_mid = midpoint(points[LEFT_ANKLE], points[RIGHT_ANKLE])

    left_knee_angle = calculate_angle(points[LEFT_HIP], points[LEFT_KNEE], points[LEFT_ANKLE])
    right_knee_angle = calculate_angle(points[RIGHT_HIP], points[RIGHT_KNEE], points[RIGHT_ANKLE])
    left_arm_angle = calculate_angle(points[LEFT_SHOULDER], points[LEFT_ELBOW], points[LEFT_WRIST])
    right_arm_angle = calculate_angle(points[RIGHT_SHOULDER], points[RIGHT_ELBOW], points[RIGHT_WRIST])

    shoulder_width = horizontal_distance(points[LEFT_SHOULDER], points[RIGHT_SHOULDER])
    foot_width = horizontal_distance(points[LEFT_ANKLE], points[RIGHT_ANKLE])

    return {
        "shoulder_mid": shoulder_mid,
        "hip_mid": hip_mid,
        "wrist_mid": wrist_mid,
        "ankle_mid": ankle_mid,
        "knee_angle": (left_knee_angle + right_knee_angle) / 2,
        "left_arm_angle": left_arm_angle,
        "right_arm_angle": right_arm_angle,
        "torso_tilt": calculate_torso_tilt_degrees(points),
        "shoulder_width": shoulder_width,
        "foot_to_shoulder_ratio": foot_width / shoulder_width if shoulder_width else 0,
        "shoulder_tilt": (
            abs(points[LEFT_SHOULDER][1] - points[RIGHT_SHOULDER][1]) / shoulder_width
            if shoulder_width
            else 0
        ),
        "wrist_height_from_hip": vertical_distance(hip_mid, wrist_mid),
        "wrist_height_from_shoulder": vertical_distance(shoulder_mid, wrist_mid),
        "head_offset_from_body": horizontal_distance(points[NOSE], hip_mid) / shoulder_width
        if shoulder_width
        else 0,
    }


def prepare_points(stage_key, landmark_samples):
    """랜드마크 평균 좌표가 부족할 때 공통 실패 결과를 만듭니다."""
    if not landmark_samples:
        return None, make_result(
            stage_key,
            False,
            ["자세를 인식하지 못했습니다. 카메라 앞에서 전신이 보이도록 서주세요."],
        )

    points = get_average_points(landmark_samples)
    if points is None:
        return None, make_result(
            stage_key,
            False,
            ["주요 관절이 충분히 보이지 않습니다. 어깨부터 발목까지 화면에 나오게 해주세요."],
        )

    return points, None


def is_pass_message(messages, prefix):
    return len(messages) == 1 and messages[0].startswith(prefix)


def analyze_address_pose(landmark_samples):
    """어드레스 자세를 1차 판단합니다."""
    stage_key = "address"
    points, error = prepare_points(stage_key, landmark_samples)
    if error:
        return error

    metrics = get_common_metrics(points)
    messages = []

    if metrics["knee_angle"] > 178:
        messages.append("무릎이 너무 펴져 있습니다. 조금 더 굽혀주세요.")
    elif metrics["knee_angle"] < 115:
        messages.append("무릎이 너무 많이 굽혀져 있습니다. 조금만 펴주세요.")

    if metrics["torso_tilt"] < 4:
        messages.append("상체가 너무 세워져 있습니다. 골반을 기준으로 조금 더 숙여주세요.")
    elif metrics["torso_tilt"] > 55:
        messages.append("상체가 너무 많이 숙여져 있습니다. 허리를 조금 세워주세요.")

    if metrics["foot_to_shoulder_ratio"] < 0.7:
        messages.append("발 간격이 너무 좁습니다.")
    elif metrics["foot_to_shoulder_ratio"] > 2.5:
        messages.append("발 간격이 너무 넓습니다.")

    if metrics["shoulder_tilt"] > 0.18:
        messages.append("어깨 높이가 많이 기울어져 있습니다.")

    if not messages:
        messages.append("어드레스 자세가 좋습니다. 다음 단계로 넘어가도 됩니다.")

    return make_result(stage_key, is_pass_message(messages, "어드레스 자세가 좋습니다"), messages, metrics)


def analyze_takeaway_pose(landmark_samples):
    """테이크백 자세를 1차 판단합니다."""
    stage_key = "takeaway"
    points, error = prepare_points(stage_key, landmark_samples)
    if error:
        return error

    metrics = get_common_metrics(points)
    messages = []

    if metrics["wrist_height_from_hip"] < -0.08:
        messages.append("손이 너무 낮습니다. 손과 클럽을 골반 높이 근처까지 천천히 빼주세요.")
    elif metrics["wrist_height_from_shoulder"] > 0.05:
        messages.append("손이 너무 높게 올라갔습니다. 아직은 테이크백 단계입니다.")

    if min(metrics["left_arm_angle"], metrics["right_arm_angle"]) < 95:
        messages.append("팔이 너무 빨리 접히고 있습니다. 팔 간격을 유지해보세요.")

    if metrics["head_offset_from_body"] > 0.9:
        messages.append("머리가 몸 중심에서 많이 벗어났습니다. 머리 위치를 조금 더 고정해주세요.")

    if not messages:
        messages.append("테이크백 자세가 좋습니다. 손과 몸통이 함께 움직이고 있습니다.")

    return make_result(stage_key, is_pass_message(messages, "테이크백 자세가 좋습니다"), messages, metrics)


def analyze_backswing_pose(landmark_samples):
    """백스윙 진행 자세를 1차 판단합니다."""
    stage_key = "backswing"
    points, error = prepare_points(stage_key, landmark_samples)
    if error:
        return error

    metrics = get_common_metrics(points)
    messages = []

    if metrics["wrist_height_from_hip"] < 0.0:
        messages.append("손이 아직 충분히 올라오지 않았습니다. 클럽을 위로 들어 올려주세요.")
    elif metrics["wrist_height_from_shoulder"] > 0.12:
        messages.append("손이 아직 어깨 높이보다 낮습니다. 백스윙 과정에서는 손이 더 올라가야 합니다.")

    if min(metrics["left_arm_angle"], metrics["right_arm_angle"]) < 85:
        messages.append("팔이 많이 접혀 있습니다. 가능한 범위에서 팔 간격을 유지해주세요.")

    if metrics["head_offset_from_body"] > 1.0:
        messages.append("머리가 몸 중심에서 많이 이동했습니다. 축을 유지해보세요.")

    if not messages:
        messages.append("백스윙 진행 자세가 좋습니다. 손이 위로 올라가고 중심도 안정적입니다.")

    return make_result(stage_key, is_pass_message(messages, "백스윙 진행 자세가 좋습니다"), messages, metrics)


def analyze_top_pose(landmark_samples):
    """백스윙 탑 자세를 1차 판단합니다."""
    stage_key = "top"
    points, error = prepare_points(stage_key, landmark_samples)
    if error:
        return error

    metrics = get_common_metrics(points)
    messages = []

    if metrics["wrist_height_from_shoulder"] > 0.02:
        messages.append("손이 충분히 올라가지 않았습니다. 탑에서는 손이 어깨보다 높아야 합니다.")

    if max(metrics["left_arm_angle"], metrics["right_arm_angle"]) < 130:
        messages.append("팔이 많이 접혀 있습니다. 가능한 범위에서 앞팔을 조금 더 펴주세요.")

    if metrics["knee_angle"] < 105:
        messages.append("무릎이 많이 무너져 있습니다. 하체 높이를 안정적으로 유지해주세요.")

    if metrics["head_offset_from_body"] > 1.0:
        messages.append("머리가 몸 중심에서 많이 이동했습니다. 축을 유지해보세요.")

    if not messages:
        messages.append("백스윙 탑 자세가 좋습니다. 팔과 하체 균형이 안정적입니다.")

    return make_result(stage_key, is_pass_message(messages, "백스윙 탑 자세가 좋습니다"), messages, metrics)


def analyze_downswing_pose(landmark_samples):
    """다운스윙 자세를 1차 판단합니다."""
    stage_key = "downswing"
    points, error = prepare_points(stage_key, landmark_samples)
    if error:
        return error

    metrics = get_common_metrics(points)
    messages = []

    if metrics["wrist_height_from_shoulder"] < -0.08:
        messages.append("손이 아직 너무 높습니다. 임팩트 존을 향해 손을 내려주세요.")
    elif metrics["wrist_height_from_hip"] < -0.18:
        messages.append("손이 너무 낮게 떨어졌습니다. 몸 앞쪽으로 자연스럽게 내려오게 해주세요.")

    if metrics["head_offset_from_body"] > 1.0:
        messages.append("다운스윙 중 머리가 많이 움직였습니다. 중심축을 유지해주세요.")

    if metrics["knee_angle"] > 182:
        messages.append("하체가 너무 일찍 펴졌습니다. 무릎의 탄성을 조금 유지해주세요.")

    if not messages:
        messages.append("다운스윙 자세가 좋습니다. 손이 임팩트 존으로 자연스럽게 내려오고 있습니다.")

    return make_result(stage_key, is_pass_message(messages, "다운스윙 자세가 좋습니다"), messages, metrics)


def analyze_impact_pose(landmark_samples):
    """임팩트 자세를 1차 판단합니다."""
    stage_key = "impact"
    points, error = prepare_points(stage_key, landmark_samples)
    if error:
        return error

    metrics = get_common_metrics(points)
    messages = []

    if metrics["wrist_height_from_hip"] < -0.12:
        messages.append("손 위치가 너무 낮습니다. 임팩트에서는 손이 골반 근처를 지나가야 합니다.")
    elif metrics["wrist_height_from_shoulder"] > 0.0:
        messages.append("손이 너무 높습니다. 임팩트 자세에서는 손을 몸 앞쪽으로 내려주세요.")

    if metrics["knee_angle"] > 182:
        messages.append("하체가 너무 펴져 있습니다. 무릎의 탄성을 조금 유지해주세요.")

    if metrics["head_offset_from_body"] > 0.9:
        messages.append("머리가 많이 움직였습니다. 공을 보는 느낌으로 중심을 유지해주세요.")

    if not messages:
        messages.append("임팩트 자세가 좋습니다. 손 위치와 중심이 안정적입니다.")

    return make_result(stage_key, is_pass_message(messages, "임팩트 자세가 좋습니다"), messages, metrics)


def analyze_follow_through_pose(landmark_samples):
    """팔로우스루 자세를 1차 판단합니다."""
    stage_key = "follow_through"
    points, error = prepare_points(stage_key, landmark_samples)
    if error:
        return error

    metrics = get_common_metrics(points)
    messages = []

    if metrics["wrist_height_from_shoulder"] > 0.10:
        messages.append("손이 충분히 뻗어나가지 않았습니다. 임팩트 후 클럽을 타겟 방향으로 보내주세요.")

    if min(metrics["left_arm_angle"], metrics["right_arm_angle"]) < 80:
        messages.append("팔이 너무 빨리 접혔습니다. 임팩트 후 팔을 자연스럽게 뻗어주세요.")

    if metrics["head_offset_from_body"] > 1.2:
        messages.append("몸 중심이 많이 흔들렸습니다. 균형을 유지하며 팔로우스루 해주세요.")

    if not messages:
        messages.append("팔로우스루 자세가 좋습니다. 클럽이 자연스럽게 타겟 방향으로 뻗고 있습니다.")

    return make_result(stage_key, is_pass_message(messages, "팔로우스루 자세가 좋습니다"), messages, metrics)


def analyze_finish_pose(landmark_samples):
    """피니쉬 자세를 1차 판단합니다."""
    stage_key = "finish"
    points, error = prepare_points(stage_key, landmark_samples)
    if error:
        return error

    metrics = get_common_metrics(points)
    messages = []

    if metrics["wrist_height_from_shoulder"] > 0.08:
        messages.append("마무리에서 손이 너무 낮습니다. 스윙 후 손을 어깨 위쪽으로 마무리해보세요.")

    if metrics["head_offset_from_body"] > 1.1:
        messages.append("피니쉬에서 몸 중심이 많이 흔들렸습니다. 균형을 잡고 멈춰주세요.")

    if metrics["shoulder_tilt"] > 0.25:
        messages.append("어깨가 많이 기울어져 있습니다. 피니쉬 자세에서 상체 균형을 잡아주세요.")

    if not messages:
        messages.append("피니쉬 자세가 좋습니다. 균형 있게 마무리되었습니다.")

    return make_result(stage_key, is_pass_message(messages, "피니쉬 자세가 좋습니다"), messages, metrics)


def analyze_stage_pose(stage_key, landmark_samples):
    """현재 선택된 단계에 맞는 자세 분석 함수를 실행합니다."""
    analyzers = {
        "address": analyze_address_pose,
        "takeaway": analyze_takeaway_pose,
        "backswing": analyze_backswing_pose,
        "top": analyze_top_pose,
        "downswing": analyze_downswing_pose,
        "impact": analyze_impact_pose,
        "follow_through": analyze_follow_through_pose,
        "finish": analyze_finish_pose,
    }
    return analyzers.get(stage_key, analyze_address_pose)(landmark_samples)
