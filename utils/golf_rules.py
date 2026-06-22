import math

import numpy as np

from utils.guide_skeleton import (
    GUIDE_POSES,
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
)


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

BODY_PART_GROUPS = {
    "head": [NOSE],
    "arms": [LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST],
    "body": [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP],
    "lower": [LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE],
}


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


def distance(point_a, point_b):
    """두 정규화 좌표 사이의 거리입니다."""
    return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)


def normalize_pose(points):
    """위치와 크기 영향을 줄이기 위해 어깨 중심/어깨 너비 기준으로 좌표를 정규화합니다."""
    shoulder_mid = midpoint(points[LEFT_SHOULDER], points[RIGHT_SHOULDER])
    shoulder_width = abs(points[LEFT_SHOULDER][0] - points[RIGHT_SHOULDER][0])

    if shoulder_width < 0.01:
        return None

    return {
        index: (
            (point[0] - shoulder_mid[0]) / shoulder_width,
            (point[1] - shoulder_mid[1]) / shoulder_width,
        )
        for index, point in points.items()
    }


def get_group_distance(user_pose, guide_pose, indexes):
    """관절 그룹별 평균 차이를 계산합니다."""
    distances = [distance(user_pose[index], guide_pose[index]) for index in indexes]
    return float(np.mean(distances))


def get_group_delta(user_pose, guide_pose, indexes):
    """관절 그룹이 기준보다 어느 방향으로 벗어났는지 계산합니다."""
    deltas = [
        (
            user_pose[index][0] - guide_pose[index][0],
            user_pose[index][1] - guide_pose[index][1],
        )
        for index in indexes
    ]
    return tuple(np.mean(deltas, axis=0))


def direction_text(delta):
    """사용자 자세가 가이드보다 높거나 낮은지 간단히 설명합니다."""
    dx, dy = delta
    directions = []

    if dy > 0.35:
        directions.append("조금 더 올려주세요")
    elif dy < -0.35:
        directions.append("조금 더 내려주세요")

    if abs(dx) > 0.45:
        directions.append("좌우 위치를 가이드에 더 맞춰주세요")

    if not directions:
        return "가이드 위치에 더 가깝게 맞춰주세요"

    return ", ".join(directions)


def build_guide_feedback(stage_key, points):
    """보조 스켈레톤과 사용자 관절을 비교해 점수와 피드백을 만듭니다."""
    guide_points = GUIDE_POSES.get(stage_key)
    if guide_points is None:
        return make_result(stage_key, False, ["현재 단계의 보조 스켈레톤 기준이 없습니다."])

    user_pose = normalize_pose(points)
    guide_pose = normalize_pose(guide_points)

    if user_pose is None or guide_pose is None:
        return make_result(stage_key, False, ["어깨 너비를 기준으로 자세를 정규화할 수 없습니다. 전신이 보이게 서주세요."])

    all_distances = [distance(user_pose[index], guide_pose[index]) for index in REQUIRED_LANDMARKS]
    group_distances = {
        name: get_group_distance(user_pose, guide_pose, indexes)
        for name, indexes in BODY_PART_GROUPS.items()
    }
    group_deltas = {
        name: get_group_delta(user_pose, guide_pose, indexes)
        for name, indexes in BODY_PART_GROUPS.items()
    }

    average_distance = float(np.mean(all_distances))
    max_group_distance = max(group_distances.values())
    score = max(0, min(100, int(100 - average_distance * 55 - max_group_distance * 18)))
    passed = score >= 70 and max_group_distance <= 1.05

    stage = get_stage_config(stage_key)
    messages = []

    if passed:
        messages.append(f"{stage['korean']} 자세가 보조 스켈레톤과 잘 맞습니다.")
    else:
        if group_distances["head"] > 0.55:
            messages.append(f"머리 위치가 가이드와 다릅니다. {direction_text(group_deltas['head'])}.")
        if group_distances["arms"] > 0.75:
            messages.append(f"팔과 손 위치가 가이드와 다릅니다. {direction_text(group_deltas['arms'])}.")
        if group_distances["body"] > 0.55:
            messages.append("어깨와 골반 위치를 보조 스켈레톤 중심에 더 맞춰주세요.")
        if group_distances["lower"] > 0.75:
            messages.append("무릎과 발목 위치를 가이드 하체 라인에 더 맞춰주세요.")
        if not messages:
            messages.append("전체 자세를 보조 스켈레톤에 조금 더 가깝게 맞춰주세요.")

    metrics = {
        "guide_score": score,
        "average_distance": average_distance,
        "max_group_distance": max_group_distance,
        "head_distance": group_distances["head"],
        "arms_distance": group_distances["arms"],
        "body_distance": group_distances["body"],
        "lower_distance": group_distances["lower"],
    }
    return make_result(stage_key, passed, messages, metrics)


def analyze_stage_pose(stage_key, landmark_samples):
    """현재 단계의 보조 스켈레톤을 실제 판정 기준으로 사용합니다."""
    if not landmark_samples:
        return make_result(
            stage_key,
            False,
            ["자세를 인식하지 못했습니다. 카메라 앞에서 전신이 보이도록 서주세요."],
        )

    points = get_average_points(landmark_samples)
    if points is None:
        return make_result(
            stage_key,
            False,
            ["주요 관절이 충분히 보이지 않습니다. 머리부터 발목까지 화면에 나오게 해주세요."],
        )

    return build_guide_feedback(stage_key, points)
