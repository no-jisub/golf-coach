import math

import numpy as np

from utils.app_config import MVP_CLUB_TYPE, MVP_VIEW
from utils.caddieset_evaluator import (
    CaddieSetProfileError,
    classify_stage_comparisons,
    compare_stage_metrics,
    select_stage_evaluation_items,
)
from utils.caddieset_metrics import average_landmark_points, calculate_pose_metrics
from utils.guide_skeleton import (
    GUIDE_POSES,
    SWING_HAND,
    get_calibrated_guide_pixels,
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

CADDIESET_VIEW = MVP_VIEW
CADDIESET_CLUB_TYPE = MVP_CLUB_TYPE
CADDIESET_DIRECTION_MULTIPLIER = -1.0 if SWING_HAND == "right" else 1.0

METRIC_CORRECTION_TEMPLATES = {
    "shoulder_angle": ("어깨선 기울기를 조금 더 만들어주세요.", "어깨선 기울기를 조금 줄여주세요."),
    "spine_angle": ("상체 기울기를 조금 더 만들어주세요.", "상체 기울기를 조금 줄여주세요."),
    "stance_ratio": ("양발 간격을 조금 넓혀주세요.", "양발 간격을 조금 좁혀주세요."),
    "upper_tilt": ("상·하체 비율이 기준보다 작습니다. 상체 높이와 무릎 굽힘을 확인하세요.", "상·하체 비율이 기준보다 큽니다. 상체 높이와 무릎 굽힘을 확인하세요."),
    "hip_rotation": ("골반 회전을 조금 더 만들어주세요.", "골반 회전을 조금 줄여주세요."),
    "left_arm_angle": ("왼팔 팔꿈치를 조금 더 펴주세요.", "왼팔 팔꿈치를 조금 더 굽혀주세요."),
    "right_arm_angle": ("오른팔 팔꿈치를 조금 더 펴주세요.", "오른팔 팔꿈치를 조금 더 굽혀주세요."),
    "left_leg_angle": ("왼쪽 무릎을 조금 더 펴주세요.", "왼쪽 무릎을 조금 더 굽혀주세요."),
    "right_leg_angle": ("오른쪽 무릎을 조금 더 펴주세요.", "오른쪽 무릎을 조금 더 굽혀주세요."),
    "right_armpit_angle": ("오른팔과 몸통 사이 공간을 조금 넓혀주세요.", "오른팔을 몸통에 조금 더 붙여주세요."),
    "shoulder_loc": ("왼쪽 어깨와 왼발의 좌우 간격을 조금 늘려주세요.", "왼쪽 어깨와 왼발의 좌우 간격을 조금 줄여주세요."),
    "hip_hanging_back": ("왼쪽 골반을 왼발에서 조금 더 멀리 두세요.", "왼쪽 골반을 왼발 위에 조금 더 가깝게 두세요."),
    "shoulder_hanging_back": ("왼쪽 어깨를 왼발에서 조금 더 멀리 두세요.", "왼쪽 어깨를 왼발 위에 조금 더 가깝게 두세요."),
    "weight_shift": ("왼발 쪽 체중 이동선을 조금 더 세워주세요.", "왼발 쪽 체중 이동선 기울기를 조금 줄여주세요."),
    "finish_angle": ("피니시 때 왼발과 골반 정렬을 조금 더 세워주세요.", "피니시 때 왼발과 골반 기울기를 조금 줄여주세요."),
}

POSITION_METRIC_LABELS = {
    "head_loc": "머리",
    "hip_line": "골반",
    "hip_shifted": "골반",
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
    """가로는 어깨 너비, 세로는 어깨-발목 높이 기준으로 좌표를 정규화합니다."""
    shoulder_mid = midpoint(points[LEFT_SHOULDER], points[RIGHT_SHOULDER])
    shoulder_width = abs(points[LEFT_SHOULDER][0] - points[RIGHT_SHOULDER][0])
    ankle_mid = midpoint(points[LEFT_ANKLE], points[RIGHT_ANKLE])
    body_height = abs(ankle_mid[1] - shoulder_mid[1])

    if shoulder_width < 0.01 or body_height < 0.01:
        return None

    return {
        index: (
            (point[0] - shoulder_mid[0]) / shoulder_width,
            (point[1] - shoulder_mid[1]) / body_height,
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


def format_caddieset_value(value, unit):
    if value is None:
        return "측정 불가"
    if unit == "degree":
        return f"{value:.1f}°"
    return f"{value:.2f}"


def build_metric_feedback_message(comparison, direction_multiplier=CADDIESET_DIRECTION_MULTIPLIER):
    """범위를 벗어난 한 항목을 화면용 교정 문구로 바꿉니다."""
    if comparison.get("status") == "unavailable":
        return f"{comparison['description']}을 측정할 관절이 충분히 보이지 않습니다."

    relation = comparison["relation"]
    is_low = relation.startswith("below")
    metric_key = comparison["metric_key"]
    if metric_key in POSITION_METRIC_LABELS:
        desired_delta = comparison["target"] - comparison["measured_value"]
        screen_delta = desired_delta / direction_multiplier
        screen_direction = "화면 오른쪽" if screen_delta > 0 else "화면 왼쪽"
        instruction = f"{POSITION_METRIC_LABELS[metric_key]} 위치를 {screen_direction}으로 조금 옮겨주세요."
    else:
        low_message, high_message = METRIC_CORRECTION_TEMPLATES.get(
            metric_key,
            (
                f"{comparison['description']} 값을 조금 높여주세요.",
                f"{comparison['description']} 값을 조금 낮춰주세요.",
            ),
        )
        instruction = low_message if is_low else high_message

    measured = format_caddieset_value(comparison["measured_value"], comparison["unit"])
    reference_low, reference_high = comparison["reference_range"]
    reference = (
        f"{format_caddieset_value(reference_low, comparison['unit'])}~"
        f"{format_caddieset_value(reference_high, comparison['unit'])}"
    )
    return f"{instruction} 현재 {measured}, 참조 {reference}"


def build_caddieset_messages(stage_key, classified_result, max_messages=3):
    stage = get_stage_config(stage_key)
    status = classified_result["overall_status"]
    comparisons = list(classified_result["comparisons"].values())

    if status == "pass":
        count = classified_result["summary"]["pass_count"]
        return [f"{stage['korean']}의 CaddieSet 평가 항목 {count}개가 참조 범위 안입니다."]

    if status == "unavailable":
        messages = ["평가에 필요한 관절이 충분히 보이지 않습니다. 전신과 양팔이 보이게 서주세요."]
        unavailable = [item for item in comparisons if item["status"] == "unavailable"]
        messages.extend(
            build_metric_feedback_message(item)
            for item in unavailable[: max(0, max_messages - 1)]
        )
        return messages[:max_messages]

    warnings = [item for item in comparisons if item["status"] == "warning"]
    warnings.sort(
        key=lambda item: (
            item.get("warning_level") == "outside_observed",
            abs(item.get("normalized_delta") or 0.0),
        ),
        reverse=True,
    )
    messages = [build_metric_feedback_message(item) for item in warnings[:max_messages]]
    if not messages:
        messages.append("일부 관절을 측정하지 못했습니다. 전신과 양팔이 보이게 서주세요.")
    return messages


def build_caddieset_feedback(stage_key, landmark_samples, calibration_profile):
    """현재 자세를 CaddieSet 단계별 관찰 범위로 판정합니다."""
    points = average_landmark_points(landmark_samples)
    address_points = (calibration_profile or {}).get("caddieset_address_points")
    measured_metrics = calculate_pose_metrics(
        points,
        address_points=address_points,
        direction_multiplier=CADDIESET_DIRECTION_MULTIPLIER,
    )
    selection = select_stage_evaluation_items(
        stage_key,
        view=CADDIESET_VIEW,
        club_type=CADDIESET_CLUB_TYPE,
    )
    compared = compare_stage_metrics(measured_metrics, selection)
    classified = classify_stage_comparisons(compared)
    messages = build_caddieset_messages(stage_key, classified)

    result = make_result(
        stage_key,
        classified["passed"],
        messages,
        {
            "profile_id": classified["profile_id"],
            **classified["summary"],
        },
    )
    result.update(
        {
            "status": classified["overall_status"],
            "source": "caddieset",
            "item_results": classified["comparisons"],
        }
    )
    return result


def calculate_caddieset_score(classified_result):
    """관찰 범위 관계를 0~100 점수와 큰 이상치 여부로 변환합니다."""
    relation_scores = {
        "within_reference": 100,
        "below_reference": 75,
        "above_reference": 75,
        "below_outer": 25,
        "above_outer": 25,
        "unavailable": 0,
    }
    comparisons = list(classified_result.get("item_results", {}).values())
    if not comparisons:
        return 0, False

    scores = [
        relation_scores.get(comparison.get("relation"), 0)
        for comparison in comparisons
    ]
    has_outer_warning = any(
        comparison.get("relation") in {"below_outer", "above_outer"}
        for comparison in comparisons
    )
    return int(round(float(np.mean(scores)))), has_outer_warning


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


def get_average_pixel_points(landmark_samples, image_width, image_height):
    """여러 프레임의 관절 좌표를 화면 픽셀 기준으로 평균냅니다."""
    normalized_points = get_average_points(landmark_samples)
    if normalized_points is None:
        return None

    return {
        index: (point[0] * image_width, point[1] * image_height)
        for index, point in normalized_points.items()
    }


def normalize_screen_points(points, calibration_profile):
    """고정된 캘리브레이션 위치와 크기를 기준으로 화면 좌표를 정규화합니다."""
    shoulder_mid = calibration_profile["shoulder_mid"]
    shoulder_width = calibration_profile["shoulder_width"]
    body_height = shoulder_width * calibration_profile["body_ratio"]

    if shoulder_width < 1 or body_height < 1:
        return None

    return {
        index: (
            (point[0] - shoulder_mid[0]) / shoulder_width,
            (point[1] - shoulder_mid[1]) / body_height,
        )
        for index, point in points.items()
    }


def build_calibrated_guide_feedback(stage_key, landmark_samples, calibration_profile, image_width, image_height):
    """고정된 보조 스켈레톤 화면 위치를 기준으로 사용자 자세를 판정합니다."""
    user_points = get_average_pixel_points(landmark_samples, image_width, image_height)
    guide_points = get_calibrated_guide_pixels(stage_key, calibration_profile)
    if user_points is None or guide_points is None:
        return make_result(
            stage_key,
            False,
            ["주요 관절이 충분히 보이지 않습니다. 고정된 보조 스켈레톤 위치에 전신을 맞춰주세요."],
        )

    user_pose = normalize_screen_points(user_points, calibration_profile)
    guide_pose = normalize_screen_points(guide_points, calibration_profile)
    if user_pose is None or guide_pose is None:
        return make_result(stage_key, False, ["캘리브레이션 기준으로 자세를 비교할 수 없습니다. c 키로 다시 보정해주세요."])

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
    score = max(0, min(100, int(100 - average_distance * 75 - max_group_distance * 20)))
    passed = score >= 70 and max_group_distance <= 0.95

    stage = get_stage_config(stage_key)
    messages = []
    if passed:
        messages.append(f"{stage['korean']} 자세가 고정된 보조 스켈레톤과 잘 맞습니다.")
    else:
        if group_distances["head"] > 0.22:
            messages.append(f"머리 위치를 고정된 가이드에 맞춰주세요. {direction_text(group_deltas['head'])}.")
        if group_distances["arms"] > 0.32:
            messages.append(f"팔과 손 위치를 고정된 가이드에 맞춰주세요. {direction_text(group_deltas['arms'])}.")
        if group_distances["body"] > 0.24:
            messages.append("어깨와 골반을 고정된 스켈레톤 중심에 맞춰주세요.")
        if group_distances["lower"] > 0.32:
            messages.append("무릎과 발목을 고정된 하체 라인에 맞춰주세요.")
        if not messages:
            messages.append("전체 자세를 고정된 보조 스켈레톤에 더 가깝게 맞춰주세요.")

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


def build_combined_feedback(
    stage_key,
    landmark_samples,
    calibration_profile,
    image_width,
    image_height,
):
    """화면 가이드와 CaddieSet 지표를 하나의 최종 판정으로 합칩니다."""
    guide_result = build_calibrated_guide_feedback(
        stage_key,
        landmark_samples,
        calibration_profile,
        image_width,
        image_height,
    )
    caddieset_result = build_caddieset_feedback(
        stage_key,
        landmark_samples,
        calibration_profile,
    )

    guide_metrics = guide_result.get("metrics", {})
    guide_score = int(guide_metrics.get("guide_score", 0))
    caddieset_score, has_outer_warning = calculate_caddieset_score(caddieset_result)
    final_score = int(round(guide_score * 0.55 + caddieset_score * 0.45))
    caddieset_unavailable = caddieset_result.get("status") == "unavailable"
    guide_unavailable = "guide_score" not in guide_metrics

    passed = (
        not caddieset_unavailable
        and not guide_unavailable
        and not has_outer_warning
        and guide_result["passed"]
        and caddieset_score >= 70
        and final_score >= 70
    )

    if caddieset_unavailable or guide_unavailable:
        status = "unavailable"
        messages = (
            caddieset_result["messages"]
            if caddieset_unavailable
            else guide_result["messages"]
        )
    elif passed:
        status = "pass"
        stage = get_stage_config(stage_key)
        messages = [
            f"{stage['korean']} 자세가 가이드와 7번 아이언 참조 기준을 모두 충족했습니다."
        ]
    else:
        status = "warning"
        messages = []
        if not guide_result["passed"]:
            messages.extend(guide_result["messages"])
        if caddieset_result.get("status") != "pass":
            messages.extend(caddieset_result["messages"])
        messages = list(dict.fromkeys(messages))[:3]
        if not messages:
            messages = ["전체 자세를 가이드 허용 범위에 조금 더 가깝게 맞춰주세요."]

    caddieset_metrics = caddieset_result.get("metrics", {})
    result = make_result(
        stage_key,
        passed,
        messages,
        {
            "final_score": final_score,
            "guide_score": guide_score,
            "caddieset_score": caddieset_score,
            "has_outer_warning": has_outer_warning,
            "profile_id": caddieset_metrics.get("profile_id"),
            "pass_count": caddieset_metrics.get("pass_count", 0),
            "measured_count": caddieset_metrics.get("measured_count", 0),
            **guide_metrics,
        },
    )
    result.update(
        {
            "status": status,
            "source": "combined",
            "guide_result": guide_result,
            "caddieset_result": caddieset_result,
            "item_results": caddieset_result.get("item_results", {}),
        }
    )
    return result


def analyze_stage_pose(stage_key, landmark_samples, calibration_profile=None, image_width=None, image_height=None):
    """보정 후에는 화면 가이드와 CaddieSet 지표를 통합해 판정합니다."""
    if not landmark_samples:
        return make_result(
            stage_key,
            False,
            ["자세를 인식하지 못했습니다. 카메라 앞에서 전신이 보이도록 서주세요."],
        )

    if calibration_profile is not None and image_width is not None and image_height is not None:
        try:
            return build_combined_feedback(
                stage_key,
                landmark_samples,
                calibration_profile,
                image_width,
                image_height,
            )
        except CaddieSetProfileError as error:
            result = make_result(
                stage_key,
                False,
                [f"CaddieSet 평가 기준을 불러오지 못했습니다: {error}"],
            )
            result["status"] = "unavailable"
            result["source"] = "combined"
            return result

    points = get_average_points(landmark_samples)
    if points is None:
        return make_result(
            stage_key,
            False,
            ["주요 관절이 충분히 보이지 않습니다. 머리부터 발목까지 화면에 나오게 해주세요."],
        )

    return build_guide_feedback(stage_key, points)
