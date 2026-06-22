import cv2


# MediaPipe pose index 중 현재 앱에서 쓰는 주요 관절만 기준 스켈레톤으로 그립니다.
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

GUIDE_CONNECTIONS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
]


# 좌표는 0~1 정규화 값입니다. x는 화면 가로, y는 피드백 패널을 제외한 세로 영역 기준입니다.
# 정면 웹캠에서 따라 하기 쉬운 단순 가이드 형태로 잡았습니다.
GUIDE_POSES = {
    "address": {
        LEFT_SHOULDER: (0.43, 0.26),
        RIGHT_SHOULDER: (0.57, 0.26),
        LEFT_ELBOW: (0.45, 0.40),
        RIGHT_ELBOW: (0.55, 0.40),
        LEFT_WRIST: (0.47, 0.53),
        RIGHT_WRIST: (0.53, 0.53),
        LEFT_HIP: (0.45, 0.50),
        RIGHT_HIP: (0.55, 0.50),
        LEFT_KNEE: (0.40, 0.70),
        RIGHT_KNEE: (0.60, 0.70),
        LEFT_ANKLE: (0.35, 0.92),
        RIGHT_ANKLE: (0.65, 0.92),
    },
    "takeaway": {
        LEFT_SHOULDER: (0.43, 0.26),
        RIGHT_SHOULDER: (0.57, 0.26),
        LEFT_ELBOW: (0.38, 0.38),
        RIGHT_ELBOW: (0.50, 0.40),
        LEFT_WRIST: (0.31, 0.48),
        RIGHT_WRIST: (0.43, 0.49),
        LEFT_HIP: (0.45, 0.50),
        RIGHT_HIP: (0.55, 0.50),
        LEFT_KNEE: (0.40, 0.70),
        RIGHT_KNEE: (0.60, 0.70),
        LEFT_ANKLE: (0.35, 0.92),
        RIGHT_ANKLE: (0.65, 0.92),
    },
    "backswing": {
        LEFT_SHOULDER: (0.43, 0.26),
        RIGHT_SHOULDER: (0.57, 0.26),
        LEFT_ELBOW: (0.34, 0.24),
        RIGHT_ELBOW: (0.47, 0.30),
        LEFT_WRIST: (0.25, 0.18),
        RIGHT_WRIST: (0.39, 0.22),
        LEFT_HIP: (0.45, 0.50),
        RIGHT_HIP: (0.55, 0.50),
        LEFT_KNEE: (0.40, 0.70),
        RIGHT_KNEE: (0.60, 0.70),
        LEFT_ANKLE: (0.35, 0.92),
        RIGHT_ANKLE: (0.65, 0.92),
    },
    "top": {
        LEFT_SHOULDER: (0.43, 0.27),
        RIGHT_SHOULDER: (0.57, 0.25),
        LEFT_ELBOW: (0.37, 0.15),
        RIGHT_ELBOW: (0.49, 0.16),
        LEFT_WRIST: (0.31, 0.07),
        RIGHT_WRIST: (0.42, 0.08),
        LEFT_HIP: (0.45, 0.50),
        RIGHT_HIP: (0.55, 0.50),
        LEFT_KNEE: (0.40, 0.70),
        RIGHT_KNEE: (0.60, 0.70),
        LEFT_ANKLE: (0.35, 0.92),
        RIGHT_ANKLE: (0.65, 0.92),
    },
    "downswing": {
        LEFT_SHOULDER: (0.43, 0.26),
        RIGHT_SHOULDER: (0.57, 0.26),
        LEFT_ELBOW: (0.39, 0.34),
        RIGHT_ELBOW: (0.51, 0.34),
        LEFT_WRIST: (0.36, 0.45),
        RIGHT_WRIST: (0.47, 0.45),
        LEFT_HIP: (0.45, 0.50),
        RIGHT_HIP: (0.55, 0.50),
        LEFT_KNEE: (0.41, 0.70),
        RIGHT_KNEE: (0.61, 0.70),
        LEFT_ANKLE: (0.35, 0.92),
        RIGHT_ANKLE: (0.65, 0.92),
    },
    "impact": {
        LEFT_SHOULDER: (0.43, 0.27),
        RIGHT_SHOULDER: (0.57, 0.25),
        LEFT_ELBOW: (0.43, 0.42),
        RIGHT_ELBOW: (0.55, 0.40),
        LEFT_WRIST: (0.45, 0.54),
        RIGHT_WRIST: (0.55, 0.53),
        LEFT_HIP: (0.45, 0.50),
        RIGHT_HIP: (0.56, 0.49),
        LEFT_KNEE: (0.41, 0.70),
        RIGHT_KNEE: (0.61, 0.70),
        LEFT_ANKLE: (0.35, 0.92),
        RIGHT_ANKLE: (0.65, 0.92),
    },
    "follow_through": {
        LEFT_SHOULDER: (0.43, 0.26),
        RIGHT_SHOULDER: (0.57, 0.26),
        LEFT_ELBOW: (0.52, 0.32),
        RIGHT_ELBOW: (0.65, 0.30),
        LEFT_WRIST: (0.62, 0.35),
        RIGHT_WRIST: (0.76, 0.33),
        LEFT_HIP: (0.45, 0.50),
        RIGHT_HIP: (0.55, 0.50),
        LEFT_KNEE: (0.41, 0.70),
        RIGHT_KNEE: (0.61, 0.70),
        LEFT_ANKLE: (0.35, 0.92),
        RIGHT_ANKLE: (0.65, 0.92),
    },
    "finish": {
        LEFT_SHOULDER: (0.43, 0.25),
        RIGHT_SHOULDER: (0.57, 0.27),
        LEFT_ELBOW: (0.55, 0.17),
        RIGHT_ELBOW: (0.67, 0.19),
        LEFT_WRIST: (0.66, 0.09),
        RIGHT_WRIST: (0.78, 0.11),
        LEFT_HIP: (0.45, 0.50),
        RIGHT_HIP: (0.55, 0.50),
        LEFT_KNEE: (0.43, 0.70),
        RIGHT_KNEE: (0.59, 0.69),
        LEFT_ANKLE: (0.38, 0.92),
        RIGHT_ANKLE: (0.63, 0.92),
    },
}


def guide_point_to_pixel(point, image_width, guide_top, guide_height):
    """정규화된 가이드 좌표를 실제 화면 좌표로 변환합니다."""
    x = int(point[0] * image_width)
    y = int(guide_top + point[1] * guide_height)
    return x, y


def draw_guide_skeleton(frame, stage_key):
    """현재 단계에서 따라 할 보조 스켈레톤을 화면 중앙에 그립니다."""
    guide_pose = GUIDE_POSES.get(stage_key)
    if not guide_pose:
        return

    image_height, image_width, _ = frame.shape
    feedback_panel_height = 180
    guide_top = 20
    guide_height = max(image_height - feedback_panel_height - 40, 120)

    overlay = frame.copy()
    line_color = (255, 180, 40)
    point_color = (255, 240, 120)

    for start_idx, end_idx in GUIDE_CONNECTIONS:
        start = guide_pose[start_idx]
        end = guide_pose[end_idx]
        start_point = guide_point_to_pixel(start, image_width, guide_top, guide_height)
        end_point = guide_point_to_pixel(end, image_width, guide_top, guide_height)
        cv2.line(overlay, start_point, end_point, line_color, 5)

    for point in guide_pose.values():
        pixel = guide_point_to_pixel(point, image_width, guide_top, guide_height)
        cv2.circle(overlay, pixel, 8, point_color, -1)
        cv2.circle(overlay, pixel, 10, line_color, 2)

    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
