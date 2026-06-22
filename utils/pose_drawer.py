import cv2


# 골프 자세 확인에 우선 필요한 주요 몸통/팔/다리 연결선입니다.
POSE_CONNECTIONS = [
    (11, 12),  # 어깨
    (11, 13),  # 왼쪽 위팔
    (13, 15),  # 왼쪽 아래팔
    (12, 14),  # 오른쪽 위팔
    (14, 16),  # 오른쪽 아래팔
    (11, 23),  # 왼쪽 몸통
    (12, 24),  # 오른쪽 몸통
    (23, 24),  # 골반
    (23, 25),  # 왼쪽 허벅지
    (25, 27),  # 왼쪽 종아리
    (24, 26),  # 오른쪽 허벅지
    (26, 28),  # 오른쪽 종아리
]


def is_visible(landmark, min_visibility=0.5):
    """visibility 값이 있는 랜드마크만 화면에 그립니다."""
    visibility = getattr(landmark, "visibility", 1.0)
    return visibility >= min_visibility


def landmark_to_point(landmark, image_width, image_height):
    """정규화된 랜드마크 좌표를 OpenCV 픽셀 좌표로 변환합니다."""
    x = int(landmark.x * image_width)
    y = int(landmark.y * image_height)
    return x, y


def draw_pose_landmarks(frame, landmarks):
    """Pose Landmarker 결과를 OpenCV 화면에 점과 선으로 표시합니다."""
    image_height, image_width, _ = frame.shape

    for start_idx, end_idx in POSE_CONNECTIONS:
        start = landmarks[start_idx]
        end = landmarks[end_idx]

        if not is_visible(start) or not is_visible(end):
            continue

        start_point = landmark_to_point(start, image_width, image_height)
        end_point = landmark_to_point(end, image_width, image_height)
        cv2.line(frame, start_point, end_point, (255, 255, 255), 2)

    for landmark in landmarks:
        if not is_visible(landmark):
            continue

        point = landmark_to_point(landmark, image_width, image_height)
        cv2.circle(frame, point, 4, (0, 255, 0), -1)
