import cv2


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

GUIDE_CONNECTIONS = [
    (NOSE, LEFT_SHOULDER),
    (NOSE, RIGHT_SHOULDER),
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


# 0~1 normalized guide pose coordinates.
# These coordinates are both the visual guide and the current scoring reference.
GUIDE_POSES = {
    "address": {
        NOSE: (0.50, 0.12),
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
        NOSE: (0.50, 0.12),
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
        NOSE: (0.50, 0.12),
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
        NOSE: (0.50, 0.12),
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
        NOSE: (0.50, 0.12),
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
        NOSE: (0.50, 0.12),
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
        NOSE: (0.50, 0.12),
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
        NOSE: (0.50, 0.12),
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


def default_guide_point_to_pixel(point, image_width, guide_top, guide_height):
    """Convert guide coordinates to the default centered screen guide."""
    return int(point[0] * image_width), int(guide_top + point[1] * guide_height)


def landmark_to_pixel(landmark, image_width, image_height):
    """Convert a MediaPipe landmark to a pixel coordinate."""
    return int(landmark.x * image_width), int(landmark.y * image_height)


def get_user_anchor(user_landmarks, image_width, image_height):
    """Use the user's shoulder center and width as the adaptive guide anchor."""
    if user_landmarks is None:
        return None

    left_shoulder = user_landmarks[LEFT_SHOULDER]
    right_shoulder = user_landmarks[RIGHT_SHOULDER]
    if getattr(left_shoulder, "visibility", 1.0) < 0.5:
        return None
    if getattr(right_shoulder, "visibility", 1.0) < 0.5:
        return None

    left_point = landmark_to_pixel(left_shoulder, image_width, image_height)
    right_point = landmark_to_pixel(right_shoulder, image_width, image_height)
    shoulder_width = abs(right_point[0] - left_point[0])
    if shoulder_width < 20:
        return None

    shoulder_mid = (
        int((left_point[0] + right_point[0]) / 2),
        int((left_point[1] + right_point[1]) / 2),
    )
    return shoulder_mid, shoulder_width


def guide_point_to_user_pixel(point, guide_pose, user_anchor):
    """Convert guide coordinates to the user's current shoulder position and scale."""
    shoulder_mid, user_shoulder_width = user_anchor
    guide_left = guide_pose[LEFT_SHOULDER]
    guide_right = guide_pose[RIGHT_SHOULDER]
    guide_shoulder_mid = (
        (guide_left[0] + guide_right[0]) / 2,
        (guide_left[1] + guide_right[1]) / 2,
    )
    guide_shoulder_width = abs(guide_right[0] - guide_left[0]) or 0.14
    scale = user_shoulder_width / guide_shoulder_width

    x = int(shoulder_mid[0] + (point[0] - guide_shoulder_mid[0]) * scale)
    y = int(shoulder_mid[1] + (point[1] - guide_shoulder_mid[1]) * scale)
    return x, y


def draw_guide_skeleton(frame, stage_key, user_landmarks=None):
    """Draw the current stage guide, adapting to the user's position and size when possible."""
    guide_pose = GUIDE_POSES.get(stage_key)
    if not guide_pose:
        return

    image_height, image_width, _ = frame.shape
    feedback_panel_height = 180
    guide_top = 20
    guide_height = max(image_height - feedback_panel_height - 40, 120)
    user_anchor = get_user_anchor(user_landmarks, image_width, image_height)

    overlay = frame.copy()
    line_color = (255, 180, 40)
    point_color = (255, 240, 120)
    head_color = (80, 220, 255)

    def to_pixel(point):
        if user_anchor is not None:
            return guide_point_to_user_pixel(point, guide_pose, user_anchor)
        return default_guide_point_to_pixel(point, image_width, guide_top, guide_height)

    for start_idx, end_idx in GUIDE_CONNECTIONS:
        cv2.line(
            overlay,
            to_pixel(guide_pose[start_idx]),
            to_pixel(guide_pose[end_idx]),
            line_color,
            5,
        )

    for index, point in guide_pose.items():
        pixel = to_pixel(point)
        if index == NOSE:
            cv2.circle(overlay, pixel, 18, head_color, 3)
            cv2.circle(overlay, pixel, 7, head_color, -1)
        else:
            cv2.circle(overlay, pixel, 8, point_color, -1)
            cv2.circle(overlay, pixel, 10, line_color, 2)

    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
