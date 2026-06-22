import math

import numpy as np


def calculate_angle(point_a, point_b, point_c):
    """세 점 A-B-C에서 B를 중심으로 한 각도를 계산합니다."""
    a = np.array(point_a, dtype=float)
    b = np.array(point_b, dtype=float)
    c = np.array(point_c, dtype=float)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0:
        return 0.0

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)
    return math.degrees(math.acos(cosine))
