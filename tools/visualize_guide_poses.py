import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.guide_skeleton import (
    ALIGNED_GUIDE_POSES,
    GENERATED_GUIDE_POSES,
    GUIDE_CONNECTIONS,
    GUIDE_POSES,
    NOSE,
    SHAFT_GUIDES,
)


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "reference_data"
    / "debug_guide_overlay"
    / "generated_guide_contact_sheet.jpg"
)

STAGES = [
    ("address", "1. Address"),
    ("takeaway", "2. Takeaway"),
    ("backswing", "3. Backswing"),
    ("top", "4. Top"),
    ("downswing", "5. Downswing"),
    ("impact", "6. Impact"),
    ("follow_through", "7. Follow-through"),
    ("finish", "8. Finish"),
]

PANEL_WIDTH = 440
PANEL_HEIGHT = 620


def point_to_pixel(point):
    x = round(point[0] * PANEL_WIDTH)
    y = round(55 + point[1] * 525)
    return x, y


def render_stage(stage_key, stage_label):
    canvas = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 24, dtype=np.uint8)
    guide_pose = GUIDE_POSES[stage_key]
    is_generated = GENERATED_GUIDE_POSES is not None and stage_key in GENERATED_GUIDE_POSES
    if ALIGNED_GUIDE_POSES is not None and stage_key in ALIGNED_GUIDE_POSES:
        source_label = "CADDIESET ALIGNED"
        source_color = (80, 220, 80)
    else:
        source_label = "REVIEWED DATA" if is_generated else "DEFAULT FALLBACK"
        source_color = (80, 220, 80) if is_generated else (0, 180, 255)

    cv2.putText(
        canvas,
        stage_label,
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        source_label,
        (18, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        source_color,
        1,
        cv2.LINE_AA,
    )

    for start_index, end_index in GUIDE_CONNECTIONS:
        cv2.line(
            canvas,
            point_to_pixel(guide_pose[start_index]),
            point_to_pixel(guide_pose[end_index]),
            (230, 180, 60),
            4,
            cv2.LINE_AA,
        )

    shaft = SHAFT_GUIDES.get(stage_key)
    if shaft:
        cv2.line(
            canvas,
            point_to_pixel(shaft[0]),
            point_to_pixel(shaft[1]),
            (70, 240, 240),
            5,
            cv2.LINE_AA,
        )

    for index, point in guide_pose.items():
        pixel = point_to_pixel(point)
        radius = 10 if index == NOSE else 6
        color = (80, 220, 255) if index == NOSE else (120, 240, 255)
        cv2.circle(canvas, pixel, radius, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(index),
            (pixel[0] + 7, pixel[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

    return canvas


def build_contact_sheet():
    panels = [render_stage(stage_key, label) for stage_key, label in STAGES]
    first_row = np.hstack(panels[:4])
    second_row = np.hstack(panels[4:])
    return np.vstack((first_row, second_row))


def parse_args():
    parser = argparse.ArgumentParser(description="런타임 최종 8단계 가이드를 접촉 시트로 렌더링합니다.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output.resolve()
    contact_sheet = build_contact_sheet()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), contact_sheet):
        raise OSError(f"이미지를 저장할 수 없습니다: {output_path}")
    print(f"저장 완료: {output_path}")
    print(f"크기: {contact_sheet.shape[1]}x{contact_sheet.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
