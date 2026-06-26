import argparse
import json
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED_DIR = PROJECT_ROOT / "reference_data" / "extracted_landmarks"
OUTPUT_DIR = PROJECT_ROOT / "reference_data" / "debug_overlay"

STAGES = [
    "address",
    "takeaway",
    "backswing",
    "top",
    "downswing",
    "impact",
    "follow_through",
    "finish",
]

CONNECTIONS = [
    (0, 11),
    (0, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]

IMPORTANT_LANDMARKS = {0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}


def load_json(json_path):
    return json.loads(json_path.read_text(encoding="utf-8"))


def landmark_to_pixel(landmark, image_width, image_height):
    return int(landmark["x"] * image_width), int(landmark["y"] * image_height)


def draw_overlay(image, data):
    height, width, _ = image.shape
    landmarks = data.get("landmarks", [])
    landmark_by_index = {landmark.get("index"): landmark for landmark in landmarks}

    for start_idx, end_idx in CONNECTIONS:
        if start_idx not in landmark_by_index or end_idx not in landmark_by_index:
            continue
        start = landmark_by_index[start_idx]
        end = landmark_by_index[end_idx]
        start_point = landmark_to_pixel(start, width, height)
        end_point = landmark_to_pixel(end, width, height)
        cv2.line(image, start_point, end_point, (255, 255, 255), 3)

    shaft = data.get("shaft")
    if shaft and shaft.get("start") and shaft.get("end"):
        shaft_start = (int(shaft["start"][0] * width), int(shaft["start"][1] * height))
        shaft_end = (int(shaft["end"][0] * width), int(shaft["end"][1] * height))
        cv2.line(image, shaft_start, shaft_end, (0, 255, 0), 5)
        cv2.circle(image, shaft_start, 7, (0, 255, 0), -1)

    for landmark in landmarks:
        index = landmark["index"]
        point = landmark_to_pixel(landmark, width, height)
        color = (0, 255, 255) if index in IMPORTANT_LANDMARKS else (140, 140, 140)
        radius = 6 if index in IMPORTANT_LANDMARKS else 3
        cv2.circle(image, point, radius, color, -1)

        if index in IMPORTANT_LANDMARKS:
            cv2.putText(
                image,
                str(index),
                (point[0] + 6, point[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

    status = "detected" if data.get("detected") else "not detected"
    label = f"{data['stage']} | {Path(data['image']).name} | {status}"
    cv2.rectangle(image, (10, 10), (520, 48), (0, 0, 0), -1)
    cv2.putText(image, label, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return image


def visualize_file(json_path):
    data = load_json(json_path)
    image_path = PROJECT_ROOT / data["image"]
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[FAIL] image not found: {image_path}")
        return False

    output_stage_dir = OUTPUT_DIR / data["stage"]
    output_stage_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_stage_dir / f"{image_path.stem}_overlay.jpg"

    overlay = draw_overlay(image, data)
    cv2.imwrite(str(output_path), overlay)
    print(f"[OK] {output_path.relative_to(PROJECT_ROOT)}")
    return True


def iter_json_files(stage=None):
    stages = [stage] if stage else STAGES
    for stage_name in stages:
        stage_dir = EXTRACTED_DIR / stage_name
        if not stage_dir.exists():
            continue
        yield from sorted(stage_dir.glob("*.json"))


def main():
    parser = argparse.ArgumentParser(description="추출된 관절 좌표를 원본 이미지 위에 그려 검수용 이미지를 생성합니다.")
    parser.add_argument("--stage", choices=STAGES, help="특정 단계만 시각화합니다.")
    args = parser.parse_args()

    count = 0
    for json_path in iter_json_files(args.stage):
        if visualize_file(json_path):
            count += 1

    print()
    print(f"생성된 overlay 이미지: {count}")


if __name__ == "__main__":
    main()
