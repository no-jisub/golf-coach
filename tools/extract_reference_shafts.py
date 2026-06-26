import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED_DIR = PROJECT_ROOT / "reference_data" / "extracted_landmarks"
OUTPUT_DIR = PROJECT_ROOT / "reference_data" / "debug_shaft_overlay"

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

LEFT_WRIST = 15
RIGHT_WRIST = 16


def load_json(json_path):
    return json.loads(json_path.read_text(encoding="utf-8"))


def save_json(json_path, data):
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def landmark_to_pixel(landmark, width, height):
    return np.array([landmark["x"] * width, landmark["y"] * height], dtype=np.float32)


def point_to_segment_distance(point, start, end):
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-6:
        return float(np.linalg.norm(point - start))

    t = float(np.dot(point - start, segment) / denom)
    t = max(0.0, min(1.0, t))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


def normalize_point(point, width, height):
    return [round(float(point[0] / width), 6), round(float(point[1] / height), 6)]


def detect_line_candidates(image):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    min_line_length = int(max(width, height) * 0.16)
    max_line_gap = int(max(width, height) * 0.035)
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if raw_lines is None:
        return []

    candidates = []
    for line in raw_lines[:, 0]:
        x1, y1, x2, y2 = map(float, line)
        start = np.array([x1, y1], dtype=np.float32)
        end = np.array([x2, y2], dtype=np.float32)
        length = float(np.linalg.norm(end - start))
        if length < min_line_length:
            continue
        candidates.append((start, end, length))

    return candidates


def score_candidate(start, end, length, grip_point, image_shape):
    height, width = image_shape[:2]
    diagonal = math.hypot(width, height)
    distance_to_grip = point_to_segment_distance(grip_point, start, end)
    endpoint_distance = min(float(np.linalg.norm(grip_point - start)), float(np.linalg.norm(grip_point - end)))

    # 샤프트는 보통 손목 주변을 지나고 길이가 긴 선입니다.
    distance_score = max(0.0, 1.0 - distance_to_grip / (diagonal * 0.10))
    endpoint_score = max(0.0, 1.0 - endpoint_distance / (diagonal * 0.22))
    length_score = min(1.0, length / (diagonal * 0.55))
    return length_score * 0.45 + distance_score * 0.40 + endpoint_score * 0.15


def extract_shaft(data, image):
    if not data.get("detected") or not data.get("landmarks"):
        return None, []

    height, width = image.shape[:2]
    landmarks = data["landmarks"]
    if len(landmarks) <= max(LEFT_WRIST, RIGHT_WRIST):
        return None, []

    left_wrist = landmark_to_pixel(landmarks[LEFT_WRIST], width, height)
    right_wrist = landmark_to_pixel(landmarks[RIGHT_WRIST], width, height)
    grip_point = (left_wrist + right_wrist) / 2

    scored = []
    for start, end, length in detect_line_candidates(image):
        score = score_candidate(start, end, length, grip_point, image.shape)
        if score < 0.35:
            continue
        scored.append({
            "start_px": start,
            "end_px": end,
            "length": length,
            "score": score,
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    top = scored[:5]
    candidates = [
        {
            "start": normalize_point(item["start_px"], width, height),
            "end": normalize_point(item["end_px"], width, height),
            "score": round(float(item["score"]), 4),
            "length_px": round(float(item["length"]), 1),
        }
        for item in top
    ]

    if not top:
        return None, candidates

    best = top[0]
    shaft = {
        "source": "opencv_hough",
        "start": normalize_point(best["start_px"], width, height),
        "end": normalize_point(best["end_px"], width, height),
        "score": round(float(best["score"]), 4),
        "length_px": round(float(best["length"]), 1),
        "candidates": candidates,
    }
    return shaft, candidates


def draw_overlay(image, data, shaft):
    overlay = image.copy()
    height, width = overlay.shape[:2]
    landmarks = data.get("landmarks", [])

    if len(landmarks) > max(LEFT_WRIST, RIGHT_WRIST):
        left_wrist = tuple(landmark_to_pixel(landmarks[LEFT_WRIST], width, height).astype(int))
        right_wrist = tuple(landmark_to_pixel(landmarks[RIGHT_WRIST], width, height).astype(int))
        grip = tuple(((np.array(left_wrist) + np.array(right_wrist)) / 2).astype(int))
        cv2.circle(overlay, left_wrist, 8, (255, 255, 0), -1)
        cv2.circle(overlay, right_wrist, 8, (255, 255, 0), -1)
        cv2.circle(overlay, grip, 8, (0, 255, 255), -1)

    if shaft:
        for index, candidate in enumerate(shaft.get("candidates", [])):
            start = (int(candidate["start"][0] * width), int(candidate["start"][1] * height))
            end = (int(candidate["end"][0] * width), int(candidate["end"][1] * height))
            color = (0, 255, 0) if index == 0 else (120, 120, 255)
            thickness = 4 if index == 0 else 2
            cv2.line(overlay, start, end, color, thickness)
            cv2.putText(
                overlay,
                f"{index + 1}:{candidate['score']:.2f}",
                start,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

    status = "shaft detected" if shaft else "shaft not detected"
    label = f"{data.get('stage')} | {Path(data.get('image', '')).name} | {status}"
    cv2.rectangle(overlay, (10, 10), (760, 48), (0, 0, 0), -1)
    cv2.putText(overlay, label, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return overlay


def process_json(json_path, write=True):
    data = load_json(json_path)
    image_path = PROJECT_ROOT / data["image"]
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[FAIL] image not found: {image_path}")
        return False

    shaft, candidates = extract_shaft(data, image)
    if shaft:
        data["shaft"] = shaft
    else:
        data["shaft"] = {
            "source": "opencv_hough",
            "detected": False,
            "candidates": candidates,
        }

    if write:
        save_json(json_path, data)

    output_stage_dir = OUTPUT_DIR / data["stage"]
    output_stage_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_stage_dir / f"{image_path.stem}_shaft_overlay.jpg"
    cv2.imwrite(str(output_path), draw_overlay(image, data, shaft))

    if shaft:
        print(f"[OK] {json_path.relative_to(PROJECT_ROOT)} score={shaft['score']:.2f} -> {output_path.relative_to(PROJECT_ROOT)}")
        return True

    print(f"[MISS] {json_path.relative_to(PROJECT_ROOT)} -> {output_path.relative_to(PROJECT_ROOT)}")
    return False


def iter_json_files(stage=None, only=None):
    stages = [stage] if stage else STAGES
    for stage_name in stages:
        stage_dir = EXTRACTED_DIR / stage_name
        if not stage_dir.exists():
            continue
        for json_path in sorted(stage_dir.glob("*.json")):
            if only and only not in json_path.name:
                continue
            yield json_path


def main():
    parser = argparse.ArgumentParser(description="OpenCV 직선 검출로 참조 이미지의 골프 샤프트 후보를 추출합니다.")
    parser.add_argument("--stage", choices=STAGES, help="특정 단계만 처리합니다.")
    parser.add_argument("--only", help="파일명에 포함된 문자열만 처리합니다. 예: pro02")
    parser.add_argument("--dry-run", action="store_true", help="JSON에는 저장하지 않고 오버레이만 생성합니다.")
    args = parser.parse_args()

    total = 0
    detected = 0
    for json_path in iter_json_files(args.stage, args.only):
        total += 1
        if process_json(json_path, write=not args.dry_run):
            detected += 1

    print()
    print(f"처리 파일: {total}")
    print(f"샤프트 검출: {detected}")
    print(f"샤프트 미검출: {total - detected}")


if __name__ == "__main__":
    main()
