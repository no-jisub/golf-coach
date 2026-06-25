import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "pose_landmarker_full.task"
RAW_IMAGES_DIR = PROJECT_ROOT / "reference_data" / "raw_images"
OUTPUT_DIR = PROJECT_ROOT / "reference_data" / "extracted_landmarks"

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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

LANDMARK_NAMES = {
    0: ("코", "nose"),
    1: ("왼쪽 눈 안쪽", "left_eye_inner"),
    2: ("왼쪽 눈", "left_eye"),
    3: ("왼쪽 눈 바깥쪽", "left_eye_outer"),
    4: ("오른쪽 눈 안쪽", "right_eye_inner"),
    5: ("오른쪽 눈", "right_eye"),
    6: ("오른쪽 눈 바깥쪽", "right_eye_outer"),
    7: ("왼쪽 귀", "left_ear"),
    8: ("오른쪽 귀", "right_ear"),
    9: ("입 왼쪽", "mouth_left"),
    10: ("입 오른쪽", "mouth_right"),
    11: ("왼쪽 어깨", "left_shoulder"),
    12: ("오른쪽 어깨", "right_shoulder"),
    13: ("왼쪽 팔꿈치", "left_elbow"),
    14: ("오른쪽 팔꿈치", "right_elbow"),
    15: ("왼쪽 손목", "left_wrist"),
    16: ("오른쪽 손목", "right_wrist"),
    17: ("왼쪽 새끼손가락", "left_pinky"),
    18: ("오른쪽 새끼손가락", "right_pinky"),
    19: ("왼쪽 검지", "left_index"),
    20: ("오른쪽 검지", "right_index"),
    21: ("왼쪽 엄지", "left_thumb"),
    22: ("오른쪽 엄지", "right_thumb"),
    23: ("왼쪽 골반", "left_hip"),
    24: ("오른쪽 골반", "right_hip"),
    25: ("왼쪽 무릎", "left_knee"),
    26: ("오른쪽 무릎", "right_knee"),
    27: ("왼쪽 발목", "left_ankle"),
    28: ("오른쪽 발목", "right_ankle"),
    29: ("왼쪽 뒤꿈치", "left_heel"),
    30: ("오른쪽 뒤꿈치", "right_heel"),
    31: ("왼쪽 발끝", "left_foot_index"),
    32: ("오른쪽 발끝", "right_foot_index"),
}


def create_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def image_to_mp_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    return image, mp_image


def landmark_to_dict(index, landmark):
    name_ko, name_en = LANDMARK_NAMES.get(index, (f"{index}번 관절", f"landmark_{index}"))
    return {
        "index": index,
        "name_ko": name_ko,
        "name_en": name_en,
        "x": landmark.x,
        "y": landmark.y,
        "z": landmark.z,
        "visibility": getattr(landmark, "visibility", 1.0),
        "presence": getattr(landmark, "presence", 1.0),
    }


def extract_image(landmarker, stage, image_path):
    image, mp_image = image_to_mp_image(image_path)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return {
            "stage": stage,
            "image": str(image_path.relative_to(PROJECT_ROOT)),
            "detected": False,
            "image_width": image.shape[1],
            "image_height": image.shape[0],
            "landmarks": [],
        }

    landmarks = result.pose_landmarks[0]
    return {
        "stage": stage,
        "image": str(image_path.relative_to(PROJECT_ROOT)),
        "detected": True,
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "landmarks": [landmark_to_dict(index, landmark) for index, landmark in enumerate(landmarks)],
    }


def iter_stage_images(stage):
    stage_dir = RAW_IMAGES_DIR / stage
    if not stage_dir.exists():
        return []

    return sorted(
        path
        for path in stage_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def save_result(result, image_path):
    stage = result["stage"]
    stage_output_dir = OUTPUT_DIR / stage
    stage_output_dir.mkdir(parents=True, exist_ok=True)

    output_path = stage_output_dir / f"{image_path.stem}.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description="프로 골프 자세 이미지에서 Pose Landmarker 좌표를 추출합니다.")
    parser.add_argument("--stage", choices=STAGES, help="특정 단계만 추출합니다.")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 필요합니다: {MODEL_PATH}")

    stages = [args.stage] if args.stage else STAGES
    extracted_count = 0
    failed_count = 0

    with create_landmarker() as landmarker:
        for stage in stages:
            image_paths = iter_stage_images(stage)
            if not image_paths:
                print(f"[SKIP] {stage}: 이미지 없음")
                continue

            for image_path in image_paths:
                result = extract_image(landmarker, stage, image_path)
                output_path = save_result(result, image_path)

                if result["detected"]:
                    extracted_count += 1
                    print(f"[OK] {stage}: {image_path.name} -> {output_path.relative_to(PROJECT_ROOT)}")
                else:
                    failed_count += 1
                    print(f"[FAIL] {stage}: {image_path.name} 포즈 인식 실패 -> {output_path.relative_to(PROJECT_ROOT)}")

    print()
    print(f"추출 성공: {extracted_count}")
    print(f"추출 실패: {failed_count}")


if __name__ == "__main__":
    main()
