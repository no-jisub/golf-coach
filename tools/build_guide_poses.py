import json
from statistics import median
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED_DIR = PROJECT_ROOT / "reference_data" / "extracted_landmarks"
OUTPUT_PATH = PROJECT_ROOT / "reference_data" / "guide_poses" / "generated_guide_poses.json"

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

GUIDE_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def midpoint(point_a, point_b):
    return ((point_a[0] + point_b[0]) / 2, (point_a[1] + point_b[1]) / 2)


def load_landmarks(json_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not data.get("detected"):
        return None

    landmarks = {}
    for landmark in data["landmarks"]:
        index = landmark["index"]
        if index in GUIDE_LANDMARKS:
            landmarks[index] = {
                "x": landmark["x"],
                "y": landmark["y"],
                "visibility": landmark.get("visibility", 1.0),
            }

    if not all(index in landmarks for index in GUIDE_LANDMARKS):
        return None

    return landmarks


def normalize_landmarks(landmarks):
    """어깨 중심/어깨 너비/어깨-발목 높이 기준으로 프로 좌표를 정규화합니다."""
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_ankle = landmarks[27]
    right_ankle = landmarks[28]

    shoulder_mid = midpoint(
        (left_shoulder["x"], left_shoulder["y"]),
        (right_shoulder["x"], right_shoulder["y"]),
    )
    ankle_mid = midpoint(
        (left_ankle["x"], left_ankle["y"]),
        (right_ankle["x"], right_ankle["y"]),
    )

    shoulder_width = abs(right_shoulder["x"] - left_shoulder["x"])
    body_height = abs(ankle_mid[1] - shoulder_mid[1])
    if shoulder_width <= 0 or body_height <= 0:
        return None

    normalized = {}
    for index in GUIDE_LANDMARKS:
        point = landmarks[index]
        normalized[index] = {
            "x": (point["x"] - shoulder_mid[0]) / shoulder_width,
            "y": (point["y"] - shoulder_mid[1]) / body_height,
            "visibility": point["visibility"],
        }

    return normalized


def denormalize_to_guide_space(normalized):
    """현재 앱의 GUIDE_POSES 형식인 0~1 좌표계로 다시 변환합니다."""
    guide_left_shoulder_x = 0.42
    guide_right_shoulder_x = 0.58
    guide_shoulder_y = 0.28
    guide_ankle_y = 0.92

    guide_shoulder_width = guide_right_shoulder_x - guide_left_shoulder_x
    guide_body_height = guide_ankle_y - guide_shoulder_y
    guide_shoulder_mid = (0.50, guide_shoulder_y)

    guide_pose = {}
    for index in GUIDE_LANDMARKS:
        point = normalized[index]
        x = guide_shoulder_mid[0] + point["x"] * guide_shoulder_width
        y = guide_shoulder_mid[1] + point["y"] * guide_body_height
        guide_pose[str(index)] = [round(x, 4), round(y, 4)]

    return guide_pose


def build_stage_pose(stage):
    stage_dir = EXTRACTED_DIR / stage
    if not stage_dir.exists():
        return None, []

    json_paths = sorted(stage_dir.glob("*.json"))
    guide_poses = []
    used_files = []

    for json_path in json_paths:
        landmarks = load_landmarks(json_path)
        if landmarks is None:
            continue

        normalized = normalize_landmarks(landmarks)
        if normalized is None:
            continue

        guide_poses.append(denormalize_to_guide_space(normalized))
        used_files.append(str(json_path.relative_to(PROJECT_ROOT)))

    if not guide_poses:
        return None, used_files

    merged = {}
    for index in map(str, GUIDE_LANDMARKS):
        xs = [pose[index][0] for pose in guide_poses]
        ys = [pose[index][1] for pose in guide_poses]
        # 평균보다 중앙값이 잘못 찍힌 관절점의 영향을 덜 받습니다.
        merged[index] = [round(median(xs), 4), round(median(ys), 4)]

    return merged, used_files


def main():
    output = {
        "schema": "golf-coach-guide-poses-v1",
        "coordinate_system": "0_to_1_screen_like_guide_space",
        "merge_method": "median",
        "landmark_indexes": GUIDE_LANDMARKS,
        "stages": {},
        "sources": {},
    }

    for stage in STAGES:
        guide_pose, used_files = build_stage_pose(stage)
        if guide_pose is None:
            print(f"[SKIP] {stage}: usable landmark JSON 없음")
            continue

        output["stages"][stage] = guide_pose
        output["sources"][stage] = used_files
        print(f"[OK] {stage}: {len(used_files)}개 JSON 반영")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"저장 완료: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
