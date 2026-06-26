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
MIN_SHOULDER_TO_BODY_RATIO = 0.04


def midpoint(point_a, point_b):
    return ((point_a[0] + point_b[0]) / 2, (point_a[1] + point_b[1]) / 2)


def load_reference_data(json_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not data.get("detected"):
        return None, None

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
        return None, None

    shaft = data.get("shaft")
    if shaft and shaft.get("start") and shaft.get("end"):
        shaft = {
            "start": tuple(shaft["start"]),
            "end": tuple(shaft["end"]),
            "source": shaft.get("source", "unknown"),
            "score": shaft.get("score"),
        }
    else:
        shaft = None

    return landmarks, shaft


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
    if shoulder_width / body_height < MIN_SHOULDER_TO_BODY_RATIO:
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


def normalize_shaft(shaft, landmarks):
    """샤프트 양 끝점을 관절 좌표와 같은 어깨 기준 정규화 공간으로 변환합니다."""
    if shaft is None:
        return None

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
    if shoulder_width / body_height < MIN_SHOULDER_TO_BODY_RATIO:
        return None

    return {
        key: {
            "x": (point[0] - shoulder_mid[0]) / shoulder_width,
            "y": (point[1] - shoulder_mid[1]) / body_height,
        }
        for key, point in (("start", shaft["start"]), ("end", shaft["end"]))
    }


def denormalize_shaft_to_guide_space(normalized):
    guide_left_shoulder_x = 0.42
    guide_right_shoulder_x = 0.58
    guide_shoulder_y = 0.28
    guide_ankle_y = 0.92

    guide_shoulder_width = guide_right_shoulder_x - guide_left_shoulder_x
    guide_body_height = guide_ankle_y - guide_shoulder_y
    guide_shoulder_mid = (0.50, guide_shoulder_y)

    return {
        key: [
            round(guide_shoulder_mid[0] + point["x"] * guide_shoulder_width, 4),
            round(guide_shoulder_mid[1] + point["y"] * guide_body_height, 4),
        ]
        for key, point in normalized.items()
    }


def build_stage_pose(stage):
    stage_dir = EXTRACTED_DIR / stage
    if not stage_dir.exists():
        return None, None, []

    json_paths = sorted(stage_dir.glob("*.json"))
    guide_poses = []
    shaft_guides = []
    used_files = []

    for json_path in json_paths:
        landmarks, shaft = load_reference_data(json_path)
        if landmarks is None:
            continue

        normalized = normalize_landmarks(landmarks)
        if normalized is None:
            continue

        guide_poses.append(denormalize_to_guide_space(normalized))
        normalized_shaft = normalize_shaft(shaft, landmarks)
        if normalized_shaft is not None:
            shaft_guides.append(denormalize_shaft_to_guide_space(normalized_shaft))
        used_files.append(str(json_path.relative_to(PROJECT_ROOT)))

    if not guide_poses:
        return None, None, used_files

    merged = {}
    for index in map(str, GUIDE_LANDMARKS):
        xs = [pose[index][0] for pose in guide_poses]
        ys = [pose[index][1] for pose in guide_poses]
        # 평균보다 중앙값이 잘못 찍힌 관절점의 영향을 덜 받습니다.
        merged[index] = [round(median(xs), 4), round(median(ys), 4)]

    merged_shaft = None
    if shaft_guides:
        merged_shaft = {}
        for key in ("start", "end"):
            xs = [shaft[key][0] for shaft in shaft_guides]
            ys = [shaft[key][1] for shaft in shaft_guides]
            merged_shaft[key] = [round(median(xs), 4), round(median(ys), 4)]

    return merged, merged_shaft, used_files


def main():
    output = {
        "schema": "golf-coach-guide-poses-v1",
        "coordinate_system": "0_to_1_screen_like_guide_space",
        "merge_method": "median",
        "landmark_indexes": GUIDE_LANDMARKS,
        "stages": {},
        "shafts": {},
        "sources": {},
    }

    for stage in STAGES:
        guide_pose, shaft_guide, used_files = build_stage_pose(stage)
        if guide_pose is None:
            print(f"[SKIP] {stage}: usable landmark JSON 없음")
            continue

        output["stages"][stage] = guide_pose
        if shaft_guide is not None:
            output["shafts"][stage] = shaft_guide
        output["sources"][stage] = used_files
        shaft_text = "샤프트 있음" if shaft_guide is not None else "샤프트 없음"
        print(f"[OK] {stage}: {len(used_files)}개 JSON 반영, {shaft_text}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"저장 완료: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
