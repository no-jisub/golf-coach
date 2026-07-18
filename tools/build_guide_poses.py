import argparse
import json
from statistics import median
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED_DIR = PROJECT_ROOT / "reference_data" / "extracted_landmarks"
OUTPUT_PATH = PROJECT_ROOT / "reference_data" / "guide_poses" / "generated_guide_poses.json"
REPORT_PATH = PROJECT_ROOT / "reference_data" / "guide_poses" / "guide_build_report.json"
REVIEW_MANIFEST_PATH = PROJECT_ROOT / "reference_data" / "review_manifest.json"

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
REVIEW_SCHEMA = "golf-coach-review-v1"


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


def load_review_manifest(manifest_path=REVIEW_MANIFEST_PATH):
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"검수 manifest가 없습니다: {manifest_path}\n"
            "먼저 tools\\audit_reference_samples.py를 실행해주세요."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != REVIEW_SCHEMA:
        raise ValueError(
            f"지원하지 않는 검수 manifest 스키마입니다: {manifest.get('schema')}"
        )
    if not isinstance(manifest.get("samples"), dict):
        raise ValueError("검수 manifest에 samples 객체가 없습니다.")
    return manifest


def get_sample_review_key(json_path):
    return json_path.relative_to(PROJECT_ROOT).as_posix()


def get_sample_inclusion(json_path, review_manifest):
    """사람이 승인한 샘플만 포함하고 자동 실패는 명시적 override를 요구합니다."""
    review_key = get_sample_review_key(json_path)
    sample_review = review_manifest.get("samples", {}).get(review_key)
    if sample_review is None:
        return False, "missing_review"

    human_review = sample_review.get("human_review", {})
    human_status = human_review.get("status")
    if human_status == "pending":
        return False, "pending"
    if human_status == "rejected":
        return False, "rejected"
    if human_status != "accepted":
        return False, "invalid_human_status"

    auto_status = sample_review.get("auto_check", {}).get("status")
    if auto_status not in {"pass", "warning", "fail"}:
        return False, "invalid_auto_status"
    if auto_status == "fail" and not human_review.get("override_auto_fail", False):
        return False, "auto_fail_without_override"
    return True, "included"


def get_sample_shaft_inclusion(json_path, review_manifest):
    review_key = get_sample_review_key(json_path)
    sample_review = review_manifest.get("samples", {}).get(review_key, {})
    return sample_review.get("human_review", {}).get("include_shaft", True)


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


def build_stage_pose(stage, review_manifest):
    stage_dir = EXTRACTED_DIR / stage
    if not stage_dir.exists():
        return None, None, [], {"total": 0, "included": 0, "missing_stage_dir": 1}

    json_paths = sorted(stage_dir.glob("*.json"))
    guide_poses = []
    shaft_guides = []
    used_files = []
    review_stats = {
        "total": len(json_paths),
        "included": 0,
        "pending": 0,
        "rejected": 0,
        "missing_review": 0,
        "auto_fail_without_override": 0,
        "invalid_human_status": 0,
        "invalid_auto_status": 0,
        "unusable_landmarks": 0,
        "shaft_included": 0,
        "shaft_excluded_by_review": 0,
    }

    for json_path in json_paths:
        included, reason = get_sample_inclusion(json_path, review_manifest)
        if not included:
            review_stats[reason] = review_stats.get(reason, 0) + 1
            continue

        landmarks, shaft = load_reference_data(json_path)
        if landmarks is None:
            review_stats["unusable_landmarks"] += 1
            continue

        normalized = normalize_landmarks(landmarks)
        if normalized is None:
            review_stats["unusable_landmarks"] += 1
            continue

        guide_poses.append(denormalize_to_guide_space(normalized))
        normalized_shaft = normalize_shaft(shaft, landmarks)
        include_shaft = get_sample_shaft_inclusion(json_path, review_manifest)
        if normalized_shaft is not None and include_shaft:
            shaft_guides.append(denormalize_shaft_to_guide_space(normalized_shaft))
            review_stats["shaft_included"] += 1
        elif normalized_shaft is not None:
            review_stats["shaft_excluded_by_review"] += 1
        used_files.append(get_sample_review_key(json_path))
        review_stats["included"] += 1

    if not guide_poses:
        return None, None, used_files, review_stats

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

    return merged, merged_shaft, used_files, review_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="사람이 승인한 참조 샘플만 사용해 단계별 보조 스켈레톤을 생성합니다."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REVIEW_MANIFEST_PATH,
        help="검수 manifest 경로입니다.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_PATH,
        help="좌표를 제외한 단계별 빌드 결과 리포트 경로입니다.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = args.manifest.resolve()
    report_path = args.report.resolve()
    review_manifest = load_review_manifest(manifest_path)
    output = {
        "schema": "golf-coach-guide-poses-v1",
        "coordinate_system": "0_to_1_screen_like_guide_space",
        "merge_method": "median",
        "landmark_indexes": GUIDE_LANDMARKS,
        "stages": {},
        "shafts": {},
        "sources": {},
        "review": {
            "schema": review_manifest["schema"],
            "manifest": str(manifest_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            if manifest_path.is_relative_to(PROJECT_ROOT)
            else str(manifest_path),
            "stages": {},
        },
    }

    for stage in STAGES:
        guide_pose, shaft_guide, used_files, review_stats = build_stage_pose(stage, review_manifest)
        output["review"]["stages"][stage] = review_stats
        if guide_pose is None:
            print(
                f"[SKIP] {stage}: 승인된 사용 가능 샘플 없음 "
                f"(pending={review_stats.get('pending', 0)}, "
                f"rejected={review_stats.get('rejected', 0)}, "
                f"auto_fail={review_stats.get('auto_fail_without_override', 0)})"
            )
            continue

        output["stages"][stage] = guide_pose
        if shaft_guide is not None:
            output["shafts"][stage] = shaft_guide
        output["sources"][stage] = used_files
        shaft_text = "샤프트 있음" if shaft_guide is not None else "샤프트 없음"
        print(f"[OK] {stage}: 승인 샘플 {len(used_files)}개 반영, {shaft_text}")

    if not output["stages"]:
        print()
        print("승인된 사용 가능 샘플이 없어 기존 guide pose 파일을 변경하지 않았습니다.")
        print("reference_data\\review_manifest.json에서 샘플을 검수한 뒤 human_review.status를 accepted로 변경해주세요.")
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    generated_stages = list(output["stages"])
    fallback_stages = [stage for stage in STAGES if stage not in output["stages"]]
    report = {
        "schema": "golf-coach-guide-build-report-v1",
        "review_manifest": output["review"]["manifest"],
        "generated_stages": generated_stages,
        "fallback_stages": fallback_stages,
        "stage_stats": output["review"]["stages"],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"저장 완료: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"빌드 리포트: {report_path.relative_to(PROJECT_ROOT) if report_path.is_relative_to(PROJECT_ROOT) else report_path}")
    if fallback_stages:
        print(f"기본 가이드 폴백 단계: {', '.join(fallback_stages)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
