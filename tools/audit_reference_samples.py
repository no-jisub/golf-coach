import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED_DIR = PROJECT_ROOT / "reference_data" / "extracted_landmarks"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reference_data" / "review_manifest.json"
REVIEW_SCHEMA = "golf-coach-review-v1"

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

REQUIRED_LANDMARKS = [
    NOSE,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
]

VISIBILITY_THRESHOLDS = {
    NOSE: 0.60,
    LEFT_SHOULDER: 0.60,
    RIGHT_SHOULDER: 0.60,
    LEFT_ELBOW: 0.50,
    RIGHT_ELBOW: 0.50,
    LEFT_WRIST: 0.50,
    RIGHT_WRIST: 0.50,
    LEFT_HIP: 0.60,
    RIGHT_HIP: 0.60,
    LEFT_KNEE: 0.60,
    RIGHT_KNEE: 0.60,
    LEFT_ANKLE: 0.60,
    RIGHT_ANKLE: 0.60,
}

MIN_SHOULDER_TO_BODY_RATIO = 0.06
SHAFT_FAIL_SCORE = 0.40
SHAFT_WARNING_SCORE = 0.50
MAX_GRIP_ENDPOINT_DISTANCE = 0.12
OUTLIER_MIN_MARGIN = 0.25
OUTLIER_MAD_MULTIPLIER = 3.0
MIN_DYNAMIC_STANCE_RATIO = 0.55
MAX_FINISH_STANCE_RATIO = 1.00


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_reason(code, severity, message):
    return {
        "code": code,
        "severity": severity,
        "message": message,
    }


def status_from_reasons(reasons):
    severities = {reason["severity"] for reason in reasons}
    if "fail" in severities:
        return "fail"
    if "warning" in severities:
        return "warning"
    return "pass"


def is_finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def midpoint(point_a, point_b):
    return ((point_a[0] + point_b[0]) / 2.0, (point_a[1] + point_b[1]) / 2.0)


def distance(point_a, point_b):
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def get_landmark_map(data):
    return {
        landmark.get("index"): landmark
        for landmark in data.get("landmarks", [])
        if isinstance(landmark, dict) and isinstance(landmark.get("index"), int)
    }


def get_body_geometry(landmarks):
    left_shoulder = landmarks[LEFT_SHOULDER]
    right_shoulder = landmarks[RIGHT_SHOULDER]
    left_ankle = landmarks[LEFT_ANKLE]
    right_ankle = landmarks[RIGHT_ANKLE]

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
    ratio = shoulder_width / body_height if body_height > 0 else 0.0
    return shoulder_mid, shoulder_width, body_height, ratio


def normalize_landmarks(landmarks):
    shoulder_mid, shoulder_width, body_height, _ = get_body_geometry(landmarks)
    if shoulder_width <= 0 or body_height <= 0:
        return None

    return {
        index: (
            (landmarks[index]["x"] - shoulder_mid[0]) / shoulder_width,
            (landmarks[index]["y"] - shoulder_mid[1]) / body_height,
        )
        for index in REQUIRED_LANDMARKS
    }


def get_grip_endpoint_distance(data, landmarks):
    shaft = data.get("shaft")
    if not shaft or not shaft.get("start") or not shaft.get("end"):
        return None

    start = shaft["start"]
    end = shaft["end"]
    if len(start) != 2 or len(end) != 2:
        return None
    if not all(is_finite_number(value) for value in (*start, *end)):
        return None

    left_wrist = landmarks[LEFT_WRIST]
    right_wrist = landmarks[RIGHT_WRIST]
    grip = midpoint(
        (left_wrist["x"], left_wrist["y"]),
        (right_wrist["x"], right_wrist["y"]),
    )
    return min(distance(grip, tuple(start)), distance(grip, tuple(end)))


def default_human_review():
    return {
        "status": "pending",
        "override_auto_fail": False,
        "note": "",
    }


def audit_sample(json_path, project_root=PROJECT_ROOT):
    try:
        relative_path = json_path.relative_to(project_root).as_posix()
    except ValueError:
        relative_path = json_path.as_posix()
    reasons = []
    metrics = {}

    try:
        data = load_json(json_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        reasons.append(make_reason("invalid_json", "fail", f"JSON을 읽을 수 없습니다: {error}"))
        return {
            "stage": json_path.parent.name,
            "source": relative_path,
            "image": None,
            "auto_check": {
                "status": "fail",
                "reasons": reasons,
                "metrics": metrics,
            },
            "human_review": default_human_review(),
            "_normalized_pose": None,
        }

    stage = data.get("stage") or json_path.parent.name
    if stage not in STAGES:
        reasons.append(make_reason("unknown_stage", "fail", f"지원하지 않는 단계입니다: {stage}"))

    if not data.get("detected"):
        reasons.append(make_reason("pose_not_detected", "fail", "포즈가 검출되지 않았습니다."))

    landmarks = get_landmark_map(data)
    missing = [index for index in REQUIRED_LANDMARKS if index not in landmarks]
    if missing:
        reasons.append(
            make_reason(
                "missing_landmarks",
                "fail",
                f"필수 관절이 없습니다: {', '.join(map(str, missing))}",
            )
        )

    invalid_coordinates = []
    out_of_bounds = []
    low_visibility = []
    if not missing:
        for index in REQUIRED_LANDMARKS:
            landmark = landmarks[index]
            x = landmark.get("x")
            y = landmark.get("y")
            if not is_finite_number(x) or not is_finite_number(y):
                invalid_coordinates.append(index)
                continue
            if x < -0.10 or x > 1.10 or y < -0.10 or y > 1.10:
                out_of_bounds.append(index)

            visibility = landmark.get("visibility", 1.0)
            threshold = VISIBILITY_THRESHOLDS[index]
            if not is_finite_number(visibility) or visibility < threshold:
                low_visibility.append((index, visibility, threshold))

    if invalid_coordinates:
        reasons.append(
            make_reason(
                "invalid_coordinates",
                "fail",
                f"좌표가 숫자가 아닌 관절이 있습니다: {', '.join(map(str, invalid_coordinates))}",
            )
        )
    if out_of_bounds:
        reasons.append(
            make_reason(
                "coordinates_out_of_bounds",
                "fail",
                f"화면 범위를 크게 벗어난 관절이 있습니다: {', '.join(map(str, out_of_bounds))}",
            )
        )
    if low_visibility:
        details = ", ".join(
            f"{index}={visibility if is_finite_number(visibility) else 'invalid'}<{threshold}"
            for index, visibility, threshold in low_visibility
        )
        reasons.append(make_reason("low_landmark_visibility", "fail", f"관절 가시성이 낮습니다: {details}"))

    normalized_pose = None
    if not missing and not invalid_coordinates:
        _, shoulder_width, body_height, body_ratio = get_body_geometry(landmarks)
        metrics["shoulder_width"] = round(shoulder_width, 5)
        metrics["body_height"] = round(body_height, 5)
        metrics["shoulder_to_body_ratio"] = round(body_ratio, 5)
        if shoulder_width <= 0 or body_height <= 0:
            reasons.append(make_reason("invalid_body_geometry", "fail", "어깨 너비 또는 신체 높이가 0입니다."))
        elif body_ratio < MIN_SHOULDER_TO_BODY_RATIO:
            reasons.append(
                make_reason(
                    "narrow_shoulder_ratio",
                    "fail",
                    f"어깨/신체 비율이 너무 작습니다: {body_ratio:.3f}",
                )
            )
        else:
            normalized_pose = normalize_landmarks(landmarks)

        visibility_values = [landmarks[index].get("visibility", 1.0) for index in REQUIRED_LANDMARKS]
        min_visibility = min(
            float(value) if is_finite_number(value) else -1.0
            for value in visibility_values
        )
        metrics["min_visibility"] = round(float(min_visibility), 5)

        grip_distance = get_grip_endpoint_distance(data, landmarks)
        if grip_distance is not None:
            metrics["shaft_grip_endpoint_distance"] = round(grip_distance, 5)
            if grip_distance > MAX_GRIP_ENDPOINT_DISTANCE:
                reasons.append(
                    make_reason(
                        "shaft_far_from_grip",
                        "warning",
                        f"샤프트 양 끝점이 손에서 멉니다: {grip_distance:.3f}",
                    )
                )

    shaft = data.get("shaft")
    if not shaft:
        reasons.append(make_reason("shaft_missing", "warning", "샤프트 검출 결과가 없습니다."))
    else:
        shaft_score = shaft.get("score")
        if not is_finite_number(shaft_score):
            reasons.append(make_reason("shaft_score_missing", "warning", "샤프트 점수가 없습니다."))
        else:
            metrics["shaft_score"] = round(float(shaft_score), 5)
            if shaft_score < SHAFT_FAIL_SCORE:
                reasons.append(
                    make_reason("shaft_score_too_low", "fail", f"샤프트 점수가 너무 낮습니다: {shaft_score:.3f}")
                )
            elif shaft_score < SHAFT_WARNING_SCORE:
                reasons.append(
                    make_reason("shaft_score_low", "warning", f"샤프트 점수가 낮습니다: {shaft_score:.3f}")
                )

    return {
        "stage": stage,
        "source": relative_path,
        "image": data.get("image"),
        "auto_check": {
            "status": status_from_reasons(reasons),
            "reasons": reasons,
            "metrics": metrics,
        },
        "human_review": default_human_review(),
        "_normalized_pose": normalized_pose,
    }


def add_stage_outlier_checks(samples):
    by_stage = defaultdict(list)
    for sample in samples.values():
        if sample.get("_normalized_pose") is not None:
            by_stage[sample["stage"]].append(sample)

    for stage_samples in by_stage.values():
        if len(stage_samples) < 4:
            continue

        stage_median = {
            index: (
                median(sample["_normalized_pose"][index][0] for sample in stage_samples),
                median(sample["_normalized_pose"][index][1] for sample in stage_samples),
            )
            for index in REQUIRED_LANDMARKS
        }
        deviations = []
        for sample in stage_samples:
            deviation = sum(
                distance(sample["_normalized_pose"][index], stage_median[index])
                for index in REQUIRED_LANDMARKS
            ) / len(REQUIRED_LANDMARKS)
            deviations.append(deviation)
            sample["auto_check"]["metrics"]["stage_pose_deviation"] = round(deviation, 5)

        center = median(deviations)
        mad = median(abs(value - center) for value in deviations)
        threshold = center + max(OUTLIER_MAD_MULTIPLIER * mad, OUTLIER_MIN_MARGIN)
        for sample, deviation in zip(stage_samples, deviations):
            sample["auto_check"]["metrics"]["stage_outlier_threshold"] = round(threshold, 5)
            if deviation > threshold:
                sample["auto_check"]["reasons"].append(
                    make_reason(
                        "stage_pose_outlier",
                        "warning",
                        f"같은 단계의 중앙 자세에서 크게 벗어났습니다: {deviation:.3f}>{threshold:.3f}",
                    )
                )
                sample["auto_check"]["status"] = status_from_reasons(sample["auto_check"]["reasons"])


def get_source_id(sample):
    stem = Path(sample["source"]).stem
    return stem.split("_video_", 1)[0]


def add_semantic_reason(sample, code, message):
    sample["auto_check"]["reasons"].append(make_reason(code, "fail", message))
    sample["auto_check"]["status"] = status_from_reasons(sample["auto_check"]["reasons"])


def add_stage_semantic_checks(samples):
    """단계 라벨과 명백히 맞지 않는 손 높이·스탠스 변화를 걸러냅니다."""
    address_stance_by_source = {}
    for sample in samples.values():
        pose = sample.get("_normalized_pose")
        if pose is None or sample.get("stage") != "address":
            continue
        stance_width = abs(pose[RIGHT_ANKLE][0] - pose[LEFT_ANKLE][0])
        body_ratio = sample["auto_check"]["metrics"].get("shoulder_to_body_ratio")
        stance_to_body = stance_width * body_ratio if body_ratio else None
        if stance_to_body and stance_to_body > 0:
            address_stance_by_source[get_source_id(sample)] = stance_to_body

    for sample in samples.values():
        pose = sample.get("_normalized_pose")
        if pose is None:
            continue

        stage = sample.get("stage")
        wrist_y = (pose[LEFT_WRIST][1] + pose[RIGHT_WRIST][1]) / 2.0
        hip_y = (pose[LEFT_HIP][1] + pose[RIGHT_HIP][1]) / 2.0
        stance_width = abs(pose[RIGHT_ANKLE][0] - pose[LEFT_ANKLE][0])
        metrics = sample["auto_check"]["metrics"]
        metrics["normalized_wrist_y"] = round(wrist_y, 5)
        metrics["normalized_hip_y"] = round(hip_y, 5)
        metrics["normalized_stance_width"] = round(stance_width, 5)
        body_ratio = metrics.get("shoulder_to_body_ratio")
        stance_to_body = stance_width * body_ratio if body_ratio else None
        if stance_to_body is not None:
            metrics["normalized_stance_to_body_height"] = round(stance_to_body, 5)

        address_stance = address_stance_by_source.get(get_source_id(sample))
        stance_ratio = None
        if address_stance and stance_to_body is not None:
            stance_ratio = stance_to_body / address_stance
            metrics["stance_to_address_ratio"] = round(stance_ratio, 5)

        if stage == "address" and wrist_y < hip_y - 0.08:
            add_semantic_reason(
                sample,
                "address_hands_too_high",
                "어드레스인데 손이 골반보다 지나치게 높습니다.",
            )
        elif stage == "takeaway" and wrist_y < -0.08:
            add_semantic_reason(
                sample,
                "takeaway_hands_too_high",
                "테이크어웨이인데 손이 이미 어깨 위에 있습니다.",
            )
        elif stage == "backswing" and wrist_y > hip_y + 0.08:
            add_semantic_reason(
                sample,
                "backswing_hands_too_low",
                "백스윙인데 손이 골반 아래에 있습니다.",
            )
        elif stage == "top":
            if wrist_y > -0.03:
                add_semantic_reason(
                    sample,
                    "top_hands_not_high_enough",
                    "백스윙 탑인데 손이 어깨 위에 있지 않습니다.",
                )
            if stance_ratio is not None and stance_ratio < MIN_DYNAMIC_STANCE_RATIO:
                add_semantic_reason(
                    sample,
                    "top_stance_looks_finished",
                    "백스윙 탑인데 스탠스가 피니시처럼 지나치게 좁습니다.",
                )
        elif stage == "downswing":
            if wrist_y < -0.12:
                add_semantic_reason(
                    sample,
                    "downswing_hands_still_at_top",
                    "다운스윙인데 손이 아직 탑 높이에 있습니다.",
                )
            if stance_ratio is not None and stance_ratio < MIN_DYNAMIC_STANCE_RATIO:
                add_semantic_reason(
                    sample,
                    "downswing_stance_looks_finished",
                    "다운스윙인데 스탠스가 피니시처럼 지나치게 좁습니다.",
                )
        elif stage == "impact":
            if wrist_y < 0.10:
                add_semantic_reason(
                    sample,
                    "impact_hands_too_high",
                    "임팩트인데 손이 어깨 근처 또는 그 위에 있습니다.",
                )
            if stance_ratio is not None and stance_ratio < MIN_DYNAMIC_STANCE_RATIO:
                add_semantic_reason(
                    sample,
                    "impact_stance_looks_finished",
                    "임팩트인데 스탠스가 피니시처럼 지나치게 좁습니다.",
                )
        elif stage == "follow_through":
            if stance_ratio is not None and stance_ratio < MIN_DYNAMIC_STANCE_RATIO:
                add_semantic_reason(
                    sample,
                    "follow_through_stance_looks_finished",
                    "팔로우스루인데 양발이 이미 피니시처럼 모였습니다.",
                )
        elif stage == "finish":
            if wrist_y > 0.05:
                add_semantic_reason(
                    sample,
                    "finish_hands_too_low",
                    "피니시인데 손이 어깨보다 아래에 있습니다.",
                )
            if stance_ratio is not None and stance_ratio > MAX_FINISH_STANCE_RATIO:
                add_semantic_reason(
                    sample,
                    "finish_stance_still_address_width",
                    "피니시인데 스탠스가 어드레스와 거의 같은 폭입니다.",
                )


def load_existing_reviews(manifest_path):
    if manifest_path is None or not manifest_path.exists():
        return {}
    try:
        manifest = load_json(manifest_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return {
        key: sample.get("human_review", default_human_review())
        for key, sample in manifest.get("samples", {}).items()
    }


def build_manifest(
    extracted_dir=EXTRACTED_DIR,
    existing_manifest_path=None,
    preserve_reviews=True,
    project_root=PROJECT_ROOT,
):
    samples = {}
    for stage in STAGES:
        stage_dir = extracted_dir / stage
        if not stage_dir.exists():
            continue
        for json_path in sorted(stage_dir.glob("*.json")):
            sample = audit_sample(json_path, project_root=project_root)
            samples[sample["source"]] = sample

    add_stage_outlier_checks(samples)
    add_stage_semantic_checks(samples)

    if preserve_reviews:
        existing_reviews = load_existing_reviews(existing_manifest_path)
        for key, human_review in existing_reviews.items():
            if key in samples:
                samples[key]["human_review"] = human_review

    for sample in samples.values():
        sample.pop("_normalized_pose", None)

    summary = {"pass": 0, "warning": 0, "fail": 0}
    for sample in samples.values():
        summary[sample["auto_check"]["status"]] += 1

    return {
        "schema": REVIEW_SCHEMA,
        "policy": {
            "required_landmarks": REQUIRED_LANDMARKS,
            "visibility_thresholds": {str(index): value for index, value in VISIBILITY_THRESHOLDS.items()},
            "min_shoulder_to_body_ratio": MIN_SHOULDER_TO_BODY_RATIO,
            "shaft_fail_score": SHAFT_FAIL_SCORE,
            "shaft_warning_score": SHAFT_WARNING_SCORE,
            "max_grip_endpoint_distance": MAX_GRIP_ENDPOINT_DISTANCE,
            "stage_outlier_method": "median_pose_distance_with_mad",
            "min_dynamic_stance_ratio": MIN_DYNAMIC_STANCE_RATIO,
            "max_finish_stance_ratio": MAX_FINISH_STANCE_RATIO,
            "stage_semantic_checks": "normalized_wrist_height_and_stance_vs_address",
        },
        "summary": {
            "total": len(samples),
            "auto_check": summary,
        },
        "samples": samples,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="참조 자세 JSON의 자동 품질 검사를 수행합니다.")
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=EXTRACTED_DIR,
        help="단계별 관절 JSON 디렉터리입니다.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="검수 manifest 저장 경로입니다.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="파일을 쓰지 않고 manifest JSON을 표준 출력으로 표시합니다.",
    )
    parser.add_argument(
        "--reset-human-review",
        action="store_true",
        help="기존 사람 검수 상태를 보존하지 않고 pending으로 초기화합니다.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    extracted_dir = args.extracted_dir.resolve()
    output_path = args.output.resolve()
    manifest = build_manifest(
        extracted_dir=extracted_dir,
        existing_manifest_path=output_path,
        preserve_reviews=not args.reset_human_review,
    )
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2)

    if args.stdout:
        print(serialized)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized + "\n", encoding="utf-8")
    summary = manifest["summary"]
    checks = summary["auto_check"]
    print(f"검사 완료: total={summary['total']} pass={checks['pass']} warning={checks['warning']} fail={checks['fail']}")
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()
