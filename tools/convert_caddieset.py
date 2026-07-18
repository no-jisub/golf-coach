import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "reference_data" / "caddieset" / "source" / "data" / "CaddieSet.csv"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "reference_data" / "caddieset" / "evaluation_profiles.json"
)

SOURCE_REPOSITORY = "https://github.com/damilab/CaddieSet"
SOURCE_COMMIT = "3c73d9d40580bb8a5a10711ad1fa10735a205ffe"
EXPECTED_SOURCE_SHA256 = "646459b081863e244d75efe6f09fad4d34750d2fed7fde44db654e7720a9a306"
SCHEMA = "golf-coach-caddieset-evaluation-v1"

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

METRIC_DEFINITIONS = {
    "LOWER-ANGLE": ("degree", "오른쪽 골반·무릎·발목이 이루는 각도"),
    "SHOULDER-ANGLE": ("degree", "수평선에 대한 어깨선 각도"),
    "SPINE-ANGLE": ("degree", "수평선에 대한 척추 각도"),
    "STANCE-RATIO": ("ratio", "어깨 너비 대비 스탠스 너비 비율"),
    "UPPER-TILT": ("ratio", "하체 대비 상체 기울기 비율"),
    "HEAD-LOC": ("ratio", "어드레스 대비 머리 위치 변화"),
    "HIP-LINE": ("ratio", "어드레스 대비 골반선 이동"),
    "HIP-ROTATION": ("degree", "어드레스 대비 골반 회전 각도"),
    "HIP-SHIFTED": ("ratio", "어드레스 대비 골반 위치 이동"),
    "LEFT-ARM-ANGLE": ("degree", "왼쪽 어깨·팔꿈치·손목이 이루는 각도"),
    "RIGHT-ARM-ANGLE": ("degree", "오른쪽 어깨·팔꿈치·손목이 이루는 각도"),
    "SHOULDER-LOC": ("ratio", "스탠스 너비 안에서 왼쪽 어깨의 상대 위치"),
    "HIP-ANGLE": ("degree", "골반 회전 각도"),
    "LEFT-LEG-ANGLE": ("degree", "왼쪽 골반·무릎·발목이 이루는 각도"),
    "RIGHT-DISTANCE": ("ratio", "오른쪽 팔꿈치와 몸통 사이 거리"),
    "RIGHT-LEG-ANGLE": ("degree", "오른쪽 골반·무릎·발목이 이루는 각도"),
    "HIP-HANGING-BACK": ("ratio", "스탠스 너비 대비 왼쪽 발목과 골반의 상대 거리"),
    "RIGHT-ARMPIT-ANGLE": ("degree", "오른쪽 팔꿈치·어깨·골반이 이루는 각도"),
    "SHOULDER-HANGING-BACK": ("ratio", "스탠스 너비 대비 왼쪽 발목과 어깨의 상대 거리"),
    "WEIGHT-SHIFT": ("degree", "왼쪽 발목과 왼쪽 골반을 잇는 선의 각도"),
    "FINISH-ANGLE": ("degree", "왼쪽 발목과 오른쪽 골반을 잇는 선의 각도"),
}

DEFAULT_PROFILE_SPECS = [
    ("FACEON", None),
    ("FACEON", "W1"),
    ("FACEON", "I7"),
    ("DTL", None),
    ("DTL", "W1"),
    ("DTL", "I7"),
]

FEATURE_COLUMN_PATTERN = re.compile(r"^([0-7])-(.+)$")
DIRECTION_ANGLE_LIMIT = 6.0
SPIN_AXIS_LIMIT = 10.0


def round_number(value):
    return round(float(value), 4)


def parse_number(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def percentile(values, percent):
    if not values:
        raise ValueError("백분위수를 계산할 값이 없습니다.")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_feature_column(column_name):
    match = FEATURE_COLUMN_PATTERN.match(column_name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def metric_key(metric_name):
    return metric_name.lower().replace("-", "_")


def is_reference_outcome(row):
    direction_angle = parse_number(row.get("DirectionAngle"))
    spin_axis = parse_number(row.get("SpinAxis"))
    if direction_angle is None or spin_axis is None:
        return False
    return (
        abs(direction_angle) <= DIRECTION_ANGLE_LIMIT
        and abs(spin_axis) <= SPIN_AXIS_LIMIT
    )


def summarize_metric(selected_rows, column_name):
    values = []
    values_by_golfer = defaultdict(list)
    for row in selected_rows:
        value = parse_number(row.get(column_name))
        if value is None:
            continue
        golfer_id = str(row.get("GolferId", "")).strip()
        values.append(value)
        if golfer_id:
            values_by_golfer[golfer_id].append(value)

    if not values:
        return None

    golfer_medians = [median(golfer_values) for golfer_values in values_by_golfer.values()]
    balanced_target = median(golfer_medians) if golfer_medians else median(values)
    summary = {
        "value_count": len(values),
        "coverage": round_number(len(values) / len(selected_rows)),
        "golfer_count": len(values_by_golfer),
        "target": round_number(balanced_target),
        "shot_median": round_number(median(values)),
        "observed_reference_range": [
            round_number(percentile(values, 10)),
            round_number(percentile(values, 90)),
        ],
        "observed_outer_range": [
            round_number(percentile(values, 5)),
            round_number(percentile(values, 95)),
        ],
    }
    if golfer_medians:
        summary["golfer_median_range"] = [
            round_number(percentile(golfer_medians, 10)),
            round_number(percentile(golfer_medians, 90)),
        ]
    return summary


def profile_id(view, club_type):
    club_label = club_type.lower() if club_type else "all"
    return f"{view.lower()}_{club_label}"


def build_profile(rows, feature_columns, view, club_type=None):
    candidate_rows = [
        row
        for row in rows
        if row.get("View") == view
        and (club_type is None or row.get("ClubType") == club_type)
    ]
    selected_rows = [row for row in candidate_rows if is_reference_outcome(row)]
    if not selected_rows:
        raise ValueError(f"참조 결과 조건을 만족하는 행이 없습니다: view={view}, club={club_type}")

    stages = {}
    for stage_index, stage_key in enumerate(STAGES):
        evaluation_items = {}
        for column_name, metric_name in feature_columns.get(stage_index, []):
            summary = summarize_metric(selected_rows, column_name)
            if summary is None:
                continue
            unit, description = METRIC_DEFINITIONS.get(
                metric_name,
                ("unknown", metric_name),
            )
            evaluation_items[metric_key(metric_name)] = {
                "source_column": column_name,
                "unit": unit,
                "description": description,
                **summary,
            }

        stages[stage_key] = {
            "source_stage_index": stage_index,
            "evaluation_items": evaluation_items,
        }

    return {
        "view": view,
        "club_type": club_type or "ALL",
        "candidate_shots": len(candidate_rows),
        "selected_reference_shots": len(selected_rows),
        "selected_golfers": len({row.get("GolferId") for row in selected_rows}),
        "stages": stages,
    }


def read_dataset(input_path):
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        raise ValueError("CaddieSet CSV에 데이터 행이 없습니다.")
    return rows, fieldnames


def discover_feature_columns(fieldnames):
    feature_columns = defaultdict(list)
    for column_name in fieldnames:
        parsed = split_feature_column(column_name)
        if parsed is None:
            continue
        stage_index, metric_name = parsed
        feature_columns[stage_index].append((column_name, metric_name))
    return feature_columns


def build_output(rows, fieldnames, input_path, profile_specs=DEFAULT_PROFILE_SPECS):
    feature_columns = discover_feature_columns(fieldnames)
    missing_stages = [index for index in range(8) if index not in feature_columns]
    if missing_stages:
        raise ValueError(f"CaddieSet 8단계 컬럼이 완전하지 않습니다: {missing_stages}")

    profiles = {}
    for view, club_type in profile_specs:
        identifier = profile_id(view, club_type)
        profiles[identifier] = build_profile(
            rows,
            feature_columns,
            view,
            club_type,
        )

    return {
        "schema": SCHEMA,
        "source": {
            "name": "CaddieSet",
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "license": "MIT",
            "csv_sha256": file_sha256(input_path),
            "total_rows": len(rows),
            "views": dict(sorted(Counter(row.get("View") for row in rows).items())),
            "club_types": dict(sorted(Counter(row.get("ClubType") for row in rows).items())),
            "golfers": len({row.get("GolferId") for row in rows}),
        },
        "conversion": {
            "current_app_profile": "faceon_all",
            "stage_mapping": {str(index): stage for index, stage in enumerate(STAGES)},
            "reference_outcome_filter": {
                "absolute_direction_angle_max_degrees": DIRECTION_ANGLE_LIMIT,
                "absolute_spin_axis_max_degrees": SPIN_AXIS_LIMIT,
                "basis": "CaddieSet paper straight-shot classification thresholds",
            },
            "aggregation": {
                "target": "median of each golfer's median (equal golfer weighting)",
                "observed_reference_range": "10th to 90th percentile of selected shots",
                "observed_outer_range": "5th to 95th percentile of selected shots",
                "warning": "Observed ranges are evidence references, not validated pass/fail limits.",
            },
        },
        "profiles": profiles,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="CaddieSet을 현재 앱의 8단계 평가 프로필 JSON으로 변환합니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--allow-unverified-source",
        action="store_true",
        help="고정된 원본 SHA-256과 다른 CSV도 변환합니다.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.exists():
        raise SystemExit(
            f"CaddieSet CSV가 없습니다: {input_path}\n"
            "먼저 py -3.12 tools\\download_caddieset.py 를 실행하세요."
        )

    actual_sha256 = file_sha256(input_path)
    if actual_sha256 != EXPECTED_SOURCE_SHA256 and not args.allow_unverified_source:
        raise SystemExit(
            "검증된 CaddieSet 원본과 SHA-256이 다릅니다. "
            "다른 버전을 의도했다면 --allow-unverified-source를 사용하세요."
        )

    rows, fieldnames = read_dataset(input_path)
    output = build_output(rows, fieldnames, input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"원본 행: {len(rows)}")
    for identifier, profile in output["profiles"].items():
        print(
            f"[{identifier}] 후보 {profile['candidate_shots']} / "
            f"참조 {profile['selected_reference_shots']} / "
            f"골퍼 {profile['selected_golfers']}"
        )
    print(f"저장: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
