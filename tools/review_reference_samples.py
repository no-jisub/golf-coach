import argparse
import json
import textwrap
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "reference_data" / "review_manifest.json"
DEBUG_OVERLAY_DIR = PROJECT_ROOT / "reference_data" / "debug_overlay"
DEBUG_SHAFT_OVERLAY_DIR = PROJECT_ROOT / "reference_data" / "debug_shaft_overlay"
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
AUTO_STATUSES = {"pass", "warning", "fail"}
HUMAN_STATUSES = {"pending", "accepted", "rejected"}


def load_manifest(manifest_path=DEFAULT_MANIFEST_PATH):
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"검수 manifest가 없습니다: {manifest_path}\n"
            "먼저 tools\\audit_reference_samples.py를 실행해주세요."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != REVIEW_SCHEMA:
        raise ValueError(f"지원하지 않는 manifest 스키마입니다: {manifest.get('schema')}")
    if not isinstance(manifest.get("samples"), dict):
        raise ValueError("manifest에 samples 객체가 없습니다.")
    return manifest


def save_manifest(manifest, manifest_path=DEFAULT_MANIFEST_PATH):
    """검수 중 중단되어도 기존 파일이 손상되지 않도록 임시 파일을 거쳐 저장합니다."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(manifest_path)


def filter_samples(manifest, stage=None, auto_status=None, review_status=None):
    rows = []
    for key, sample in manifest["samples"].items():
        if stage is not None and sample.get("stage") != stage:
            continue
        if auto_status is not None and sample.get("auto_check", {}).get("status") != auto_status:
            continue
        if review_status is not None and sample.get("human_review", {}).get("status") != review_status:
            continue
        rows.append((key, sample))
    return sorted(rows, key=lambda item: (STAGES.index(item[1].get("stage")), item[0]))


def apply_decision(sample, status, override_auto_fail=False, note=None):
    if status not in HUMAN_STATUSES:
        raise ValueError(f"지원하지 않는 사람 검수 상태입니다: {status}")

    auto_status = sample.get("auto_check", {}).get("status")
    if status == "accepted" and auto_status == "fail" and not override_auto_fail:
        raise ValueError("자동 fail 샘플은 override_auto_fail=True로만 승인할 수 있습니다.")

    human_review = sample.setdefault("human_review", {})
    human_review["status"] = status
    human_review["override_auto_fail"] = bool(status == "accepted" and override_auto_fail)
    if note is not None:
        human_review["note"] = note
    else:
        human_review.setdefault("note", "")
    return human_review


def update_review_summary(manifest):
    counts = {status: 0 for status in HUMAN_STATUSES}
    override_count = 0
    for sample in manifest["samples"].values():
        human_review = sample.get("human_review", {})
        status = human_review.get("status", "pending")
        if status in counts:
            counts[status] += 1
        if human_review.get("override_auto_fail"):
            override_count += 1

    manifest.setdefault("summary", {})["human_review"] = {
        "pending": counts["pending"],
        "accepted": counts["accepted"],
        "rejected": counts["rejected"],
        "auto_fail_overrides": override_count,
    }
    return manifest["summary"]["human_review"]


def resolve_project_path(path_text):
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_sample_image_paths(sample):
    image_path = resolve_project_path(sample.get("image"))
    if image_path is None:
        return []

    stage = sample.get("stage")
    stem = image_path.stem
    candidates = [
        DEBUG_OVERLAY_DIR / stage / f"{stem}_overlay.jpg",
        DEBUG_SHAFT_OVERLAY_DIR / stage / f"{stem}_shaft_overlay.jpg",
        image_path,
    ]
    return [path for path in candidates if path.exists()]


def read_display_images(sample):
    loaded = []
    for image_path in get_sample_image_paths(sample):
        image = cv2.imread(str(image_path))
        if image is not None:
            loaded.append((image_path.name, image))
    return loaded


def resize_to_height(image, target_height):
    scale = target_height / image.shape[0]
    width = max(1, round(image.shape[1] * scale))
    return cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA)


def put_lines(canvas, lines, x, y, color=(235, 235, 235), scale=0.55, line_height=24):
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
        y += line_height
    return y


def build_review_frame(sample_key, sample, current_index, total_count, message=""):
    images = read_display_images(sample)
    image_height = 610
    if images:
        resized = [resize_to_height(image, image_height) for _, image in images[:3]]
        image_row = np.hstack(resized)
    else:
        image_row = np.zeros((image_height, 1000, 3), dtype=np.uint8)
        cv2.putText(
            image_row,
            "No review image found",
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

    max_width = 1800
    if image_row.shape[1] > max_width:
        scale = max_width / image_row.shape[1]
        image_row = cv2.resize(
            image_row,
            (max_width, round(image_row.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )

    panel_height = 255
    canvas = np.zeros((image_row.shape[0] + panel_height, image_row.shape[1], 3), dtype=np.uint8)
    canvas[: image_row.shape[0]] = image_row
    panel_y = image_row.shape[0]

    auto_check = sample.get("auto_check", {})
    human_review = sample.get("human_review", {})
    auto_status = auto_check.get("status", "unknown")
    human_status = human_review.get("status", "pending")
    status_colors = {
        "pass": (80, 220, 80),
        "warning": (0, 210, 255),
        "fail": (80, 80, 255),
    }

    header = (
        f"[{current_index + 1}/{total_count}] {sample.get('stage')} | "
        f"AUTO={auto_status} | REVIEW={human_status} | override={human_review.get('override_auto_fail', False)}"
    )
    cv2.putText(
        canvas,
        header,
        (18, panel_y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_colors.get(auto_status, (255, 255, 255)),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        sample_key,
        (18, panel_y + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    reason_codes = [reason.get("code", "unknown") for reason in auto_check.get("reasons", [])]
    reason_text = "Reasons: " + (", ".join(reason_codes) if reason_codes else "none")
    reason_lines = textwrap.wrap(reason_text, width=max(55, image_row.shape[1] // 13))[:3]
    y = put_lines(canvas, reason_lines, 18, panel_y + 82, color=(230, 230, 230))

    metrics = auto_check.get("metrics", {})
    metric_keys = [
        "min_visibility",
        "shaft_score",
        "shaft_grip_endpoint_distance",
        "stage_pose_deviation",
        "stage_outlier_threshold",
    ]
    metric_text = " | ".join(
        f"{key}={metrics[key]}" for key in metric_keys if key in metrics
    ) or "No selected metrics"
    y = put_lines(canvas, textwrap.wrap(metric_text, width=max(55, image_row.shape[1] // 13))[:2], 18, y)

    help_text = "A accept | O override auto-fail | R reject | P pending | LEFT/J previous | RIGHT/K next | Q quit"
    cv2.putText(
        canvas,
        help_text,
        (18, panel_y + panel_height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if message:
        cv2.putText(
            canvas,
            message,
            (18, panel_y + panel_height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def print_sample_list(rows):
    for key, sample in rows:
        auto_status = sample.get("auto_check", {}).get("status", "unknown")
        human_status = sample.get("human_review", {}).get("status", "pending")
        reasons = ",".join(reason.get("code", "unknown") for reason in sample.get("auto_check", {}).get("reasons", []))
        print(f"{sample.get('stage'):15} auto={auto_status:7} review={human_status:8} {key} {reasons}")


def review_interactively(manifest, manifest_path, rows, start_key=None):
    if not rows:
        print("조건에 맞는 검수 샘플이 없습니다.")
        return 0

    index = 0
    if start_key:
        for candidate_index, (key, _) in enumerate(rows):
            if start_key in key:
                index = candidate_index
                break

    window_name = "Golf Coach Reference Review"
    message = ""
    while True:
        sample_key, sample = rows[index]
        frame = build_review_frame(sample_key, sample, index, len(rows), message)
        cv2.imshow(window_name, frame)
        key = cv2.waitKeyEx(0)
        message = ""

        if key in {ord("q"), ord("Q"), 27}:
            break
        if key in {2424832, ord("j"), ord("J")}:
            index = max(0, index - 1)
            continue
        if key in {2555904, ord("k"), ord("K")}:
            index = min(len(rows) - 1, index + 1)
            continue

        try:
            if key in {ord("a"), ord("A")}:
                apply_decision(sample, "accepted")
                message = "Saved: accepted"
            elif key in {ord("o"), ord("O")}:
                apply_decision(sample, "accepted", override_auto_fail=True)
                message = "Saved: accepted with auto-fail override"
            elif key in {ord("r"), ord("R")}:
                apply_decision(sample, "rejected")
                message = "Saved: rejected"
            elif key in {ord("p"), ord("P")}:
                apply_decision(sample, "pending")
                message = "Saved: pending"
            else:
                continue
        except ValueError as error:
            message = str(error)
            continue

        update_review_summary(manifest)
        save_manifest(manifest, manifest_path)
        if index < len(rows) - 1:
            index += 1

    cv2.destroyAllWindows()
    summary = update_review_summary(manifest)
    save_manifest(manifest, manifest_path)
    print(f"검수 현황: {summary}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="참조 이미지와 오버레이를 보면서 검수 상태를 저장합니다.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--auto-status", choices=sorted(AUTO_STATUSES))
    parser.add_argument("--review-status", choices=sorted(HUMAN_STATUSES))
    parser.add_argument("--start", help="파일명 일부가 일치하는 샘플부터 시작합니다.")
    parser.add_argument("--list", action="store_true", help="GUI를 열지 않고 대상 목록만 출력합니다.")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    rows = filter_samples(
        manifest,
        stage=args.stage,
        auto_status=args.auto_status,
        review_status=args.review_status,
    )
    if args.list:
        print_sample_list(rows)
        print(f"count={len(rows)}")
        return 0
    return review_interactively(manifest, manifest_path, rows, start_key=args.start)


if __name__ == "__main__":
    raise SystemExit(main())
