import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.sample_review import (  # noqa: E402
    apply_coach_review,
    build_review_manifest,
    load_json,
    write_json,
)


DEFAULT_SAMPLE_ROOTS = [
    PROJECT_ROOT / "reference_data" / "runtime_samples",
    PROJECT_ROOT / "reference_data" / "webcam_dataset" / "captures",
]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "reference_data"
    / "webcam_dataset"
    / "reviews"
    / "coach_reviews.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="웹캠 샘플을 우선순위 순으로 빠르게 good/bad/uncertain 검수합니다."
    )
    parser.add_argument("--sample-root", action="append", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reviewer", default="coach")
    parser.add_argument("--stage")
    parser.add_argument("--include-reviewed", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def _fit(image, width, height):
    if image is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def build_review_canvas(item, index, total):
    raw = cv2.imread(item.get("artifacts", {}).get("raw_frame", ""))
    overlay = cv2.imread(item.get("artifacts", {}).get("overlay_frame", ""))
    canvas = np.hstack((_fit(raw, 560, 630), _fit(overlay, 560, 630)))
    lines = [
        f"{index + 1}/{total}  {item['sample_id']}  stage={item['stage_key']}",
        (
            f"preliminary={item['preliminary']['label']} "
            f"({item['preliminary']['source']}, {item['preliminary']['confidence']:.2f})"
        ),
        (
            f"priority={item['priority']['score']} "
            f"{','.join(item['priority']['reasons']) or '-'}"
        ),
        (
            f"coach={item['coach_review']['label']}  "
            "keys: [g] good [b] bad [u] uncertain [p] pending [j/k] prev/next [q] quit"
        ),
    ]
    for line_index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (18, 26 + line_index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def main():
    args = parse_args()
    roots = args.sample_root or DEFAULT_SAMPLE_ROOTS
    existing = load_json(args.manifest) if args.manifest.exists() else None
    manifest = build_review_manifest(roots, existing)
    write_json(args.manifest, manifest)

    items = [
        manifest["samples"][sample_id]
        for sample_id in manifest["review_order"]
        if (not args.stage or manifest["samples"][sample_id]["stage_key"] == args.stage)
        and (
            args.include_reviewed
            or manifest["samples"][sample_id]["coach_review"]["label"] == "pending"
        )
    ]
    if args.list:
        for item in items:
            print(
                f"{item['sample_id']} {item['stage_key']:14s} "
                f"priority={item['priority']['score']:3d} "
                f"preliminary={item['preliminary']['label']}"
            )
        print(f"검수 대상: {len(items)}")
        return 0
    if not items:
        print("검수할 샘플이 없습니다.")
        return 0

    index = 0
    key_labels = {
        ord("g"): "good",
        ord("b"): "bad",
        ord("u"): "uncertain",
        ord("p"): "pending",
    }
    window = "Golf Coach - Webcam Sample Review"
    try:
        while 0 <= index < len(items):
            item = items[index]
            cv2.imshow(window, build_review_canvas(item, index, len(items)))
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            if key == ord("j"):
                index = max(0, index - 1)
                continue
            if key == ord("k"):
                index = min(len(items) - 1, index + 1)
                continue
            if key in key_labels:
                apply_coach_review(
                    manifest,
                    item["sample_id"],
                    key_labels[key],
                    reviewed_by=args.reviewer,
                )
                write_json(args.manifest, manifest)
                index = min(index + 1, len(items) - 1)
    finally:
        cv2.destroyAllWindows()

    print(
        f"검수완료={manifest['summary']['reviewed_count']} "
        f"대기={manifest['summary']['pending_count']} "
        f"판단보류={manifest['summary']['uncertain_count']}"
    )
    print(f"검수 매니페스트: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
