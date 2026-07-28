import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.guide_alignment import STAGE_KEYS  # noqa: E402
from utils.stage_candidate_review import (  # noqa: E402
    candidate_indexes,
    finalize_stage_candidate_review,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "reference_data" / "swing_stage_ground_truth.json"
DEFAULT_CANDIDATE_ROOT = PROJECT_ROOT / "analysis_sessions" / "stage_candidates"


def parse_args():
    parser = argparse.ArgumentParser(
        description="단계별 후보 중 코치 정답 프레임을 선택해 스윙 정답 매니페스트를 갱신합니다."
    )
    parser.add_argument("video_id")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--note", default="")
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _fit(image, width=300, height=250):
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    scale = min(width / image.shape[1], (height - 35) / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
    )
    x = (width - resized.shape[1]) // 2
    y = 35 + (height - 35 - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def stage_canvas(candidate_dir, candidate_manifest, stage_key, selected_frame=None):
    panels = []
    candidates = candidate_manifest["stages"][stage_key]["candidates"]
    for index, candidate in enumerate(candidates):
        image = cv2.imread(str(candidate_dir / candidate["image"]))
        if image is None:
            raise OSError(f"후보 이미지를 읽지 못했습니다: {candidate['image']}")
        panel = _fit(image)
        is_selected = candidate["frame_index"] == selected_frame
        color = (80, 255, 120) if is_selected else (255, 255, 255)
        cv2.putText(
            panel,
            f"[{index + 1}] frame={candidate['frame_index']}",
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    canvas = np.hstack(panels)
    cv2.putText(
        canvas,
        f"{stage_key}: number=select  j/k=prev/next  q=save/quit",
        (10, canvas.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main():
    args = parse_args()
    manifest = load_json(args.manifest)
    candidate_dir = args.candidate_root / args.video_id
    candidate_manifest = load_json(candidate_dir / "candidate_manifest.json")
    existing_events = manifest["videos"][args.video_id].get("events", {})
    selections = {
        stage: frame
        for stage, frame in existing_events.items()
        if isinstance(frame, int)
        and frame in candidate_indexes(candidate_manifest, stage)
    }
    stage_index = 0
    window = "Golf Coach - Stage Candidate Review"
    try:
        while 0 <= stage_index < len(STAGE_KEYS):
            stage_key = STAGE_KEYS[stage_index]
            cv2.imshow(
                window,
                stage_canvas(
                    candidate_dir,
                    candidate_manifest,
                    stage_key,
                    selections.get(stage_key),
                ),
            )
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            if key == ord("j"):
                stage_index = max(0, stage_index - 1)
                continue
            if key == ord("k"):
                stage_index = min(len(STAGE_KEYS) - 1, stage_index + 1)
                continue
            candidate_index = key - ord("1")
            candidates = candidate_manifest["stages"][stage_key]["candidates"]
            if 0 <= candidate_index < len(candidates):
                selections[stage_key] = int(
                    candidates[candidate_index]["frame_index"]
                )
                stage_index = min(len(STAGE_KEYS) - 1, stage_index + 1)
    finally:
        cv2.destroyAllWindows()

    if set(selections) != set(STAGE_KEYS):
        missing = [stage for stage in STAGE_KEYS if stage not in selections]
        print(f"저장하지 않음: 선택하지 않은 단계 {', '.join(missing)}")
        return 1
    updated = finalize_stage_candidate_review(
        manifest,
        video_id=args.video_id,
        candidate_manifest=candidate_manifest,
        selections=selections,
        reviewed_by=args.reviewer,
        note=args.note,
        project_root=PROJECT_ROOT,
    )
    write_json(args.manifest, updated)
    print(f"{args.video_id} 검수 완료: {selections}")
    print(f"정답 매니페스트: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
