"""자동 단계 검출 주변의 후보 프레임과 검수용 접촉 시트를 생성합니다."""

import json
from pathlib import Path

import cv2
import numpy as np

from utils.guide_alignment import STAGE_KEYS


CANDIDATE_SCHEMA = "golf-coach-stage-candidates-v1"
STAGE_LABELS = {
    "address": "Address",
    "takeaway": "Takeaway",
    "backswing": "Backswing",
    "top": "Top",
    "downswing": "Downswing",
    "impact": "Impact",
    "follow_through": "Follow-through",
    "finish": "Finish",
}


def _read_frame(capture, frame_index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    if not ok:
        raise OSError(f"영상 프레임을 읽지 못했습니다: {frame_index}")
    return frame


def _candidate_offsets(radius_frames, candidate_count):
    if radius_frames < 0:
        raise ValueError("radius_frames는 0 이상이어야 합니다.")
    if candidate_count < 1 or candidate_count % 2 == 0:
        raise ValueError("candidate_count는 1 이상의 홀수여야 합니다.")
    if candidate_count == 1:
        return [0]
    return [
        round(-radius_frames + (2 * radius_frames * index / (candidate_count - 1)))
        for index in range(candidate_count)
    ]


def _fit(frame, width, height):
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(
        frame,
        (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))),
    )
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def extract_stage_candidates(
    video_path,
    stage_events,
    output_dir,
    *,
    radius_frames=8,
    candidate_count=5,
):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    stages = stage_events.get("stages", {})
    missing = [stage for stage in STAGE_KEYS if stage not in stages]
    if missing:
        raise ValueError(f"단계 이벤트가 부족합니다: {', '.join(missing)}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"영상을 열 수 없습니다: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if total_frames <= 0:
        capture.release()
        raise ValueError("영상의 전체 프레임 수를 읽을 수 없습니다.")

    offsets = _candidate_offsets(radius_frames, candidate_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_stages = {}
    sheet_rows = []
    try:
        for stage_index, stage_key in enumerate(STAGE_KEYS, start=1):
            automatic_frame = int(stages[stage_key]["frame_index"])
            indexes = [
                max(0, min(total_frames - 1, automatic_frame + offset))
                for offset in offsets
            ]
            candidates = []
            panels = []
            for rank, frame_index in enumerate(indexes):
                frame = _read_frame(capture, frame_index)
                filename = (
                    f"{stage_index:02d}_{stage_key}_"
                    f"{rank + 1:02d}_frame{frame_index:06d}.jpg"
                )
                path = output_dir / filename
                if not cv2.imwrite(str(path), frame):
                    raise OSError(f"후보 이미지를 저장하지 못했습니다: {path}")
                offset = frame_index - automatic_frame
                candidates.append(
                    {
                        "rank": rank + 1,
                        "frame_index": frame_index,
                        "offset_from_automatic": offset,
                        "timestamp_ms": (
                            round(frame_index / fps * 1000.0, 3) if fps > 0 else None
                        ),
                        "image": filename,
                        "is_automatic": frame_index == automatic_frame,
                    }
                )
                panel = _fit(frame, 230, 190)
                cv2.putText(
                    panel,
                    f"{STAGE_LABELS[stage_key]} f={frame_index} ({offset:+d})",
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.47,
                    (50, 255, 120) if offset == 0 else (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                panels.append(panel)
            manifest_stages[stage_key] = {
                "automatic_frame": automatic_frame,
                "candidates": candidates,
            }
            sheet_rows.append(np.hstack(panels))
    finally:
        capture.release()

    sheet = np.vstack(sheet_rows)
    sheet_path = output_dir / "stage_candidates.jpg"
    if not cv2.imwrite(str(sheet_path), sheet):
        raise OSError(f"후보 접촉 시트를 저장하지 못했습니다: {sheet_path}")

    manifest = {
        "schema": CANDIDATE_SCHEMA,
        "source_video": str(video_path),
        "video": {
            "fps": fps,
            "total_frames": total_frames,
        },
        "settings": {
            "radius_frames": radius_frames,
            "candidate_count": candidate_count,
            "offsets": offsets,
        },
        "stages": manifest_stages,
        "contact_sheet": sheet_path.name,
    }
    manifest_path = output_dir / "candidate_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
