from pathlib import Path

import cv2
import numpy as np

from utils.guide_alignment import STAGE_KEYS


PANEL_WIDTH = 720
PANEL_HEIGHT = 300
SHEET_COLUMNS = 2
SHEET_ROWS = 4
THUMBNAIL_WIDTH = 226
THUMBNAIL_HEIGHT = 230
STAGE_LABELS = {
    "address": "1. Address",
    "takeaway": "2. Takeaway",
    "backswing": "3. Backswing",
    "top": "4. Top",
    "downswing": "5. Downswing",
    "impact": "6. Impact",
    "follow_through": "7. Follow-through",
    "finish": "8. Finish",
}


def candidate_frame_indexes(auto_frame, total_frames, offset_frames):
    maximum = max(0, int(total_frames) - 1)
    return [
        max(0, min(maximum, int(auto_frame) - int(offset_frames))),
        max(0, min(maximum, int(auto_frame))),
        max(0, min(maximum, int(auto_frame) + int(offset_frames))),
    ]


def _fit_frame(frame, width, height):
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized_width = max(1, round(frame.shape[1] * scale))
    resized_height = max(1, round(frame.shape[0] * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def _read_frame(capture, frame_index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    success, frame = capture.read()
    if not success:
        raise RuntimeError(f"검수 후보 프레임을 읽지 못했습니다: {frame_index}")
    return frame


def build_stage_contact_sheet(
    video_path,
    detected_events,
    ground_truth_entry,
    *,
    candidate_offset_sec=0.15,
):
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    offset_frames = max(1, round(fps * candidate_offset_sec))
    panels = []
    candidate_metadata = {}
    review_status = ground_truth_entry.get("review_status", "pending")

    try:
        for stage_key in STAGE_KEYS:
            panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 18, dtype=np.uint8)
            auto_frame = int(detected_events[stage_key]["frame_index"])
            ground_truth = ground_truth_entry.get("events", {}).get(stage_key)
            candidates = candidate_frame_indexes(auto_frame, total_frames, offset_frames)
            candidate_metadata[stage_key] = candidates
            ground_truth_text = (
                f"GT {ground_truth}  delta {auto_frame - ground_truth:+d}"
                if ground_truth is not None
                else f"GT {review_status.upper()}"
            )
            cv2.putText(
                panel,
                f"{STAGE_LABELS[stage_key]} | AUTO {auto_frame} | {ground_truth_text}",
                (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )

            for candidate_index, frame_index in enumerate(candidates):
                frame = _fit_frame(
                    _read_frame(capture, frame_index),
                    THUMBNAIL_WIDTH,
                    THUMBNAIL_HEIGHT,
                )
                left = 8 + candidate_index * (THUMBNAIL_WIDTH + 7)
                top = 39
                panel[top : top + THUMBNAIL_HEIGHT, left : left + THUMBNAIL_WIDTH] = frame
                is_auto = candidate_index == 1
                is_ground_truth = ground_truth is not None and frame_index == ground_truth
                color = (70, 220, 90) if is_ground_truth else (70, 210, 255) if is_auto else (150, 150, 150)
                cv2.rectangle(
                    panel,
                    (left, top),
                    (left + THUMBNAIL_WIDTH - 1, top + THUMBNAIL_HEIGHT - 1),
                    color,
                    3 if is_auto or is_ground_truth else 1,
                )
                labels = ("BEFORE", "AUTO", "AFTER")
                cv2.putText(
                    panel,
                    f"{labels[candidate_index]} {frame_index}",
                    (left + 7, 290),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            panels.append(panel)
    finally:
        capture.release()

    rows = []
    for row_index in range(SHEET_ROWS):
        start = row_index * SHEET_COLUMNS
        rows.append(np.hstack(panels[start : start + SHEET_COLUMNS]))
    return np.vstack(rows), {
        "fps": round(fps, 6),
        "total_frames": total_frames,
        "candidate_offset_sec": candidate_offset_sec,
        "candidate_offset_frames": offset_frames,
        "candidates": candidate_metadata,
    }


def generate_audit_contact_sheets(audit, manifest, *, project_root, output_root):
    output_root = Path(output_root)
    generated = 0
    failed = 0
    for video_id, result in audit.get("videos", {}).items():
        if result.get("status") != "ok":
            continue
        source_path = Path(result["source"])
        if not source_path.is_absolute():
            source_path = Path(project_root) / source_path
        output_path = output_root / video_id / "stage_contact_sheet.jpg"
        try:
            sheet, metadata = build_stage_contact_sheet(
                source_path,
                result["stage_detection"]["events"],
                manifest["videos"][video_id],
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), sheet):
                raise OSError(f"검수 시트를 저장하지 못했습니다: {output_path}")
            result.setdefault("artifacts", {})["contact_sheet"] = str(output_path)
            result["contact_sheet"] = metadata
            generated += 1
        except Exception as error:
            result["contact_sheet_error"] = f"{type(error).__name__}: {error}"
            failed += 1
    audit["summary"]["contact_sheet_count"] = generated
    audit["summary"]["contact_sheet_failed_count"] = failed
    return audit
