"""자동 단계 프레임과 코치 검수 프레임을 나란히 비교하는 이미지를 만듭니다."""

from pathlib import Path

import cv2
import numpy as np

from utils.guide_alignment import STAGE_KEYS


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


def _panel(frame, width, height, title, color):
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    available_height = height - 34
    scale = min(width / frame.shape[1], available_height / frame.shape[0])
    resized = cv2.resize(
        frame,
        (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))),
    )
    x = (width - resized.shape[1]) // 2
    y = 34 + (available_height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.putText(
        canvas,
        title,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        1,
        cv2.LINE_AA,
    )
    return canvas


def build_stage_comparison_sheet(
    video_path,
    reviewed_events,
    automatic_events,
    output_path,
):
    """8단계의 automatic/reviewed 프레임을 2열 접촉 시트로 저장합니다."""
    missing_reviewed = [
        stage for stage in STAGE_KEYS if not isinstance(reviewed_events.get(stage), int)
    ]
    automatic_stages = automatic_events.get("stages", {})
    missing_automatic = [
        stage for stage in STAGE_KEYS if stage not in automatic_stages
    ]
    if missing_reviewed or missing_automatic:
        raise ValueError(
            "비교에 필요한 단계가 부족합니다: "
            + ", ".join(missing_reviewed + missing_automatic)
        )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"영상을 열 수 없습니다: {video_path}")
    rows = []
    comparisons = {}
    try:
        for stage_key in STAGE_KEYS:
            auto_frame = int(automatic_stages[stage_key]["frame_index"])
            reviewed_frame = int(reviewed_events[stage_key])
            auto_image = _read_frame(capture, auto_frame)
            reviewed_image = _read_frame(capture, reviewed_frame)
            delta = auto_frame - reviewed_frame
            rows.append(
                np.hstack(
                    (
                        _panel(
                            auto_image,
                            480,
                            260,
                            f"{STAGE_LABELS[stage_key]} AUTO f={auto_frame}",
                            (80, 210, 255),
                        ),
                        _panel(
                            reviewed_image,
                            480,
                            260,
                            f"REVIEWED f={reviewed_frame} delta={delta:+d}",
                            (80, 255, 120),
                        ),
                    )
                )
            )
            comparisons[stage_key] = {
                "automatic_frame": auto_frame,
                "reviewed_frame": reviewed_frame,
                "signed_error_frames": delta,
                "absolute_error_frames": abs(delta),
            }
    finally:
        capture.release()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), np.vstack(rows)):
        raise OSError(f"비교 이미지를 저장하지 못했습니다: {output_path}")
    return {
        "source_video": str(video_path),
        "output": str(output_path),
        "comparisons": comparisons,
    }
