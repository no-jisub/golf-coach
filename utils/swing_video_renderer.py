from bisect import bisect_left
from pathlib import Path
from types import SimpleNamespace

import cv2

from utils.guide_alignment import STAGE_KEYS
from utils.guide_skeleton import create_calibration_profile, draw_guide_skeleton
from utils.guide_tolerance import get_stage_tolerance_regions
from utils.pose_drawer import draw_pose_landmarks


USER_DISPLAY_LANDMARKS = {0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}


def deserialize_landmarks(frame_record):
    landmarks = []
    for point in frame_record.get("landmarks", []):
        landmarks.append(
            SimpleNamespace(
                x=float(point["x"]),
                y=float(point["y"]),
                z=float(point.get("z", 0.0)),
                visibility=float(point.get("visibility", 1.0)),
                presence=float(point.get("presence", 1.0)),
            )
        )
    return landmarks


def stage_boundaries(stage_detection):
    frames = [
        int(stage_detection["stages"][stage_key]["frame_index"])
        for stage_key in STAGE_KEYS
    ]
    return [round((left + right) / 2.0) for left, right in zip(frames, frames[1:])]


def stage_for_frame(frame_index, stage_detection):
    position = bisect_left(stage_boundaries(stage_detection), int(frame_index))
    return STAGE_KEYS[min(position, len(STAGE_KEYS) - 1)]


def _nearest_record(frame_index, records, record_indexes, max_gap):
    if not records:
        return None
    position = bisect_left(record_indexes, frame_index)
    candidates = []
    if position < len(records):
        candidates.append(records[position])
    if position > 0:
        candidates.append(records[position - 1])
    nearest = min(candidates, key=lambda record: abs(record["frame_index"] - frame_index))
    if abs(nearest["frame_index"] - frame_index) > max_gap:
        return None
    return nearest


def _video_writer(output_path, fps, width, height):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".avi":
        codecs = ("MJPG", "XVID")
    else:
        codecs = ("mp4v", "avc1")
    for codec in codecs:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise OSError(f"결과 영상 writer를 열 수 없습니다: {output_path}")


def _draw_video_labels(frame, stage_key, stage_analysis, frame_index):
    status = stage_analysis["stages"][stage_key]["evaluation"]["status"]
    status_colors = {
        "pass": (70, 210, 90),
        "warning": (30, 170, 255),
        "unavailable": (160, 160, 160),
    }
    color = status_colors.get(status, (255, 255, 255))
    panel_right = min(frame.shape[1] - 8, 470)
    cv2.rectangle(frame, (8, 8), (panel_right, 86), (15, 15, 15), -1)
    cv2.putText(
        frame,
        f"Stage: {stage_key} | {status.upper()}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Frame {frame_index} | Guide: blue | User: magenta",
        (18, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Green zones: CaddieSet tolerance",
        (18, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (120, 235, 140),
        1,
        cv2.LINE_AA,
    )


def render_annotated_video(
    video_path,
    landmark_payload,
    stage_detection,
    stage_analysis,
    output_path,
):
    """Render guide, tolerance, and detected user pose on the source timeline."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer, codec = _video_writer(output_path, fps, width, height)

    records = sorted(
        (
            frame
            for frame in landmark_payload.get("frames", [])
            if frame.get("detected") and frame.get("landmarks")
        ),
        key=lambda frame: frame["frame_index"],
    )
    record_indexes = [record["frame_index"] for record in records]
    sample_step = max(1, int(landmark_payload.get("sampling", {}).get("sample_step", 1)))
    address_index = stage_detection["stages"]["address"]["frame_index"]
    address_record = _nearest_record(address_index, records, record_indexes, sample_step)
    calibration_profile = None
    if address_record is not None:
        calibration_profile = create_calibration_profile(
            [deserialize_landmarks(address_record)],
            width,
            height,
        )

    written = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            stage_key = stage_for_frame(written, stage_detection)
            record = _nearest_record(written, records, record_indexes, sample_step)
            landmarks = deserialize_landmarks(record) if record is not None else None
            draw_guide_skeleton(
                frame,
                stage_key,
                user_landmarks=landmarks,
                calibration_profile=calibration_profile,
                tolerance_regions=get_stage_tolerance_regions(stage_key),
            )
            if landmarks:
                draw_pose_landmarks(
                    frame,
                    landmarks,
                    line_color=(245, 245, 245),
                    point_color=(255, 80, 255),
                    point_indexes=USER_DISPLAY_LANDMARKS,
                )
            _draw_video_labels(frame, stage_key, stage_analysis, written)
            writer.write(frame)
            written += 1
    finally:
        capture.release()
        writer.release()

    if written == 0:
        raise ValueError(f"결과 영상에 기록된 프레임이 없습니다: {video_path}")
    return {
        "path": str(output_path),
        "codec": codec,
        "fps": round(fps, 6),
        "width": width,
        "height": height,
        "frame_count": written,
        "audio_preserved": False,
    }
