"""Utilities for rotating a looped video onto a linear timeline."""

from __future__ import annotations

from pathlib import Path

import cv2


MAPPING_SCHEMA = "golf-coach-loop-video-frame-mapping-v1"


def frame_from_seconds(seconds, fps):
    """Convert a timestamp to the nearest zero-based frame index."""
    seconds = float(seconds)
    fps = float(fps)
    if seconds < 0:
        raise ValueError("split_sec must be zero or greater")
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    return int(round(seconds * fps))


def map_frame_index(frame_index, total_frames, split_frame):
    """Map a source frame to its index after rotating at ``split_frame``."""
    frame_index = int(frame_index)
    total_frames = int(total_frames)
    split_frame = int(split_frame)
    if total_frames < 1:
        raise ValueError("total_frames must be at least 1")
    if not 0 <= split_frame < total_frames:
        raise ValueError("split_frame must be inside the source video")
    if not 0 <= frame_index < total_frames:
        raise ValueError("frame_index must be inside the source video")
    return (frame_index - split_frame) % total_frames


def map_ground_truth_events(events, total_frames, split_frame):
    """Return stage events mapped onto a rotated timeline.

    ``None`` values are retained so this can also be used with pending manifests.
    The input mapping is never mutated.
    """
    mapped = {}
    for stage, frame_index in events.items():
        mapped[stage] = (
            None
            if frame_index is None
            else map_frame_index(frame_index, total_frames, split_frame)
        )
    return mapped


def build_frame_mapping(total_frames, split_frame):
    """Build a serializable source-to-normalized frame mapping."""
    total_frames = int(total_frames)
    return {
        str(frame_index): map_frame_index(frame_index, total_frames, split_frame)
        for frame_index in range(total_frames)
    }


def _codec_for_path(path):
    return "MJPG" if Path(path).suffix.lower() == ".avi" else "mp4v"


def normalize_loop_video(input_path, output_path, split_frame, *, progress=False):
    """Rotate a video so ``split_frame`` becomes frame zero.

    Frames are written in the order ``split_frame..end, 0..split_frame-1``.
    The source is opened read-only and the destination must not already exist.
    """
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if input_path == output_path:
        raise ValueError("output_path must be different from input_path")
    if not input_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {input_path}")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    writer = None
    written_frames = 0
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or width <= 0 or height <= 0 or total_frames <= 0:
            raise ValueError("Video metadata is invalid")

        split_frame = int(split_frame)
        if not 0 <= split_frame < total_frames:
            raise ValueError(
                f"split_frame must be between 0 and {total_frames - 1}: {split_frame}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*_codec_for_path(output_path)),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise ValueError(f"Could not create output video: {output_path}")

        source_ranges = ((split_frame, total_frames), (0, split_frame))
        for range_start, range_end in source_ranges:
            if range_start >= range_end:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, range_start)
            for _ in range(range_start, range_end):
                success, frame = capture.read()
                if not success:
                    raise ValueError(
                        f"Video decoding stopped after {written_frames} frames"
                    )
                if frame.shape[1] != width or frame.shape[0] != height:
                    raise ValueError("Video resolution changed while decoding")
                writer.write(frame)
                written_frames += 1
                if progress and written_frames % 100 == 0:
                    print(f"[NORMALIZE] {written_frames}/{total_frames}")

        if written_frames != total_frames:
            raise ValueError(
                f"Expected {total_frames} frames but wrote {written_frames}"
            )
    except Exception:
        if writer is not None:
            writer.release()
            writer = None
        if output_path.exists():
            output_path.unlink()
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    return {
        "schema": MAPPING_SCHEMA,
        "source_video": str(input_path),
        "normalized_video": str(output_path),
        "video": {
            "fps": round(fps, 6),
            "width": width,
            "height": height,
            "total_frames": total_frames,
        },
        "split": {
            "source_frame": split_frame,
            "source_sec": round(split_frame / fps, 6),
        },
        "frame_mapping": build_frame_mapping(total_frames, split_frame),
    }
