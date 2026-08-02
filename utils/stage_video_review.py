"""프로 영상 전체 타임라인 검수를 위한 프레임 탐색과 진행 상태를 관리합니다."""

import json
from datetime import datetime
from pathlib import Path

from utils.guide_alignment import STAGE_KEYS


REVIEW_PROGRESS_SCHEMA = "golf-coach-stage-review-progress-v1"


class FrameNavigator:
    def __init__(self, total_frames, *, fps=0.0, initial_frame=0):
        if int(total_frames) <= 0:
            raise ValueError("total_frames는 0보다 커야 합니다.")
        self.total_frames = int(total_frames)
        self.fps = float(fps)
        self.current_frame = 0
        self.seek(initial_frame)

    def clamp(self, frame_index):
        return max(0, min(self.total_frames - 1, int(frame_index)))

    def seek(self, frame_index):
        self.current_frame = self.clamp(frame_index)
        return self.current_frame

    def step(self, delta):
        return self.seek(self.current_frame + int(delta))

    @property
    def timestamp_ms(self):
        if self.fps <= 0:
            return None
        return round(self.current_frame / self.fps * 1000.0, 3)


def build_review_progress(
    *,
    video_id,
    source_video,
    selections,
    current_stage_index,
    current_frame,
    updated_at=None,
):
    return {
        "schema": REVIEW_PROGRESS_SCHEMA,
        "video_id": video_id,
        "source_video": str(source_video),
        "updated_at": (updated_at or datetime.now().astimezone()).isoformat(),
        "current_stage_index": max(
            0, min(len(STAGE_KEYS) - 1, int(current_stage_index))
        ),
        "current_frame": max(0, int(current_frame)),
        "selections": {
            stage: int(frame)
            for stage, frame in selections.items()
            if stage in STAGE_KEYS and frame is not None
        },
    }


def save_review_progress(path, progress):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_review_progress(path, *, video_id=None):
    path = Path(path)
    if not path.exists():
        return None
    progress = json.loads(path.read_text(encoding="utf-8"))
    if progress.get("schema") != REVIEW_PROGRESS_SCHEMA:
        raise ValueError(f"지원하지 않는 검수 진행 형식입니다: {path}")
    if video_id is not None and progress.get("video_id") != video_id:
        raise ValueError("검수 진행 파일의 영상 ID가 현재 요청과 다릅니다.")
    return progress
