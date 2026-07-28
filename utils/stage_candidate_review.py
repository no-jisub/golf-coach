"""단계 후보 프레임 선택을 검수 완료 스윙 정답 매니페스트로 변환합니다."""

from copy import deepcopy
from datetime import datetime

from utils.guide_alignment import STAGE_KEYS
from utils.swing_stage_audit import (
    update_manifest_summary,
    validate_ground_truth_manifest,
)


def candidate_indexes(candidate_manifest, stage_key):
    if stage_key not in candidate_manifest.get("stages", {}):
        raise ValueError(f"후보 매니페스트에 단계가 없습니다: {stage_key}")
    return [
        int(candidate["frame_index"])
        for candidate in candidate_manifest["stages"][stage_key].get(
            "candidates", []
        )
    ]


def finalize_stage_candidate_review(
    ground_truth_manifest,
    *,
    video_id,
    candidate_manifest,
    selections,
    reviewed_by,
    note="",
    reviewed_at=None,
    project_root=None,
):
    """8단계 선택값을 검증한 뒤 원본을 변경하지 않고 reviewed 매니페스트를 만듭니다."""
    if video_id not in ground_truth_manifest.get("videos", {}):
        raise KeyError(f"정답 매니페스트에 없는 영상 ID입니다: {video_id}")
    if not str(reviewed_by or "").strip():
        raise ValueError("reviewed_by가 필요합니다.")
    missing = [stage for stage in STAGE_KEYS if stage not in selections]
    if missing:
        raise ValueError(f"선택하지 않은 단계가 있습니다: {', '.join(missing)}")

    events = {}
    for stage_key in STAGE_KEYS:
        frame_index = int(selections[stage_key])
        if frame_index not in candidate_indexes(candidate_manifest, stage_key):
            raise ValueError(
                f"{stage_key}의 선택 프레임이 후보 목록에 없습니다: {frame_index}"
            )
        events[stage_key] = frame_index

    updated = deepcopy(ground_truth_manifest)
    video = updated["videos"][video_id]
    video.update(
        {
            "review_status": "reviewed",
            "events": events,
            "reviewed_by": str(reviewed_by).strip(),
            "reviewed_at": (
                reviewed_at or datetime.now().astimezone()
            ).isoformat(),
            "note": str(note or "").strip(),
            "review_source": {
                "type": "stage_candidate_selection",
                "candidate_schema": candidate_manifest.get("schema"),
                "source_video": candidate_manifest.get("source_video"),
            },
        }
    )
    update_manifest_summary(updated)
    validate_ground_truth_manifest(
        updated,
        project_root=project_root,
        require_files=False,
    )
    return updated
