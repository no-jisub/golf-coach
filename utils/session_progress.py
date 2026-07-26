"""단계 통과 유지 시간, 자동 이동, 최종 점수를 관리합니다."""


class StageProgressTracker:
    def __init__(self, stage_keys, pass_hold_sec=2.0):
        if not stage_keys:
            raise ValueError("stage_keys는 비어 있을 수 없습니다.")
        if pass_hold_sec <= 0:
            raise ValueError("pass_hold_sec는 0보다 커야 합니다.")
        self.stage_keys = tuple(stage_keys)
        self.pass_hold_sec = float(pass_hold_sec)
        self.reset()

    def reset(self):
        self.current_index = 0
        self.pass_started_at = None
        self.scores = {}
        self.completed = False

    def select_stage(self, index):
        if not 0 <= index < len(self.stage_keys):
            raise IndexError("지원하지 않는 단계 번호입니다.")
        self.current_index = index
        self.pass_started_at = None
        self.completed = False

    @property
    def current_stage_key(self):
        return self.stage_keys[self.current_index]

    def pass_progress(self, now):
        if self.pass_started_at is None:
            return 0.0
        return min(max(now - self.pass_started_at, 0.0) / self.pass_hold_sec, 1.0)

    def update(self, now, feedback, *, stable):
        """현재 판정을 반영하고 자동 이동 이벤트를 반환합니다."""
        event = {
            "advanced": False,
            "completed": self.completed,
            "from_index": self.current_index,
            "to_index": self.current_index,
            "pass_progress": self.pass_progress(now),
        }
        if self.completed:
            return event

        if not stable or not feedback or not feedback.get("passed"):
            self.pass_started_at = None
            event["pass_progress"] = 0.0
            return event

        if self.pass_started_at is None:
            self.pass_started_at = now
            event["pass_progress"] = 0.0
            return event

        elapsed = now - self.pass_started_at
        event["pass_progress"] = min(elapsed / self.pass_hold_sec, 1.0)
        if elapsed < self.pass_hold_sec:
            return event

        metrics = feedback.get("metrics", {})
        score = metrics.get("final_score", metrics.get("guide_score", 100))
        self.scores[self.current_stage_key] = int(round(score))
        self.pass_started_at = None

        if self.current_index == len(self.stage_keys) - 1:
            self.completed = True
            event["completed"] = True
            return event

        self.current_index += 1
        event.update(
            {
                "advanced": True,
                "to_index": self.current_index,
                "pass_progress": 0.0,
            }
        )
        return event

    def summary(self):
        ordered_scores = [
            {
                "stage_key": stage_key,
                "score": self.scores.get(stage_key),
            }
            for stage_key in self.stage_keys
        ]
        completed_scores = [item["score"] for item in ordered_scores if item["score"] is not None]
        average_score = (
            round(sum(completed_scores) / len(completed_scores))
            if completed_scores
            else None
        )
        return {
            "completed": self.completed,
            "completed_count": len(completed_scores),
            "total_count": len(self.stage_keys),
            "average_score": average_score,
            "scores": ordered_scores,
        }
