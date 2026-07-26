"""웹캠 코칭 상태를 진단용 스냅샷과 화면 패널로 변환합니다."""

import cv2


PANEL_WIDTH = 350
LINE_HEIGHT = 24


def _score(metrics, key):
    value = (metrics or {}).get(key)
    return None if value is None else int(round(value))


def build_runtime_diagnostics(
    *,
    phase,
    pose_detected,
    visibility=None,
    address=None,
    stability=None,
    feedback=None,
    pass_progress=0.0,
):
    """여러 런타임 상태를 저장과 화면 출력에 적합한 평면 구조로 만듭니다."""
    metrics = (feedback or {}).get("metrics", {})
    visibility_passed = None if visibility is None else bool(visibility.get("passed"))
    address_score = None if address is None else address.get("score")
    stability_ready = None if stability is None else bool(stability.get("ready"))
    stability_passed = None if stability is None else bool(stability.get("stable"))

    if not pose_detected:
        blocker = "NO_POSE"
    elif visibility_passed is False:
        blocker = "FULL_BODY"
    elif phase == "calibration" and address is not None and not address.get("passed"):
        blocker = "ADDRESS"
    elif stability is not None and not stability.get("ready"):
        blocker = "HOLD_STILL"
    elif stability_ready and not stability_passed:
        blocker = "MOVEMENT"
    elif feedback and feedback.get("status") == "unavailable":
        blocker = "NO_DATA"
    elif feedback and not feedback.get("passed"):
        blocker = "POSE_RULES"
    elif feedback and feedback.get("passed") and pass_progress < 1.0:
        blocker = "PASS_HOLD"
    else:
        blocker = "NONE"

    return {
        "phase": phase,
        "pose_detected": bool(pose_detected),
        "visibility": {
            "passed": visibility_passed,
            "missing": list((visibility or {}).get("missing", [])),
            "clipped": list((visibility or {}).get("clipped", [])),
        },
        "address": {
            "passed": None if address is None else bool(address.get("passed")),
            "score": None if address_score is None else int(round(address_score)),
        },
        "stability": {
            "ready": stability_ready,
            "stable": stability_passed,
            "duration_sec": None
            if stability is None
            else round(float(stability.get("duration_sec", 0.0)), 3),
            "mean_jitter": None
            if stability is None or stability.get("mean_jitter") is None
            else round(float(stability["mean_jitter"]), 5),
            "max_joint_jitter": None
            if stability is None or stability.get("max_joint_jitter") is None
            else round(float(stability["max_joint_jitter"]), 5),
        },
        "scores": {
            "guide": _score(metrics, "guide_score"),
            "caddieset_i7": _score(metrics, "caddieset_score"),
            "final": _score(metrics, "final_score"),
        },
        "feedback_status": None if feedback is None else feedback.get("status"),
        "pass_progress": round(float(pass_progress), 3),
        "blocker": blocker,
    }


def _format_value(value, *, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def diagnostic_lines(diagnostics):
    """OpenCV 패널에 표시할 짧은 영문 진단 행을 만듭니다."""
    visibility = diagnostics["visibility"]
    stability = diagnostics["stability"]
    scores = diagnostics["scores"]
    if visibility["passed"] is True:
        visibility_text = "OK"
    elif visibility["passed"] is False:
        issue_count = len(visibility["missing"]) + len(visibility["clipped"])
        visibility_text = f"CHECK ({issue_count})"
    else:
        visibility_text = "-"

    stable_text = "-"
    if stability["ready"] is False:
        stable_text = "COLLECTING"
    elif stability["stable"] is True:
        stable_text = "STABLE"
    elif stability["stable"] is False:
        stable_text = "MOVING"

    return [
        f"Phase       {diagnostics['phase'].upper()}",
        f"Pose        {'YES' if diagnostics['pose_detected'] else 'NO'}",
        f"Full body   {visibility_text}",
        f"Address     {_format_value(diagnostics['address']['score'], digits=0)}",
        (
            "Stability   "
            f"{stable_text} / {_format_value(stability['duration_sec'], digits=2)}s"
        ),
        (
            "Jitter      "
            f"{_format_value(stability['mean_jitter'], digits=4)} / "
            f"{_format_value(stability['max_joint_jitter'], digits=4)}"
        ),
        (
            "Scores      "
            f"G {_format_value(scores['guide'], digits=0)} | "
            f"I7 {_format_value(scores['caddieset_i7'], digits=0)} | "
            f"F {_format_value(scores['final'], digits=0)}"
        ),
        f"Pass hold   {diagnostics['pass_progress'] * 100:.0f}%",
        f"Blocker     {diagnostics['blocker']}",
    ]


def draw_runtime_diagnostics(frame, diagnostics, *, enabled=True):
    """웹캠 화면 우측 상단에 반투명 진단 패널을 그립니다."""
    if not enabled:
        return frame

    lines = diagnostic_lines(diagnostics)
    frame_height, frame_width = frame.shape[:2]
    panel_width = min(PANEL_WIDTH, max(240, frame_width - 20))
    panel_height = 42 + len(lines) * LINE_HEIGHT
    left = max(10, frame_width - panel_width - 10)
    top = 10
    bottom = min(frame_height - 10, top + panel_height)

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (left, top),
        (left + panel_width, bottom),
        (15, 15, 15),
        -1,
    )
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(
        frame,
        "RUNTIME DIAGNOSTICS [d]",
        (left + 12, top + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (80, 220, 255),
        2,
    )
    for index, line in enumerate(lines):
        y = top + 50 + index * LINE_HEIGHT
        if y >= bottom - 4:
            break
        color = (90, 255, 120) if line.endswith(("OK", "STABLE", "NONE")) else (235, 235, 235)
        cv2.putText(
            frame,
            line,
            (left + 12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame
