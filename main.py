from collections import deque
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils.app_config import MVP_CLUB_LABEL, MVP_VIEW_LABEL
from utils.caddieset_metrics import average_landmark_points
from utils.diagnostic_capture import save_runtime_sample
from utils.golf_rules import STAGE_CONFIGS, analyze_stage_pose
from utils.guide_skeleton import SWING_HAND, create_calibration_profile, draw_guide_skeleton, get_user_anchor
from utils.guide_tolerance import get_stage_tolerance_regions
from utils.pose_drawer import draw_pose_landmarks
from utils.pose_quality import (
    check_full_body_visibility,
    evaluate_calibration_frame,
    evaluate_pose_stability,
    trim_timed_samples,
)
from utils.runtime_diagnostics import (
    build_runtime_diagnostics,
    draw_runtime_diagnostics,
)
from utils.session_progress import StageProgressTracker


# Pose Landmarker 모델 파일은 프로젝트 루트에 둡니다.
MODEL_PATH = Path(__file__).with_name("pose_landmarker_full.task")

# 기본 웹캠 인덱스입니다. 다른 카메라를 쓰려면 1, 2 등으로 바꿔보세요.
CAMERA_INDEX = 0

# 자세를 멈춘 상태에서 최근 프레임을 모아 평균으로 판단합니다.
ANALYSIS_WINDOW_SEC = 2.0
ANALYSIS_STABILITY_SEC = 1.5
CALIBRATION_HOLD_SEC = 5.0
CALIBRATION_STABILITY_SEC = 1.5
CALIBRATION_MAX_ANCHOR_SHIFT_PX = 35
AUTO_PASS_HOLD_SEC = 2.0

# Windows 기본 한글 폰트입니다.
KOREAN_FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")


def create_pose_landmarker():
    """MediaPipe Tasks API 기반 Pose Landmarker를 생성합니다."""
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def get_video_timestamp_ms(start_time, last_timestamp_ms):
    """VIDEO 모드에서 필요한 증가하는 타임스탬프를 만듭니다."""
    timestamp_ms = int((time.monotonic() - start_time) * 1000)
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    return timestamp_ms


def update_pose_samples(pose_samples, landmarks, now):
    """최근 분석 구간 안의 랜드마크만 유지합니다."""
    pose_samples.append((now, landmarks))
    trim_timed_samples(pose_samples, now, ANALYSIS_WINDOW_SEC)


def print_feedback_if_changed(feedback, last_feedback_key):
    """화면 피드백과 별도로 PowerShell에도 피드백을 남깁니다."""
    feedback_key = (
        feedback["stage_key"],
        feedback["passed"],
        *feedback["messages"],
    )
    if feedback_key == last_feedback_key:
        return last_feedback_key

    print()
    print(f"[{feedback['stage_korean']} 자세 피드백]")
    for message in feedback["messages"]:
        print(f"- {message}")
    status_labels = {
        "pass": "통과",
        "warning": "주의",
        "unavailable": "측정 불가",
    }
    status = feedback.get("status", "pass" if feedback["passed"] else "warning")
    print(f"- 판정: {status_labels.get(status, status)}")
    return feedback_key


def load_korean_font(size):
    """Windows 한글 폰트를 불러옵니다."""
    if KOREAN_FONT_PATH.exists():
        return ImageFont.truetype(str(KOREAN_FONT_PATH), size)
    return ImageFont.load_default()


def wrap_text(text, font, max_width):
    """PIL 폰트 폭 기준으로 문장을 여러 줄로 나눕니다."""
    lines = []
    current_line = ""

    for char in text:
        next_line = current_line + char
        bbox = font.getbbox(next_line)
        text_width = bbox[2] - bbox[0]

        if text_width <= max_width or not current_line:
            current_line = next_line
        else:
            lines.append(current_line)
            current_line = char

    if current_line:
        lines.append(current_line)

    return lines


def get_stage_status_text(current_stage, latest_feedback):
    """하단 패널의 제목과 색상을 만듭니다."""
    stage_name = f"{current_stage['korean']} ({current_stage['label']})"

    if latest_feedback is None:
        return f"{stage_name} 분석 대기", (255, 255, 255)

    feedback_status = latest_feedback.get(
        "status",
        "pass" if latest_feedback["passed"] else "warning",
    )
    if feedback_status == "pass":
        return f"{stage_name} 통과", (80, 255, 120)
    if feedback_status == "unavailable":
        return f"{stage_name} 측정 불가", (255, 150, 100)

    return f"{stage_name} 주의", (255, 220, 80)


def get_help_text():
    """하단 패널에 표시할 단계 조작 안내입니다."""
    return "d 진단 | s 저장 | g 좋은자세 | b 나쁜자세 | c 재보정 | q 종료"


def draw_korean_feedback_panel(
    frame,
    current_stage,
    latest_feedback,
    waiting_message=None,
    session_summary=None,
    pass_progress=0.0,
):
    """웹캠 화면 하단에 현재 단계와 한글 자세 피드백을 표시합니다."""
    image_height, image_width, _ = frame.shape
    panel_height = 180
    panel_y = image_height - panel_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_y), (image_width, image_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(pil_image)

    title_font = load_korean_font(24)
    body_font = load_korean_font(20)
    help_font = load_korean_font(15)

    if session_summary and session_summary.get("completed"):
        average_score = session_summary.get("average_score", 0)
        title = f"8단계 코칭 완료 - 평균 {average_score}점"
        title_color = (80, 255, 120)
        score_items = session_summary["scores"]
        first_half = " | ".join(
            f"{index + 1}:{item['score']}"
            for index, item in enumerate(score_items[:4])
        )
        second_half = " | ".join(
            f"{index + 5}:{item['score']}"
            for index, item in enumerate(score_items[4:])
        )
        messages = [
            f"1~4단계 점수: {first_half}",
            f"5~8단계 점수: {second_half}",
            "c 키를 누르면 체형 보정부터 다시 시작합니다.",
        ]
    else:
        title, title_color = get_stage_status_text(current_stage, latest_feedback)
        if latest_feedback is None:
            messages = [
                waiting_message
                or "카메라 앞에서 전신이 보이도록 서서 현재 자세를 유지해주세요."
            ]
        else:
            messages = list(latest_feedback["messages"])
            if latest_feedback.get("passed"):
                messages.append(
                    f"다음 단계 이동까지 통과 자세 유지: {pass_progress * 100:.0f}%"
                )

    draw.text((24, panel_y + 14), title, font=title_font, fill=title_color)
    draw.text((24, panel_y + 45), get_help_text(), font=help_font, fill=(210, 210, 210))

    y = panel_y + 74
    max_text_width = image_width - 48
    visible_lines = []
    for message in messages:
        visible_lines.extend(wrap_text(f"- {message}", body_font, max_text_width))

    for line in visible_lines[:3]:
        draw.text((24, y), line, font=body_font, fill=(255, 255, 255))
        y += 28

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def draw_status_text(
    frame,
    status_text,
    status_color,
    pose_samples,
    latest_feedback,
    current_stage_index,
    pass_progress=0.0,
):
    """화면 좌측 상단에 현재 상태를 표시합니다."""
    current_stage = STAGE_CONFIGS[current_stage_index]
    total_stages = len(STAGE_CONFIGS)

    cv2.putText(
        frame,
        status_text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        2,
    )
    cv2.putText(
        frame,
        "Press q to quit",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Stage {current_stage_index + 1}/{total_stages}: {current_stage['label']}",
        (30, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    swing_hand_text = "Right-handed" if SWING_HAND == "right" else "Left-handed"
    cv2.putText(
        frame,
        f"MVP: {MVP_VIEW_LABEL} / {MVP_CLUB_LABEL} / {swing_hand_text}",
        (30, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    sample_duration = 0.0
    if len(pose_samples) >= 2:
        sample_duration = pose_samples[-1][0] - pose_samples[0][0]
    sample_progress = min(sample_duration / ANALYSIS_STABILITY_SEC, 1.0)
    cv2.putText(
        frame,
        f"Stable sample: {sample_progress * 100:.0f}%",
        (30, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    if latest_feedback:
        feedback_status = latest_feedback.get(
            "status",
            "pass" if latest_feedback["passed"] else "warning",
        )
        result_texts = {
            "pass": "PASS",
            "warning": "WARNING",
            "unavailable": "NO DATA",
        }
        result_colors = {
            "pass": (0, 255, 0),
            "warning": (0, 200, 255),
            "unavailable": (80, 150, 255),
        }
        result_text = result_texts.get(feedback_status, "CHECK")
        result_color = result_colors.get(feedback_status, (0, 200, 255))
        cv2.putText(
            frame,
            result_text,
            (30, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            result_color,
            2,
        )
        feedback_metrics = latest_feedback.get("metrics", {})
        if latest_feedback.get("source") == "combined":
            final_score = feedback_metrics.get("final_score", 0)
            guide_score = feedback_metrics.get("guide_score", 0)
            caddieset_score = feedback_metrics.get("caddieset_score", 0)
            cv2.putText(
                frame,
                f"Score: {final_score} (Guide {guide_score} / I7 {caddieset_score})",
                (30, 290),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
        else:
            score = feedback_metrics.get("guide_score")
            if score is not None:
                cv2.putText(
                    frame,
                    f"Guide score: {score}",
                    (30, 290),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

    cv2.putText(
        frame,
        f"Pass hold: {pass_progress * 100:.0f}%",
        (30, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (80, 255, 120) if pass_progress > 0 else (180, 180, 180),
        2,
    )


def draw_calibration_status(
    frame,
    calibration_progress,
    calibration_profile,
    calibration_message=None,
):
    """초기 사용자 체형 보정 상태를 표시합니다."""
    if calibration_profile is not None:
        cv2.putText(
            frame,
            "Calibration: LOCKED",
            (30, 370),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Press c to recalibrate guide position",
            (30, 405),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return

    cv2.putText(
        frame,
        f"Calibration: {calibration_progress * 100:.0f}%",
        (30, 370),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 200, 255),
        2,
    )
    cv2.putText(
        frame,
        calibration_message or "Show full body, match address guide, and stand still",
        (30, 405),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )


def reset_analysis_state():
    """단계가 바뀔 때 이전 프레임 평균과 피드백을 초기화합니다."""
    return deque(), None, None


def anchor_shift_too_large(base_anchor, current_anchor):
    """캘리브레이션 중 사용자가 많이 움직였는지 확인합니다."""
    if base_anchor is None or current_anchor is None:
        return False

    base_mid, base_width = base_anchor
    current_mid, current_width = current_anchor
    dx = current_mid[0] - base_mid[0]
    dy = current_mid[1] - base_mid[1]
    center_shift = (dx * dx + dy * dy) ** 0.5
    width_shift = abs(current_width - base_width)
    return center_shift > CALIBRATION_MAX_ANCHOR_SHIFT_PX or width_shift > CALIBRATION_MAX_ANCHOR_SHIFT_PX


def handle_key(key, current_stage_index):
    """키 입력으로 단계 변경이나 종료 여부를 처리합니다."""
    if key == ord("q"):
        return current_stage_index, True, False

    if key == ord("n"):
        next_index = min(current_stage_index + 1, len(STAGE_CONFIGS) - 1)
        return next_index, False, next_index != current_stage_index

    if key == ord("p"):
        next_index = max(current_stage_index - 1, 0)
        return next_index, False, next_index != current_stage_index

    number_keys = {ord(str(index + 1)): index for index in range(len(STAGE_CONFIGS))}
    if key in number_keys:
        next_index = number_keys[key]
        return next_index, False, next_index != current_stage_index

    return current_stage_index, False, False


def main():
    if not MODEL_PATH.exists():
        print("pose_landmarker_full.task 파일이 필요합니다.")
        print(f"현재 찾는 위치: {MODEL_PATH}")
        print("MediaPipe Pose Landmarker full 모델을 다운로드한 뒤 프로젝트 폴더에 넣어주세요.")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"카메라를 열 수 없습니다. 카메라 인덱스 {CAMERA_INDEX}를 확인해주세요.")
        return

    window_name = "Golf Coach - Pose Landmarker"
    start_time = time.monotonic()
    last_timestamp_ms = -1
    stage_keys = [stage["key"] for stage in STAGE_CONFIGS]
    progress_tracker = StageProgressTracker(stage_keys, AUTO_PASS_HOLD_SEC)
    pose_samples, latest_feedback, last_feedback_key = reset_analysis_state()
    calibration_samples = deque()
    calibration_start_time = None
    calibration_base_anchor = None
    calibration_profile = None
    calibration_progress = 0.0
    calibration_message = "전신을 보이고 어드레스 가이드에 맞춰주세요."
    analysis_message = "현재 단계 자세를 잡고 잠시 멈춰주세요."
    diagnostics_enabled = True
    capture_notice = None
    capture_notice_until = 0.0

    try:
        with create_pose_landmarker() as landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("카메라 프레임을 읽을 수 없습니다.")
                    break

                now = time.monotonic()

                # 거울처럼 보이도록 좌우 반전합니다.
                frame = cv2.flip(frame, 1)
                raw_camera_frame = frame.copy()

                # OpenCV는 BGR, MediaPipe는 RGB 이미지를 사용합니다.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                last_timestamp_ms = get_video_timestamp_ms(start_time, last_timestamp_ms)
                result = landmarker.detect_for_video(mp_image, last_timestamp_ms)
                display_stage_index = (
                    0 if calibration_profile is None else progress_tracker.current_index
                )
                current_stage = STAGE_CONFIGS[display_stage_index]
                diagnostic_visibility = None
                diagnostic_address = None
                diagnostic_stability = None

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]

                    if calibration_profile is None:
                        quality = evaluate_calibration_frame(landmarks)
                        diagnostic_visibility = quality.get("visibility")
                        diagnostic_address = quality.get("address")
                        current_anchor = get_user_anchor(
                            landmarks,
                            frame.shape[1],
                            frame.shape[0],
                        )

                        if not quality["passed"]:
                            calibration_samples.clear()
                            calibration_start_time = None
                            calibration_base_anchor = None
                            calibration_progress = 0.0
                            calibration_message = quality["messages"][0]
                        else:
                            if calibration_start_time is None:
                                calibration_start_time = now
                                calibration_base_anchor = current_anchor
                                calibration_samples.clear()
                            elif anchor_shift_too_large(
                                calibration_base_anchor,
                                current_anchor,
                            ):
                                calibration_start_time = now
                                calibration_base_anchor = current_anchor
                                calibration_samples.clear()

                            calibration_samples.append((now, landmarks))
                            recent_calibration_samples = deque(
                                sample
                                for sample in calibration_samples
                                if now - sample[0] <= CALIBRATION_STABILITY_SEC
                            )
                            calibration_stability = evaluate_pose_stability(
                                recent_calibration_samples,
                                min_duration_sec=CALIBRATION_STABILITY_SEC,
                            )
                            diagnostic_stability = calibration_stability

                            if (
                                calibration_stability["ready"]
                                and not calibration_stability["stable"]
                            ):
                                calibration_samples.clear()
                                calibration_samples.append((now, landmarks))
                                calibration_start_time = now
                                calibration_base_anchor = current_anchor
                                calibration_progress = 0.0
                                calibration_message = calibration_stability["message"]
                            else:
                                elapsed = now - calibration_start_time
                                calibration_progress = min(
                                    elapsed / CALIBRATION_HOLD_SEC,
                                    1.0,
                                )
                                calibration_message = (
                                    "어드레스 자세를 움직이지 말고 유지해주세요. "
                                    f"보정 {calibration_progress * 100:.0f}%"
                                    if not calibration_stability["stable"]
                                    else "자세가 안정적입니다. 그대로 유지해주세요. "
                                    f"보정 {calibration_progress * 100:.0f}%"
                                )

                                if (
                                    elapsed >= CALIBRATION_HOLD_SEC
                                    and calibration_stability["stable"]
                                ):
                                    calibration_landmarks = [
                                        sample[1] for sample in calibration_samples
                                    ]
                                    calibration_profile = create_calibration_profile(
                                        calibration_landmarks,
                                        frame.shape[1],
                                        frame.shape[0],
                                    )
                                    if calibration_profile is not None:
                                        calibration_profile[
                                            "caddieset_address_points"
                                        ] = average_landmark_points(
                                            calibration_landmarks
                                        )
                                        progress_tracker.reset()
                                        pose_samples, latest_feedback, last_feedback_key = (
                                            reset_analysis_state()
                                        )
                                        analysis_message = (
                                            "어드레스 자세를 유지하면 분석을 시작합니다."
                                        )

                    elif not progress_tracker.completed:
                        visibility = check_full_body_visibility(landmarks)
                        diagnostic_visibility = visibility
                        if not visibility["passed"]:
                            pose_samples.clear()
                            latest_feedback = None
                            analysis_message = visibility["messages"][0]
                            progress_tracker.update(now, None, stable=False)
                        else:
                            update_pose_samples(pose_samples, landmarks, now)
                            stability = evaluate_pose_stability(
                                pose_samples,
                                min_duration_sec=ANALYSIS_STABILITY_SEC,
                            )
                            diagnostic_stability = stability
                            analysis_message = stability["message"]

                            if stability["ready"] and stability["stable"]:
                                latest_feedback = analyze_stage_pose(
                                    current_stage["key"],
                                    [sample[1] for sample in pose_samples],
                                    calibration_profile,
                                    frame.shape[1],
                                    frame.shape[0],
                                )
                                last_feedback_key = print_feedback_if_changed(
                                    latest_feedback,
                                    last_feedback_key,
                                )
                                event = progress_tracker.update(
                                    now,
                                    latest_feedback,
                                    stable=True,
                                )
                                if event["advanced"]:
                                    pose_samples, latest_feedback, last_feedback_key = (
                                        reset_analysis_state()
                                    )
                                    analysis_message = (
                                        "다음 단계 자세를 잡고 잠시 멈춰주세요."
                                    )
                            else:
                                latest_feedback = None
                                progress_tracker.update(now, None, stable=False)

                    display_stage_index = (
                        0 if calibration_profile is None else progress_tracker.current_index
                    )
                    current_stage = STAGE_CONFIGS[display_stage_index]
                    draw_guide_skeleton(
                        frame,
                        current_stage["key"],
                        landmarks,
                        calibration_profile,
                        get_stage_tolerance_regions(current_stage["key"]),
                    )
                    draw_pose_landmarks(frame, landmarks)
                    status_text = "Pose detected"
                    status_color = (0, 255, 0)
                else:
                    draw_guide_skeleton(
                        frame,
                        current_stage["key"],
                        calibration_profile=calibration_profile,
                        tolerance_regions=get_stage_tolerance_regions(
                            current_stage["key"]
                        ),
                    )
                    if calibration_profile is None:
                        calibration_samples.clear()
                        calibration_start_time = None
                        calibration_base_anchor = None
                        calibration_progress = 0.0
                        calibration_message = "사람을 인식하지 못했습니다. 전신을 보여주세요."
                    else:
                        progress_tracker.update(now, None, stable=False)
                        analysis_message = "사람을 인식하지 못했습니다. 전신을 보여주세요."
                    pose_samples.clear()
                    latest_feedback = None
                    status_text = "No pose detected"
                    status_color = (0, 0, 255)

                display_stage_index = (
                    0 if calibration_profile is None else progress_tracker.current_index
                )
                current_stage = STAGE_CONFIGS[display_stage_index]
                draw_status_text(
                    frame,
                    status_text,
                    status_color,
                    pose_samples,
                    latest_feedback,
                    display_stage_index,
                    progress_tracker.pass_progress(now),
                )
                draw_calibration_status(
                    frame,
                    calibration_progress,
                    calibration_profile,
                    calibration_message,
                )
                diagnostic_phase = (
                    "completed"
                    if progress_tracker.completed
                    else "calibration"
                    if calibration_profile is None
                    else "analysis"
                )
                runtime_diagnostics = build_runtime_diagnostics(
                    phase=diagnostic_phase,
                    pose_detected=bool(result.pose_landmarks),
                    visibility=diagnostic_visibility,
                    address=diagnostic_address,
                    stability=diagnostic_stability,
                    feedback=latest_feedback,
                    pass_progress=progress_tracker.pass_progress(now),
                )
                draw_runtime_diagnostics(
                    frame,
                    runtime_diagnostics,
                    enabled=diagnostics_enabled,
                )
                frame = draw_korean_feedback_panel(
                    frame,
                    current_stage,
                    latest_feedback,
                    calibration_message
                    if calibration_profile is None
                    else analysis_message,
                    progress_tracker.summary(),
                    progress_tracker.pass_progress(now),
                )
                if capture_notice and now < capture_notice_until:
                    cv2.putText(
                        frame,
                        capture_notice,
                        (24, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (80, 255, 120),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                capture_labels = {
                    ord("s"): "pending",
                    ord("g"): "expected_pass",
                    ord("b"): "expected_fail",
                }
                if key in capture_labels:
                    try:
                        capture = save_runtime_sample(
                            raw_frame=raw_camera_frame,
                            overlay_frame=frame,
                            stage_key=current_stage["key"],
                            landmarks=(
                                result.pose_landmarks[0]
                                if result.pose_landmarks
                                else []
                            ),
                            diagnostics=runtime_diagnostics,
                            feedback=latest_feedback,
                            expected_label=capture_labels[key],
                        )
                        capture_notice = (
                            f"Saved: {capture['expected_label']} / "
                            f"{capture['discrepancy']}"
                        )
                        capture_notice_until = time.monotonic() + 2.0
                        print(f"[진단 샘플 저장] {capture['sample_dir']}")
                    except (OSError, ValueError) as error:
                        capture_notice = f"Save failed: {error}"
                        capture_notice_until = time.monotonic() + 3.0
                        print(f"[진단 샘플 저장 실패] {error}")
                    continue
                if key == ord("d"):
                    diagnostics_enabled = not diagnostics_enabled
                    continue
                if key == ord("c"):
                    calibration_samples.clear()
                    calibration_start_time = None
                    calibration_base_anchor = None
                    calibration_profile = None
                    calibration_progress = 0.0
                    calibration_message = (
                        "전신을 보이고 어드레스 가이드에 맞춰주세요."
                    )
                    progress_tracker.reset()
                    pose_samples, latest_feedback, last_feedback_key = (
                        reset_analysis_state()
                    )
                    continue

                next_stage_index, should_quit, stage_changed = handle_key(
                    key,
                    progress_tracker.current_index,
                )
                if should_quit:
                    break
                if stage_changed and calibration_profile is not None:
                    progress_tracker.select_stage(next_stage_index)
                    pose_samples, latest_feedback, last_feedback_key = (
                        reset_analysis_state()
                    )
                    analysis_message = "선택한 단계 자세를 잡고 잠시 멈춰주세요."
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
