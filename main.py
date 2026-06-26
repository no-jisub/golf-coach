from collections import deque
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils.golf_rules import STAGE_CONFIGS, analyze_stage_pose
from utils.guide_skeleton import SWING_HAND, create_calibration_profile, draw_guide_skeleton, get_user_anchor
from utils.pose_drawer import draw_pose_landmarks


# Pose Landmarker 모델 파일은 프로젝트 루트에 둡니다.
MODEL_PATH = Path(__file__).with_name("pose_landmarker_full.task")

# 기본 웹캠 인덱스입니다. 다른 카메라를 쓰려면 1, 2 등으로 바꿔보세요.
CAMERA_INDEX = 0

# 자세를 멈춘 상태에서 최근 프레임을 모아 평균으로 판단합니다.
ANALYSIS_WINDOW_SEC = 2.0
MIN_SAMPLES_FOR_ANALYSIS = 20
CALIBRATION_HOLD_SEC = 5.0
CALIBRATION_MAX_ANCHOR_SHIFT_PX = 35

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

    while pose_samples and now - pose_samples[0][0] > ANALYSIS_WINDOW_SEC:
        pose_samples.popleft()


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
    print(f"- 판정: {'통과' if feedback['passed'] else '수정 필요'}")
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

    if latest_feedback["passed"]:
        return f"{stage_name} 통과", (80, 255, 120)

    return f"{stage_name} 수정 필요", (255, 220, 80)


def get_help_text():
    """하단 패널에 표시할 단계 조작 안내입니다."""
    return "1-8 단계 선택 | n/p 이전/다음 | c 보정 다시 | q 종료"


def draw_korean_feedback_panel(frame, current_stage, latest_feedback):
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

    title, title_color = get_stage_status_text(current_stage, latest_feedback)
    if latest_feedback is None:
        messages = ["카메라 앞에서 전신이 보이도록 서서 현재 자세를 1~2초 유지해주세요."]
    else:
        messages = latest_feedback["messages"]

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


def draw_status_text(frame, status_text, status_color, pose_samples, latest_feedback, current_stage_index):
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
        f"Guide: {swing_hand_text}",
        (30, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    sample_progress = min(len(pose_samples) / MIN_SAMPLES_FOR_ANALYSIS, 1.0)
    cv2.putText(
        frame,
        f"Sample: {sample_progress * 100:.0f}%",
        (30, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    if latest_feedback:
        result_text = "PASS" if latest_feedback["passed"] else "CHECK"
        result_color = (0, 255, 0) if latest_feedback["passed"] else (0, 200, 255)
        cv2.putText(
            frame,
            result_text,
            (30, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            result_color,
            2,
        )
        score = latest_feedback.get("metrics", {}).get("guide_score")
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


def draw_calibration_status(frame, calibration_start_time, calibration_profile):
    """초기 사용자 체형 보정 상태를 표시합니다."""
    if calibration_profile is not None:
        cv2.putText(
            frame,
            "Calibration: LOCKED",
            (30, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Press c to recalibrate guide position",
            (30, 365),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return

    elapsed = 0.0
    if calibration_start_time is not None:
        elapsed = time.monotonic() - calibration_start_time
    progress = min(elapsed / CALIBRATION_HOLD_SEC, 1.0)
    cv2.putText(
        frame,
        f"Calibration: {progress * 100:.0f}% ({elapsed:.1f}/{CALIBRATION_HOLD_SEC:.0f}s)",
        (30, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 200, 255),
        2,
    )
    cv2.putText(
        frame,
        "Match address guide and stand still",
        (30, 365),
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
    current_stage_index = 0
    pose_samples, latest_feedback, last_feedback_key = reset_analysis_state()
    calibration_samples = deque()
    calibration_start_time = None
    calibration_base_anchor = None
    calibration_profile = None

    with create_pose_landmarker() as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("카메라 프레임을 읽을 수 없습니다.")
                break

            # 거울처럼 보이도록 좌우 반전합니다.
            frame = cv2.flip(frame, 1)

            # OpenCV는 BGR, MediaPipe는 RGB 이미지를 사용합니다.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            last_timestamp_ms = get_video_timestamp_ms(start_time, last_timestamp_ms)
            result = landmarker.detect_for_video(mp_image, last_timestamp_ms)
            display_stage_index = 0 if calibration_profile is None else current_stage_index
            current_stage = STAGE_CONFIGS[display_stage_index]

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                if calibration_profile is None:
                    current_anchor = get_user_anchor(landmarks, frame.shape[1], frame.shape[0])
                    if calibration_start_time is None:
                        calibration_start_time = time.monotonic()
                        calibration_base_anchor = current_anchor
                    elif anchor_shift_too_large(calibration_base_anchor, current_anchor):
                        calibration_samples.clear()
                        calibration_start_time = time.monotonic()
                        calibration_base_anchor = current_anchor

                    calibration_samples.append(landmarks)
                    if time.monotonic() - calibration_start_time >= CALIBRATION_HOLD_SEC:
                        calibration_profile = create_calibration_profile(
                            list(calibration_samples),
                            frame.shape[1],
                            frame.shape[0],
                        )
                        current_stage_index = 0
                        current_stage = STAGE_CONFIGS[current_stage_index]
                        pose_samples, latest_feedback, last_feedback_key = reset_analysis_state()

                draw_guide_skeleton(frame, current_stage["key"], landmarks, calibration_profile)
                draw_pose_landmarks(frame, landmarks)

                now = time.monotonic()
                if calibration_profile is not None:
                    update_pose_samples(pose_samples, landmarks, now)

                if calibration_profile is not None and len(pose_samples) >= MIN_SAMPLES_FOR_ANALYSIS:
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

                status_text = "Pose detected"
                status_color = (0, 255, 0)
            else:
                draw_guide_skeleton(frame, current_stage["key"], calibration_profile=calibration_profile)
                calibration_samples.clear()
                calibration_start_time = None
                calibration_base_anchor = None
                pose_samples.clear()
                latest_feedback = None
                status_text = "No pose detected"
                status_color = (0, 0, 255)

            draw_status_text(
                frame,
                status_text,
                status_color,
                pose_samples,
                latest_feedback,
                display_stage_index,
            )
            draw_calibration_status(frame, calibration_start_time, calibration_profile)
            frame = draw_korean_feedback_panel(frame, current_stage, latest_feedback)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                calibration_samples.clear()
                calibration_start_time = None
                calibration_base_anchor = None
                calibration_profile = None
                current_stage_index = 0
                pose_samples, latest_feedback, last_feedback_key = reset_analysis_state()
                continue

            current_stage_index, should_quit, stage_changed = handle_key(key, current_stage_index)
            if should_quit:
                break
            if stage_changed:
                pose_samples, latest_feedback, last_feedback_key = reset_analysis_state()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
