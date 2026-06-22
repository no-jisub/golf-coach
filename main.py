from collections import deque
from pathlib import Path
import time

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils.golf_rules import analyze_address_pose
from utils.pose_drawer import draw_pose_landmarks


# Pose Landmarker 모델 파일은 프로젝트 루트에 둡니다.
MODEL_PATH = Path(__file__).with_name("pose_landmarker_full.task")

# 기본 웹캠 인덱스입니다. 다른 카메라를 쓰려면 1, 2 등으로 바꿔보세요.
CAMERA_INDEX = 0

# 자세를 멈춘 상태에서 최근 프레임을 모아 평균으로 판단합니다.
ANALYSIS_WINDOW_SEC = 2.0
MIN_SAMPLES_FOR_ANALYSIS = 20


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
    """OpenCV 기본 글꼴은 한글 표시가 제한적이므로 콘솔에도 피드백을 출력합니다."""
    feedback_key = tuple(feedback["messages"])
    if feedback_key == last_feedback_key:
        return last_feedback_key

    print()
    print("[어드레스 자세 피드백]")
    for message in feedback["messages"]:
        print(f"- {message}")
    print(f"- 판정: {'통과' if feedback['passed'] else '수정 필요'}")
    return feedback_key


def draw_status_text(frame, status_text, status_color, pose_samples, latest_feedback):
    """화면 좌측 상단에 현재 상태를 표시합니다."""
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

    sample_progress = min(len(pose_samples) / MIN_SAMPLES_FOR_ANALYSIS, 1.0)
    cv2.putText(
        frame,
        f"Address sample: {sample_progress * 100:.0f}%",
        (30, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    if latest_feedback:
        address_text = "Address: PASS" if latest_feedback["passed"] else "Address: CHECK"
        address_color = (0, 255, 0) if latest_feedback["passed"] else (0, 200, 255)
        cv2.putText(
            frame,
            address_text,
            (30, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            address_color,
            2,
        )
        cv2.putText(
            frame,
            "Korean feedback is printed in PowerShell",
            (30, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )


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
    pose_samples = deque()
    latest_feedback = None
    last_feedback_key = None

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

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                draw_pose_landmarks(frame, landmarks)

                now = time.monotonic()
                update_pose_samples(pose_samples, landmarks, now)

                if len(pose_samples) >= MIN_SAMPLES_FOR_ANALYSIS:
                    latest_feedback = analyze_address_pose([sample[1] for sample in pose_samples])
                    last_feedback_key = print_feedback_if_changed(
                        latest_feedback,
                        last_feedback_key,
                    )

                status_text = "Pose detected"
                status_color = (0, 255, 0)
            else:
                pose_samples.clear()
                latest_feedback = None
                status_text = "No pose detected"
                status_color = (0, 0, 255)

            draw_status_text(frame, status_text, status_color, pose_samples, latest_feedback)
            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
