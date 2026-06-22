import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("카메라를 읽을 수 없습니다.")
            break

        # OpenCV는 BGR, MediaPipe는 RGB 사용
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 자세 분석
        results = pose.process(image)

        # 다시 화면 출력용 BGR로 변환
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 관절 점 그리기
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow("Golf Pose Coach - Test", image)

        # q 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
