# Golf Coach

초보자를 위한 단계별 골프 자세 코칭 프로그램입니다.

현재 목표는 MediaPipe Tasks API의 Pose Landmarker로 웹캠 화면에서 사람 관절을 인식하고, 주요 관절 점과 선을 OpenCV 화면에 표시한 뒤 어드레스 자세를 1차로 판단하는 것입니다.

## 실행 환경

- Windows
- Python 3.12.0
- PowerShell

## 설치

```powershell
cd C:\project\golf-coach
py -3.12 -m pip install -r requirements.txt
```

## 모델 파일

프로젝트 루트에 다음 파일이 필요합니다.

```text
pose_landmarker_full.task
```

파일이 없으면 `main.py` 실행 시 안내 메시지가 출력됩니다.

## 실행

```powershell
cd C:\project\golf-coach
py -3.12 main.py
```

웹캠 화면이 열리고 사람이 인식되면 어깨, 팔꿈치, 손목, 골반, 무릎, 발목 등에 점과 선이 표시됩니다.

종료하려면 OpenCV 창에서 `q` 키를 누릅니다.

## 어드레스 자세 분석

사람이 인식되면 최근 프레임을 약 2초 동안 모아 평균값으로 어드레스 자세를 판단합니다.

현재 확인하는 항목은 다음과 같습니다.

- 무릎 각도
- 상체 기울기
- 발 간격
- 어깨 기울기

OpenCV 기본 글꼴은 한글 표시가 제한적이므로, 자세 피드백 문장은 PowerShell 콘솔에 출력됩니다. 화면에는 `Address: PASS` 또는 `Address: CHECK`가 표시됩니다.

## 다음 개발 단계

1. 실제 촬영 각도에 맞춰 어드레스 기준값을 조정합니다.
2. 한글 피드백을 OpenCV 화면에 직접 표시합니다.
3. 테이크어웨이 자세 분석을 추가합니다.
4. 백스윙 탑 자세 분석을 추가합니다.
