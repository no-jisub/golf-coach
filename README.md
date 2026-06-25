# Golf Coach

초보자를 위한 단계별 골프 자세 코칭 프로그램입니다.

MediaPipe Tasks API의 Pose Landmarker로 웹캠 화면에서 사람 관절을 인식하고, 주요 관절 점과 선을 OpenCV 화면에 표시합니다. 사용자가 각 자세를 천천히 잡고 멈추면 최근 프레임 평균값으로 자세를 1차 판단합니다.

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

프로그램 시작 후 먼저 캘리브레이션이 진행됩니다. 사용자는 어드레스 보조 스켈레톤에 몸을 맞추고, 머리부터 발목까지 보이도록 선 상태에서 5초 동안 자세를 유지하면 됩니다. 캘리브레이션 중에는 보조 스켈레톤이 사용자 위치와 체형 비율에 맞춰 따라옵니다.

5초 유지가 끝나면 그 순간까지의 평균 어깨 중심, 어깨 너비, 어깨-발목 세로 비율을 기준으로 보조 스켈레톤 위치와 크기가 고정됩니다. 중간에 몸이 크게 움직이거나 인식이 끊기면 캘리브레이션은 다시 시작됩니다. 다시 보정하려면 `c` 키를 누릅니다.

종료하려면 OpenCV 창에서 `q` 키를 누릅니다.

## 8단계 자세

현재 지원하는 골프 스윙 단계는 다음과 같습니다.

- `1`: 어드레스 Address
- `2`: 테이크백 Takeaway
- `3`: 백스윙 Backswing
- `4`: 백스윙 탑 Top of Swing
- `5`: 다운스윙 Downswing
- `6`: 임팩트 Impact
- `7`: 팔로우스루 Follow-through
- `8`: 피니쉬 Finish

추가 조작:

- `n`: 다음 단계
- `p`: 이전 단계
- `c`: 캘리브레이션 다시 시작
- `q`: 종료

## 현재 분석 방식

사람이 인식되면 최근 프레임을 약 2초 동안 모아 평균값으로 현재 선택된 단계의 자세를 판단합니다. 사용자 관절 좌표와 보조 스켈레톤 좌표를 어깨 중심, 어깨 너비, 어깨-발목 높이 기준으로 정규화한 뒤, 머리/팔/몸통/하체 차이를 계산합니다.

현재 기준은 MVP용 1차 기준입니다. 실제 골프 레슨 수준의 정밀 판정이 아니라, 단계별 자세를 잡고 피드백을 확인하는 흐름을 먼저 만들기 위한 기준입니다.

## 기준 자세 데이터 생성

프로 골프 자세 캡처 이미지는 `reference_data/raw_images` 아래 단계별 폴더에 저장합니다. 원본 이미지는 `.gitignore`로 제외되어 GitHub에 올라가지 않습니다.

이미지에서 관절 좌표를 추출하려면 다음을 실행합니다.

```powershell
py -3.12 tools\extract_reference_poses.py
```

추출된 좌표로 보조 스켈레톤 기준 데이터를 만들려면 다음을 실행합니다.

```powershell
py -3.12 tools\build_guide_poses.py
```

생성 결과는 다음 파일에 저장됩니다.

```text
reference_data\guide_poses\generated_guide_poses.json
```

앱은 이 파일이 있으면 자동으로 프로 사진 기반 보조 스켈레톤을 우선 사용하고, 파일이 없으면 코드에 들어있는 기본 스켈레톤을 사용합니다.

추출된 좌표가 사진 위에 잘 찍혔는지 확인하려면 overlay 이미지를 생성합니다.

```powershell
py -3.12 tools\visualize_reference_poses.py
```

검수 이미지는 다음 위치에 저장됩니다.

```text
reference_data\debug_overlay
```

관절 좌표가 이상하게 찍힌 경우 JSON을 이미지 위에서 직접 보정할 수 있습니다.

```powershell
py -3.12 tools\edit_reference_landmarks.py --stage address
```

편집 도구 조작:

- 마우스 드래그: 가까운 관절 이동
- 방향키 또는 `w/a/s/d`: 선택 관절 미세 이동
- `s`: JSON 저장
- `q` 또는 `Esc`: 종료

수동 보정 후에는 기준 스켈레톤을 다시 생성합니다.

```powershell
py -3.12 tools\build_guide_poses.py
```

## 다음 개발 단계

1. 실제 촬영 각도에 맞춰 각 단계 기준값을 조정합니다.
2. 오른손잡이/왼손잡이 설정을 추가합니다.
3. 단계별 점수를 저장하고 마지막에 요약 화면을 표시합니다.
4. 단계별 통과 시 자동으로 다음 단계로 넘어가게 만듭니다.

## 풀스윙 동영상에서 기준 프레임 뽑기

프로 풀스윙 동영상은 `reference_data\raw_videos` 폴더에 저장합니다. 동영상 원본은 `.gitignore`로 제외되어 GitHub에 올라가지 않습니다.

```powershell
py -3.12 tools\extract_video_frames.py reference_data\raw_videos\pro01_full_swing.mp4
```

도구 창에서 단계별 구간을 지정한 뒤 단계 키를 누르면 해당 구간에서 여러 프레임이 자동으로 저장됩니다. 구간을 지정하지 않고 단계 키를 누르면 현재 프레임 1장만 저장됩니다.

- `b`: 현재 프레임을 구간 시작으로 지정
- `e`: 현재 프레임을 구간 끝으로 지정
- `c`: 지정한 구간 초기화
- `Space` 또는 `p`: 재생/일시정지
- `a`/`d` 또는 방향키: 이전/다음 프레임 이동
- `q`: 종료

구간을 잡은 뒤 다음 키를 누르면 해당 단계 폴더에 프레임이 저장됩니다.

- `1`: Address
- `2`: Takeaway
- `3`: Backswing
- `4`: Top of Swing
- `5`: Downswing
- `6`: Impact
- `7`: Follow-through
- `8`: Finish

프레임 저장 후 기존 흐름을 그대로 실행합니다.

```powershell
py -3.12 tools\extract_reference_poses.py
py -3.12 tools\visualize_reference_poses.py
py -3.12 tools\build_guide_poses.py
```
