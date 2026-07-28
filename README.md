# Golf Coach

초보자를 위한 단계별 골프 자세 코칭 프로그램입니다.

현재 목표는 빠른 풀스윙을 실시간으로 완벽하게 분석하는 것이 아닙니다. 사용자가 각 자세를 천천히 잡고 잠시 멈추면, PC 웹캠 화면에서 사람 관절을 인식하고 보조 스켈레톤과 비교해서 통과 여부와 수정 피드백을 주는 방식으로 개발하고 있습니다.

이 문서는 다른 개발자나 다른 AI가 프로젝트 의도와 현재 방향을 이어받을 수 있도록 작성한 개발 인수인계 문서입니다.

## 개발 환경

- OS: Windows
- Shell: PowerShell
- Python: 3.12.0
- 실행 기준: `py -3.12`
- 프로젝트 경로: `C:\project\golf-coach`

설치:

```powershell
cd C:\project\golf-coach
py -3.12 -m pip install -r requirements.txt
```

현재 의존성:

```text
opencv-python
mediapipe
numpy
Pillow
```

`Pillow`는 OpenCV 화면 하단에 한국어 피드백을 출력하기 위해 사용합니다.

## 실행 방법

```powershell
cd C:\project\golf-coach
py -3.12 main.py
```

필수 모델 파일:

```text
pose_landmarker_full.task
```

이 파일은 프로젝트 루트에 있어야 합니다. 없으면 `main.py`가 친절한 오류 메시지를 출력하고 종료합니다.

기본 카메라 인덱스는 `0`입니다. 카메라가 열리지 않으면 `main.py`의 `CAMERA_INDEX` 값을 `1`, `2` 등으로 바꿔 테스트합니다.

종료는 OpenCV 창에서 `q`를 누릅니다.

## 핵심 목표

최종 목표는 다음 흐름입니다.

1. 사용자가 어드레스 자세를 잡고 멈춤
2. 프로그램이 웹캠 프레임에서 관절 좌표를 추출
3. 보조 스켈레톤과 현재 사용자 자세를 비교
4. 통과 또는 수정 피드백을 화면 하단에 표시
5. 다음 단계로 이동
6. 8단계 완료 후 단계별 점수와 피드백 제공

현재 MVP는 8단계 전체 화면 흐름과 보조 스켈레톤 비교까지 들어가 있지만, 판정 기준은 아직 실제 레슨 품질 수준이 아닙니다. 우선 “단계별 자세를 잡고 피드백을 받는 제품 흐름”을 만든 뒤, 기준 자세 데이터를 점진적으로 개선하는 방향입니다.

## 현재 8단계

프로그램은 정면 웹캠 기준으로 아래 8단계를 지원합니다.

- `1`: 어드레스 Address
- `2`: 테이크어웨이 Takeaway
- `3`: 백스윙 Backswing
- `4`: 백스윙 탑 Top of Swing
- `5`: 다운스윙 Downswing
- `6`: 임팩트 Impact
- `7`: 팔로우스루 Follow-through
- `8`: 피니쉬 Finish

키 조작:

- `1`~`8`: 단계 직접 선택
- `n`: 다음 단계
- `p`: 이전 단계
- `c`: 사용자 체형/위치 보정 다시 시작
- `d`: 웹캠 자세 진단 패널 표시/숨김
- `s`: 현재 프레임을 판정 라벨 없이 로컬 저장
- `g`: 현재 프레임을 실제 좋은 자세로 라벨링해 로컬 저장
- `b`: 현재 프레임을 실제 수정 필요 자세로 라벨링해 로컬 저장
- `q`: 종료

현재 MVP 촬영 조건:

- 정면(`FACEON`)
- 우타
- 7번 아이언(`I7`)

판정이 통과인 상태를 2초 동안 유지하면 점수를 저장하고 다음 단계로 자동 이동합니다.
8단계를 모두 마치면 화면 하단에 단계별 점수와 평균 점수가 표시됩니다.

## 현재 앱 동작 방식

`main.py`가 메인 실행 파일입니다.

동작 순서:

1. `pose_landmarker_full.task` 모델 파일 확인
2. OpenCV로 웹캠 열기
3. MediaPipe Tasks API `PoseLandmarker`를 `VIDEO` 모드로 실행
4. 웹캠 프레임을 좌우 반전해서 거울처럼 표시
5. 사람 관절 좌표 추출
6. 사용자 관절점 표시
7. 보조 스켈레톤과 샤프트 표시
8. 주요 관절이 1.5초 이상 안정적인지 검사
9. 안정된 최근 자세만 평균 자세로 계산
10. 화면 가이드와 7번 아이언 CaddieSet 범위를 통합해 피드백과 점수 생성
11. 통과를 2초 유지하면 다음 단계로 자동 이동
12. 화면 하단에 한국어 피드백 표시

웹캠 진단 패널에는 전신 노출, 어드레스 유사도, 자세 안정 시간과 흔들림,
가이드 점수, 7번 아이언 점수, 최종 점수, 통과 유지 시간과 현재 차단 사유가
표시됩니다. `g`와 `b`로 저장한 테스트 자세는
`reference_data/runtime_samples`에 원본 이미지, 오버레이 이미지, 관절 좌표 JSON으로
남습니다. 이 폴더는 `.gitignore`에 포함되어 GitHub에 올라가지 않습니다.

중요한 방향:

- `mp.solutions.pose` 방식은 예전 테스트용입니다.
- 최종 방향은 MediaPipe Tasks API `PoseLandmarker`입니다.
- 실시간 빠른 스윙 분석보다, 멈춘 자세를 1~2초 동안 안정적으로 수집해 판단하는 방식입니다.

## 사용자 보정 단계

프로그램 시작 직후 바로 자세 판정을 하지 않습니다.

먼저 사용자가 어드레스 보조 스켈레톤에 몸을 맞춘 상태로 약 5초 동안 멈추면, 프로그램이 사용자의 위치와 체형 비율을 기준으로 보조 스켈레톤 크기와 위치를 고정합니다.

보정에서 사용하는 주요 기준:

- 머리, 양팔, 골반, 무릎, 발목의 화면 노출과 visibility
- 신체가 화면 가장자리에서 잘리지 않았는지
- 현재 자세가 어드레스 가이드와 유사한지
- 최근 1.5초 동안 주요 관절이 안정적인지
- 어깨 중심
- 어깨 너비
- 어깨에서 발목까지의 세로 길이
- 사용자 위치 변화량

전신 인식이 끊기거나 어드레스와 다른 자세를 잡거나 보정 중 크게 움직이면 5초
유지 시간이 다시 시작됩니다. 보정이 완료되면 화면에 `Calibration: LOCKED`가 표시됩니다.

## 보조 스켈레톤과 판정 기준 관계

현재 구조에서 보조 스켈레톤은 단순한 그림이 아닙니다.

`utils/guide_skeleton.py`의 `GUIDE_POSES`가 화면에 그려지는 보조 스켈레톤 기준이고, `utils/golf_rules.py`도 이 좌표를 사용해서 사용자 자세와 비교합니다.

보정 이후 최종 점수는 다음 두 기준을 합칩니다.

```text
최종 점수 = 가이드 유사도 55% + 7번 아이언 CaddieSet 지표 점수 45%
```

가이드 자체도 `faceon_i7` 범위에 맞춰 정렬하고, 화면의 관절 허용 영역도 같은
프로필에서 생성합니다. 관절 측정이 불가능하거나 CaddieSet 바깥 관찰 범위를 크게
벗어나거나 가이드 유사도가 낮으면 통과하지 않습니다.

## 좌타/우타 기준

`utils/guide_skeleton.py`의 `SWING_HAND` 값으로 기준을 바꿉니다.

```python
SWING_HAND = "right"
```

- `right`: 우타 기준
- `left`: 좌타 기준

현재 기본값은 우타 기준입니다. 내부 기본 좌표가 한쪽 기준으로 만들어져 있고, 필요하면 좌우 반전해서 사용합니다.

## 샤프트 표시

보조 스켈레톤에는 골프 클럽을 “샤프트 막대기” 형태로 표시합니다.

현재 샤프트는 정교한 클럽 헤드/페이스 방향 인식이 아닙니다. 목적은 사용자가 클럽 방향을 대략 맞출 수 있게 하는 시각 가이드입니다.

샤프트 기준은 두 방식으로 동작합니다.

1. `reference_data/guide_poses/generated_guide_poses.json`에 샤프트 데이터가 있으면 그것을 우선 사용
2. 없으면 `utils/guide_skeleton.py`의 기본 샤프트 좌표 사용

샤프트 자동 추출은 `tools/extract_reference_shafts.py`에서 OpenCV Canny + Hough line 기반으로 시도합니다. 영상/이미지 품질, 배경, 클럽 색상에 따라 실패하거나 엉뚱한 선을 잡을 수 있으므로 overlay 검수가 필요합니다.

## 프로젝트 구조

```text
C:\project\golf-coach
├─ main.py
├─ main_solutions_legacy.py
├─ requirements.txt
├─ pose_landmarker_full.task
├─ README.md
├─ utils
│  ├─ angle_calculator.py
│  ├─ golf_rules.py
│  ├─ guide_skeleton.py
│  └─ pose_drawer.py
├─ tools
│  ├─ build_guide_poses.py
│  ├─ detect_swing_events_mediapipe.py
│  ├─ edit_reference_landmarks.py
│  ├─ extract_reference_poses.py
│  ├─ extract_reference_shafts.py
│  ├─ extract_swing_reference_from_video.py
│  ├─ extract_video_frames.py
│  └─ visualize_reference_poses.py
└─ reference_data
   ├─ raw_videos
   ├─ raw_images
   ├─ extracted_landmarks
   ├─ debug_overlay
   ├─ debug_shaft_overlay
   └─ guide_poses
```

`main_solutions_legacy.py`는 예전 MediaPipe Solutions 방식 테스트 파일입니다. 삭제하지 말고 참고용으로만 둡니다.

## 기존 영상 회귀 분석

`analysis_sessions/stage_audit/pro01~09`에 이미 추출된 관절 좌표 캐시가 있으면,
현재 웹캠의 통합 판정 기준으로 8단계를 다시 평가할 수 있습니다.

```powershell
py -3.12 tools\run_runtime_regression.py
```

결과:

```text
analysis_sessions/runtime_regression/runtime_regression.json
analysis_sessions/runtime_regression/runtime_regression.md
```

검수 완료 상태인 영상은 `swing_stage_ground_truth.json`의 확정 프레임을 사용하고,
나머지는 자동 단계 검출 프레임을 사용합니다. 보고서의 `strict_candidate`와
`lenient_candidate`는 판정 기준 조정 대상을 찾는 진단값입니다. 풀스윙 단일 프레임은
정지 자세 웹캠 조건과 다르고 자동 단계 프레임도 코치 검수 정답이 아니므로, 보고서
결과만으로 통과 기준을 자동 변경하지 않습니다.

## 기준 데이터 생성 흐름

현재는 프로 스윙 영상 또는 캡처 이미지에서 단계별 기준 좌표를 만들어 보조 스켈레톤을 개선하는 방향으로 진행 중입니다.

전체 흐름:

```text
raw_videos 또는 raw_images
→ 단계별 프레임 추출
→ PoseLandmarker로 관절 좌표 JSON 생성
→ 샤프트 후보 추출
→ overlay 이미지로 검수
→ generated_guide_poses.json 생성
→ main.py에서 보조 스켈레톤/판정 기준으로 사용
```

### 로컬 영상에서 자동 추출

```powershell
py -3.12 tools\extract_swing_reference_from_video.py reference_data\raw_videos\pro03_fullswing.mp4 --prefix pro03 --overwrite --event-source mediapipe --sample-step 2
```

이 도구가 하는 일:

- 영상에서 8단계 후보 프레임 자동 선택
- 단계별 JPG 저장
- MediaPipe PoseLandmarker로 관절 JSON 생성
- 샤프트 후보 추출
- 관절/샤프트 overlay 이미지 생성

주의: 현재 8단계 자동 선택은 완벽한 SwingNet 같은 모델이 아니라 MediaPipe 관절 움직임 기반 휴리스틱입니다. 그래서 반드시 overlay를 확인해야 합니다.

### 이미지에서 관절 추출

```powershell
py -3.12 tools\extract_reference_poses.py
```

### 관절 overlay 생성

```powershell
py -3.12 tools\visualize_reference_poses.py
```

결과 위치:

```text
reference_data\debug_overlay
```

### 샤프트 추출

```powershell
py -3.12 tools\extract_reference_shafts.py
```

결과 위치:

```text
reference_data\debug_shaft_overlay
```

### 보조 스켈레톤 기준 생성

```powershell
py -3.12 tools\build_guide_poses.py
```

생성 결과:

```text
reference_data\guide_poses\generated_guide_poses.json
```

이 파일이 있으면 `main.py` 실행 시 기본 좌표보다 이 생성 좌표를 우선 사용합니다.

## 참조 샘플 품질 검사와 검수

참조 JSON을 가이드에 반영하기 전에 자동 품질 검사와 사람 검수를 모두 통과해야 합니다.

자동 검사 및 manifest 갱신:

```powershell
py -3.12 tools\audit_reference_samples.py
```

검수 화면 실행:

```powershell
py -3.12 tools\review_reference_samples.py --review-status pending
```

주요 키:

- `a`: 승인
- `o`: 자동 실패 샘플 강제 승인
- `r`: 제외
- `p`: 보류
- `s`: 자세는 유지하고 샤프트 포함 여부 전환
- `j`/`k` 또는 좌우 방향키: 이전/다음
- `q`: 종료

검수 상태는 `reference_data/review_manifest.json`에 저장합니다. 가이드 생성기는 사람 검수가
`accepted`인 샘플만 사용하며, 자동 검사 `fail`은 `override_auto_fail: true`가 있어야 반영합니다.

최종 런타임 가이드 접촉 시트 생성:

```powershell
py -3.12 tools\visualize_guide_poses.py
```

결과 위치:

```text
reference_data\debug_guide_overlay\generated_guide_contact_sheet.jpg
```

현재 검수 결과는 64개 중 승인 34개, 제외 30개입니다. 승인 데이터가 있는 6단계는 생성 좌표를
사용하고, 올바른 샘플이 없는 `downswing`, `follow_through`는 기본 가이드로 폴백합니다. 샤프트는
최종 시각 검수에서 방향이 맞지 않았던 `backswing`, `impact`도 기본 샤프트로 폴백합니다.

좌표가 들어 있는 `generated_guide_poses.json`과 접촉 시트 이미지는 Git에서 제외합니다. 좌표를
제외한 단계별 반영 수량은 `reference_data/guide_poses/guide_build_report.json`에 기록합니다.

## 데이터와 GitHub 정책

원본 영상, 원본 이미지, 추출 JSON, debug overlay, 생성된 guide pose JSON은 GitHub에 올리지 않는 방향입니다.

현재 `.gitignore`에서 제외하는 주요 데이터:

- `reference_data/raw_videos/**/*.mp4`
- `reference_data/raw_videos/**/*.mov`
- `reference_data/raw_videos/**/*.avi`
- `reference_data/raw_videos/**/*.mkv`
- `reference_data/raw_videos/**/*.webm`
- `reference_data/raw_images/**/*.jpg`
- `reference_data/raw_images/**/*.png`
- `reference_data/extracted_landmarks/**/*.json`
- `reference_data/debug_overlay/**/*`
- `reference_data/debug_shaft_overlay/**/*`
- `reference_data/guide_poses/generated_guide_poses.json`

이유:

- 프로 영상/이미지는 저작권 문제가 있을 수 있음
- AIHub 같은 외부 데이터셋은 이용 약관을 따라야 함
- 추출 JSON도 원본 데이터에서 파생된 데이터일 수 있음
- GitHub에는 코드, 도구, 문서만 올리는 것이 안전함

중요한 코드 변경은 커밋/푸시해도 되지만, 데이터 파일은 명시 요청이 있어도 법적 위험을 먼저 확인해야 합니다.

## AIHub 데이터 검토 메모

검토한 데이터셋:

```text
AIHub 스포츠 사람 동작 (골프)
https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=65
```

확인된 내용:

- 골프 영상 데이터
- 사람, 공, 골프채 객체 정보
- 관절 좌표
- 동작 단계 라벨
- 동작 평가 라벨
- JSON 어노테이션

AIHub 단계 라벨과 우리 단계 매핑:

```text
Adress       -> address
Takeback     -> takeaway
Backswing    -> backswing
Backswingtop -> top
Downswing    -> downswing
Impact       -> impact
Follow       -> follow_through
Finish       -> finish
```

주의할 점:

- 페이지 본문 기준으로는 정면/측면 카메라 방향 라벨이 명확히 확인되지 않았음
- 우리 앱은 현재 정면 웹캠 기준이므로, AIHub 데이터를 그대로 평균내면 안 됨
- 다운로드 후 실제 JSON에 촬영 방향 필드가 있는지 확인해야 함
- 방향 필드가 없다면 정면 샘플만 필터링하는 도구가 필요함

AIHub 데이터를 실제로 쓰게 되면 `tools/import_aihub_golf_dataset.py` 같은 변환기를 새로 만들어, AIHub JSON을 우리 내부 기준 JSON 형식으로 변환하는 것이 다음 작업입니다.

## 현재 한계

현재 프로젝트는 기능 흐름을 만든 MVP 단계입니다.

알려진 한계:

- 보조 스켈레톤 좌표가 아직 실제 골프 레슨 기준으로 충분히 정교하지 않음
- 영상에서 8단계 프레임을 고르는 로직은 휴리스틱이라 오차가 있음
- 샤프트 추출은 배경과 클럽 색상에 따라 실패 가능
- 정면/측면 영상이 섞이면 기준 자세가 망가질 수 있음
- 현재 통합 판정 임계값은 실제 사용자와 골프 코치 검수를 거치지 않은 초기값임
- 샤프트는 아직 화면 가이드이며 실시간 통과 판정에는 포함되지 않음

## 다음 개발 우선순위

권장 순서:

1. 실제 웹캠에서 전신·어드레스·정지 검사의 민감도 확인
2. 다양한 체형의 사용자 어드레스 샘플 수집
3. 골프 코치 검수 결과로 어드레스 임계값 조정
4. 테이크어웨이부터 실시간 샤프트 검출을 통과 판정에 포함
5. 단계별 자동 이동 속도와 피드백 문구 사용성 조정
6. 드라이버 선택 모드와 `faceon_w1` 프로필 추가
7. 나중에 스마트폰 카메라 입력 방식 검토

현재 자동 테스트는 `py -3.12 -m unittest discover -s tests -v`로 실행하며,
보정·정지 검사·통합 판정·자동 진행·회귀 분석을 포함한 115개 테스트가 있습니다.

## 웹캠 데이터셋 수집과 판정 분석

실제 사용자 샘플은 `reference_data/webcam_dataset` 아래에서 개인정보, 비식별 세션,
원본 캡처, 코치 검수와 분석 결과를 분리합니다. 수집·검수 절차는
`reference_data/webcam_dataset/README.md`에 정리되어 있습니다.

기존 프로 영상에서 자동 검출 프레임 주변의 단계별 후보를 추출:

```powershell
python tools\extract_stage_candidates.py
```

현재 런타임 회귀 결과의 단계·관절 병목과 코치 정답의 점수 임계값별
오탐·미탐·정밀도·재현율을 분석:

```powershell
python tools\analyze_scoring_thresholds.py
```

## 개발 원칙

- 처음부터 빠른 풀스윙 실시간 분석으로 가지 않는다.
- 단계별 정지 자세 분석을 먼저 안정화한다.
- 사용자가 보는 보조 스켈레톤과 실제 판정 기준은 같은 데이터를 사용한다.
- 데이터 기반 기준을 만들되, 정면/측면/좌타/우타가 섞이지 않게 관리한다.
- 외부 영상, 이미지, 데이터셋 원본과 파생 JSON은 GitHub에 올리지 않는다.
- 기능을 추가할 때는 `main.py`를 너무 복잡하게 만들지 말고 `utils`와 `tools`로 분리한다.
