# Webcam dataset storage

이 폴더는 실제 웹캠 자세 샘플을 참가자·세션·촬영 조건별로 수집하기 위한 로컬 데이터 영역입니다.

## 개인정보 분리 원칙

- `private/`: 이름, 연락 참조, 동의서 참조처럼 직접 식별 가능한 정보
- `sessions/`: 익명 참가자 ID, 체형 구간, 촬영 조건
- `captures/`: 원본·오버레이 이미지, 관절 좌표와 런타임 판정
- `reviews/`: 코치 검수 진행 상태
- `exports/`: 비식별 정답 데이터와 분석 결과
- `active_session.json`: 현재 `main.py`가 사용할 로컬 세션 포인터

위 데이터 폴더와 JSON은 모두 `.gitignore`로 제외합니다. 이 README와 코드만 Git에 포함됩니다.
정답 데이터도 자동으로 공개 저장소에 포함하지 않으며, 비식별 여부를 사람이 확인한 뒤 명시적으로 별도 경로에 복사해야 합니다.

## 수집 세션 시작

```powershell
python tools\configure_webcam_collection.py `
  --consent-confirmed `
  --height-band 170_179 `
  --body-build average `
  --experience-level beginner `
  --handedness right `
  --view FACEON `
  --club-type I7 `
  --distance-band recommended `
  --lighting normal `
  --background plain `
  --width 1280 `
  --height 720
```

출력된 익명 참가자 ID를 같은 참가자의 후속 촬영에서 `--participant-id`로 재사용합니다.
그 다음 `main.py`를 실행하면 `s`, `g`, `b`로 저장한 샘플에 참가자·세션·촬영 조건이 함께 기록됩니다.

보정 완료 뒤 저장한 샘플은 `golf-coach-runtime-sample-v3` 형식입니다. 이 형식은
화면 한 장뿐 아니라 실제 판정에 사용한 전체 관절 시간창, 5fps 대표 원본 프레임,
평균 관절, 캘리브레이션 프로필, 런타임 임계값, 패키지 버전, Git SHA와
MediaPipe·가이드·CaddieSet 파일 SHA-256을 함께 기록합니다.

저장 당시 결과를 현재 코드로 다시 판정:

```powershell
python tools\replay_runtime_samples.py
```

결과는 `analysis_sessions/runtime_sample_replay/replay_report.json`에 저장되며,
통과 여부 변경과 점수 변화, 모델·가이드·코드 버전 차이를 분리해서 보여줍니다.

활성 세션 해제:

```powershell
python tools\configure_webcam_collection.py --deactivate
```

## 코치 검수와 정답 변환

검수 대상과 우선순위만 먼저 확인:

```powershell
python tools\review_webcam_samples.py --list
```

이미지 검수 화면:

```powershell
python tools\review_webcam_samples.py --reviewer coach-id
```

- `g`: 좋은 자세
- `b`: 나쁜 자세
- `u`: 판단 보류
- `p`: 미검수로 되돌리기
- `j` / `k`: 이전 / 다음
- `q`: 종료

수집자 라벨과 런타임 판정이 충돌하거나 70점 임계값에 가까운 샘플이 먼저 표시됩니다.
코치 검수를 비식별 정답 JSON으로 변환:

```powershell
python tools\export_coach_ground_truth.py
```

정답 JSON은 이후 `tools/analyze_scoring_thresholds.py`와 통합 회귀 파이프라인에서
임계값별 오탐·미탐 계산에 자동으로 사용됩니다.
