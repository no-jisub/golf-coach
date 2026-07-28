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

활성 세션 해제:

```powershell
python tools\configure_webcam_collection.py --deactivate
```
