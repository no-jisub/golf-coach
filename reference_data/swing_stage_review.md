# 스윙 8단계 정답 검수 절차

자동 단계 감지 정확도는 사람이 확인한 정답 프레임만 기준으로 계산합니다. `pending` 영상을 자동 감지 결과와 같다고 간주하지 않습니다.

## 1. 검수 자료 생성

프로젝트 루트에서 실행합니다.

```powershell
py -3.12 tools\audit_swing_stage_detection.py
```

영상별 검수 시트는 `analysis_sessions/stage_audit/<영상 ID>/stage_contact_sheet.jpg`에 생성됩니다. 각 단계마다 자동 선택 프레임과 앞·뒤 후보 프레임을 함께 보여 줍니다.

## 2. 정답 프레임 기록

`reference_data/swing_stage_ground_truth.json`에서 해당 영상의 `events`에 0부터 시작하는 프레임 번호를 기록합니다.

- address부터 finish까지 8개 프레임을 모두 기록합니다.
- 프레임은 중복 없이 시간순으로 증가해야 합니다.
- 확인이 끝나면 `review_status`를 `reviewed`로 바꿉니다.
- 검수자와 검수 시각은 `reviewed_by`, `reviewed_at`에 기록합니다.
- 영상이 불완전하거나 정면 영상이 아니면 `excluded`로 바꾸고 `note`에 사유를 적습니다.

검수 시트의 세 후보 사이에 정확한 순간이 없다면 원본 영상을 프레임 단위로 확인해 번호를 기록합니다.

## 3. 정확도 보고서 갱신

같은 명령을 다시 실행하면 `analysis_sessions/stage_audit/stage_accuracy_report.json`이 갱신됩니다.

기본 허용 오차는 150ms입니다. 다른 기준을 시험하려면 다음처럼 실행합니다.

```powershell
py -3.12 tools\audit_swing_stage_detection.py --tolerance-ms 100
```

보고서에는 영상별·단계별 절대 프레임 오차, 시간 오차, 허용 오차 내 비율이 들어갑니다. 검수 완료 영상이 없으면 정확도 값은 `null`이며 검수 대기 수만 표시됩니다.
