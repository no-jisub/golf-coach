# CaddieSet integration

이 디렉터리는 CaddieSet 원본을 현재 골프 코치의 8단계 평가 항목으로 변환한 결과를 관리합니다.

## 데이터 출처

- 공식 저장소: <https://github.com/damilab/CaddieSet>
- 고정 커밋: `3c73d9d40580bb8a5a10711ad1fa10735a205ffe`
- 라이선스: MIT
- 원본 규모: 1,757샷, 골퍼 8명
- 촬영 방향: FACEON 924샷, DTL 833샷

원본 CSV의 관절 기반 지표는 비전 모델로 자동 추출된 값입니다. 따라서 모션캡처 정답 좌표가 아니며, 아래 범위를 그대로 의학적·생체역학적 합격 기준으로 사용해서는 안 됩니다.

## 내려받기와 변환

```powershell
py -3.12 tools\download_caddieset.py
py -3.12 tools\convert_caddieset.py
```

원본은 `reference_data/caddieset/source`에 내려받으며 Git에서 제외합니다. 다운로드 도구는 고정된 공식 커밋의 CSV와 라이선스를 받고 SHA-256을 확인합니다.

변환 결과는 `reference_data/caddieset/evaluation_profiles.json`입니다.

## 8단계 매핑

| CaddieSet 인덱스 | 현재 앱 단계 |
| --- | --- |
| 0 | `address` |
| 1 | `takeaway` |
| 2 | `backswing` |
| 3 | `top` |
| 4 | `downswing` |
| 5 | `impact` |
| 6 | `follow_through` |
| 7 | `finish` |

생성되는 기본 프로필은 다음과 같습니다.

- `faceon_all`: 현재 정면 앱에서 우선 참고할 전체 클럽 프로필
- `faceon_w1`, `faceon_i7`: 정면 드라이버·7번 아이언 프로필
- `dtl_all`, `dtl_w1`, `dtl_i7`: 향후 후방 촬영 모드를 위한 프로필

현재 앱은 클럽 종류를 입력받지 않으므로 `faceon_all`을 기본 프로필로 지정합니다. 실제 판정에 연결할 때는 클럽 선택 기능을 먼저 추가하고 가능한 한 `faceon_w1` 또는 `faceon_i7`처럼 같은 클럽끼리 비교해야 합니다.

## 참조 샷과 통계 기준

CaddieSet 논문의 직진 샷 분류 기준을 따라 다음 샷만 참조 통계에 포함합니다.

- `abs(DirectionAngle) <= 6도`
- `abs(SpinAxis) <= 10도`

한 골퍼의 샷 수가 다른 골퍼보다 많아 기준값을 지배하지 않도록 `target`은 골퍼별 중앙값을 먼저 구한 뒤 그 중앙값들의 중앙값으로 계산합니다.

- `target`: 골퍼 균형 중앙값
- `observed_reference_range`: 선택된 샷의 10~90 백분위 범위
- `observed_outer_range`: 선택된 샷의 5~95 백분위 범위
- `golfer_median_range`: 골퍼별 중앙값의 10~90 백분위 범위

이 값들은 관찰된 증거 범위입니다. 런타임의 통과·실패 기준으로 사용하기 전 골프 코치 검수와 사용자 체형·클럽·촬영 각도별 검증이 필요합니다.
