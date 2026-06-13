# MARS Submission Documents

이 폴더는 최종 제출과 평가 확인에 필요한 공식 문서만 모아 둔 문서 인덱스입니다. 실험 중간 메모, 임시 스크린샷, 개인 작업 노트는 포함하지 않는 것을 기준으로 합니다.

| Document | Purpose |
| --- | --- |
| `architecture.md` | Docker Compose, API, dashboard, simulator, worker를 포함한 전체 시스템 구조 |
| `api.md` | `/api/search`, `/api/recommend`, `/api/events`, `/api/metrics`, `/api/ab/*` 계약 |
| `evaluation_report.md` | 검색, 추천, A/B 테스트, Continuous Training 평가 결과 |
| `code_traceability.md` | 명세서 요구사항과 실제 구현 파일 대응표 |
| `runtime_bundle_guide.md` | 다른 PC에 data/artifact를 전달하고 실행하는 방법 |

## Recommended Review Order

1. `architecture.md`로 전체 구성과 데이터 흐름을 먼저 확인합니다.
2. `api.md`로 외부 채점 또는 동료 검증 시 호출할 endpoint를 확인합니다.
3. `evaluation_report.md`로 정량 지표와 산출 방식을 확인합니다.
4. `code_traceability.md`로 요구사항별 구현 위치를 확인합니다.
5. `runtime_bundle_guide.md`로 다른 PC에서 실행할 때 필요한 파일과 절차를 확인합니다.
