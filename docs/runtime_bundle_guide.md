# Runtime Bundle Guide

작성일: 2026-05-12 KST

## 1. 왜 번들이 필요한가

GitHub에는 코드, 설정, 문서, 테스트를 올리고 `data/`, `artifacts/`, `logs/` 같은 대용량 산출물은 일반적으로 올리지 않는다. H&M 원본 이미지와 거래 데이터는 라이선스와 용량 이슈도 있으므로 공개 저장소에 그대로 올리지 않는 편이 안전하다.

그래서 다른 컴퓨터에서 처음 받은 것처럼 실행하려면 다음 두 가지 중 하나를 선택한다.

## 2. 권장 방식: 소스 + 런타임 번들

이 방식은 다른 PC에서 full-scale artifact를 다시 만들지 않고 바로 Docker Compose로 실행하는 제출/시연용 방식이다.

보내는 것:

- GitHub repository
- `dist/mars_runtime_bundle.zip`
- 이미지 미리보기까지 필요하면 `dist/mars_runtime_bundle_with_images.zip`

번들 생성:

```powershell
cd <repository-root>
python -m scripts.packaging.package_runtime_bundle --dry-run
python -m scripts.packaging.package_runtime_bundle --output dist/mars_runtime_bundle.zip
```

이미지 미리보기 포함:

```powershell
python -m scripts.packaging.package_runtime_bundle --output dist/mars_runtime_bundle_with_images.zip --include-images
```

받는 PC에서 실행:

```powershell
git clone <repo-url>
cd <repo-folder>
Expand-Archive <path-to>\mars_runtime_bundle.zip -DestinationPath .
docker compose up --build
```

기본 번들은 원본 H&M 이미지 파일을 포함하지 않는다. API serving과 정량 지표 검증은 사전 계산된 embedding/index artifact로 동작한다. Dashboard 상품 preview는 실제 이미지 파일이 있으면 사진을 보여주고, 없으면 runtime-light placeholder와 원래 `image_path`를 보여준다.

검증:

```powershell
curl http://localhost:8000/healthz
curl http://localhost:8501
```

브라우저:

- API docs: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

## 3. 대안 방식: 원본 데이터로 재생성

이 방식은 H&M raw/processed data와 이미지가 같은 경로에 준비되어 있어야 하며, CPU 환경에서는 시간이 오래 걸린다.

```powershell
python -m scripts.runtime.bootstrap_runtime --config configs/config.yaml --mode full --rebuild-raw --clean-processed --clean-artifacts --register
docker compose up --build
```

권장하지 않는 경우:

- 발표 직전
- 다른 PC의 Docker/Python cache가 비어 있는 경우
- CLIP model download가 느린 네트워크 환경

## 4. 번들에 포함되는 것

기본 번들:

- `data/processed/`
- `artifacts/search/`
- `artifacts/recsys/`
- `artifacts/reports/`
- `artifacts/registry/`

`--include-images` 번들:

- 기본 번들 전체
- `data/processed/products.parquet`의 `image_path`가 참조하는 50K 상품 이미지

## 5. Runtime-Light 패키지 구성

Drive로 공유하는 runtime-light 패키지는 Docker 실행과 정량 검증에 필요한 파일만 포함한다.

포함:

- `apps/`, `src/`, `scripts/`, `configs/`, `docs/`, `tests/`
- `data/processed/`
- `artifacts/search/`, `artifacts/recsys/`, `artifacts/reports/`, `artifacts/registry/`
- `docker-compose.yml`, `Dockerfile`, `README.md`, `requirements.txt`

제외:

- `data/external/`: H&M 원본 이미지와 외부 raw dataset
- `data/raw/`: simulator 중간 CSV
- `logs/`: 실행 중 생성되는 runtime log
- `.venv/`, `.git/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`

## 6. 주의사항

- H&M raw images/transactions를 공개 GitHub에 올리지 말 것.
- 기본 제출 폴더와 기본 runtime bundle에는 `data/raw/`, `data/external/`, `logs/`를 포함하지 않는다.
- active model version은 live worker가 실행되면 계속 증가할 수 있다.
- 다른 PC에서 version이 달라도 `data scale`, `artifact readiness`, `metric target status`가 통과하면 정상이다.
- 이미지 포함 번들은 매우 커질 수 있으므로 Google Drive, OneDrive, 외장 SSD 등으로 전달하는 것이 현실적이다.


