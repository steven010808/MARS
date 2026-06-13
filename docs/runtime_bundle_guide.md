# Runtime Bundle Guide

GitHub에는 코드와 문서만 올린다. Full-scale 실행에 필요한 `data/processed`와 `artifacts`는 용량이 크기 때문에 runtime bundle로 따로 전달한다.

## 1. 권장 실행 방식

다른 PC에서 가장 빠르게 실행하려면 repository를 clone한 뒤 runtime bundle을 같은 루트에 풀면 된다.

```powershell
git clone <repo-url>
cd <repo-folder>
Expand-Archive <path-to>\mars_runtime_bundle.zip -DestinationPath .
docker compose up --build
```

접속 주소:

| 서비스 | 주소 |
| --- | --- |
| Dashboard | `http://localhost:8501` |
| API docs | `http://localhost:8000/docs` |
| Health check | `http://localhost:8000/healthz` |

## 2. Bundle 생성

기본 bundle은 실행과 정량 검증에 필요한 processed data와 artifact를 포함한다.

```powershell
python -m scripts.packaging.package_runtime_bundle --dry-run
python -m scripts.packaging.package_runtime_bundle --output dist\mars_runtime_bundle.zip
```

상품 이미지 preview까지 포함하려면 `--include-images`를 사용한다.

```powershell
python -m scripts.packaging.package_runtime_bundle `
  --output dist\mars_runtime_bundle_with_images.zip `
  --include-images
```

기본 bundle에 포함되는 경로:

| 경로 | 내용 |
| --- | --- |
| `data/processed/` | full-scale parquet table |
| `artifacts/search/` | CLIP embedding, FAISS index, query behavior model |
| `artifacts/recsys/` | 추천 artifact |
| `artifacts/reports/` | 평가 결과 |
| `artifacts/registry/` | active model version |

## 3. 원본 데이터로 재생성

Runtime bundle 없이 재생성하려면 아래 파일을 먼저 준비한다.

| 파일/폴더 | 위치 |
| --- | --- |
| Clean H&M 50K product master | `data/external/hm/processed/hm_products_master_clean_50k.csv` |
| Microsoft H&M search queries | `data/external/hnm_search/raw/queries.csv` |
| Microsoft H&M search qrels | `data/external/hnm_search/raw/qrels.csv` |
| H&M product images | `data/external/hm/raw/images/` |

`hm_products_master_clean_50k.csv`가 없으면 원본 master에서 먼저 생성한다.

```powershell
python -m scripts.artifacts.build_clean_hm_catalog_50k `
  --input data/external/hm/processed/hm_products_master.csv `
  --output data/external/hm/processed/hm_products_master_clean_50k.csv `
  --image-root data/external/hm/raw/images `
  --hnm-search-qrels data/external/hnm_search/raw/qrels.csv
```

그 다음 runtime data와 artifact를 다시 만든다.

```powershell
python -m scripts.runtime.bootstrap_runtime `
  --config configs/config.yaml `
  --mode full `
  --rebuild-raw `
  --clean-processed `
  --clean-artifacts `
  --register
```

## 4. CLIP 모델 캐시

검색 encoder는 `openai/clip-vit-base-patch32`를 사용한다. Docker Compose는 fresh PC에서도 모델을 받을 수 있도록 `MARS_CLIP_LOCAL_FILES_ONLY=0`으로 실행한다.

인터넷 연결이 없는 환경에서는 Hugging Face cache를 미리 준비해야 한다.

기본 Windows cache 위치:

```text
%USERPROFILE%\.cache\huggingface
```

## 5. 실행 확인

```powershell
curl http://localhost:8000/healthz
python -m scripts.checks.smoke_api --base-url http://localhost:8000 --timeout 240
```

정상 실행 후 dashboard에서 Search, Recommendation, Experiments, Live Logs, QA Gate 탭을 확인한다.
