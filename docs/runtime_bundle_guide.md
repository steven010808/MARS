# Runtime Bundle Guide

This document explains how to run MARS on a fresh machine in two different ways.

- Test 1: run with prepared processed data and artifacts.
- Test 2: rebuild from raw H&M files.

The code branch for both tests is:

```powershell
git clone https://github.com/steven010808/MARS.git
cd MARS
git checkout raw-rebuild-runtime-pipeline
```

## Test 1. Prepared Runtime Bundle

Use this path when the reviewer receives a Google Drive runtime bundle. This is
the fastest path because it reuses already generated full-scale Parquet tables,
FAISS indexes, model artifacts, reports, and registry state.

### 1. Place The Bundle

Extract or copy the runtime bundle at the repository root.

After extraction, these paths must exist:

| Runtime path | Purpose |
| --- | --- |
| `data/processed/` | Full-scale Parquet tables |
| `artifacts/search/` | Search embeddings, FAISS indexes, query behavior model |
| `artifacts/recsys/` | Recommendation artifacts |
| `artifacts/reports/` | Metrics report |
| `artifacts/registry/` | Active model registry |

### 2. Run

```powershell
docker compose up --build
```

### 3. Check

| Service | URL |
| --- | --- |
| Dashboard | `http://localhost:8501` |
| API docs | `http://localhost:8000/docs` |
| Health check | `http://localhost:8000/healthz` |

Smoke test:

```powershell
python -m scripts.checks.smoke_api --base-url http://localhost:8000 --timeout 240
```

## Test 2. Raw Data Rebuild

Use this path when the reviewer wants to verify that the repository can rebuild
the submitted data from raw inputs.

### 1. Place Raw Inputs

Put the files at the exact paths below:

| Input | Expected path |
| --- | --- |
| H&M articles | `data/external/hm/raw/articles.csv` |
| H&M transactions | `data/external/hm/raw/transactions_train.csv` |
| H&M images | `data/external/hm/raw/images/` |
| Microsoft H&M search queries | `data/external/hnm_search/raw/queries.csv` |
| Microsoft H&M search qrels | `data/external/hnm_search/raw/qrels.csv` |

### 2. Rebuild

```powershell
python -m scripts.runtime.bootstrap_runtime `
  --config configs/config.yaml `
  --mode full `
  --rebuild-raw `
  --clean-processed `
  --clean-artifacts `
  --register
```

For a faster sanity check:

```powershell
python -m scripts.runtime.bootstrap_runtime `
  --config configs/config.yaml `
  --mode dev `
  --rebuild-raw `
  --clean-processed `
  --clean-artifacts `
  --encoder fallback `
  --register
```

### 3. Rebuild Flow

The bootstrap creates missing files in this order:

```text
H&M raw CSV/images
-> data/external/hm/processed/hm_products_master.csv
-> data/external/hm/processed/hm_products_master_clean_50k.csv
-> data/raw/products.csv, users.csv, events.csv
-> data/processed/*.parquet
-> artifacts/search, artifacts/recsys, artifacts/reports, artifacts/registry
```

### 4. Run

```powershell
docker compose up --build
```

## Runtime Bundle Creation

Create a bundle from a machine that already has `data/processed` and `artifacts`.

```powershell
python -m scripts.packaging.package_runtime_bundle --dry-run
python -m scripts.packaging.package_runtime_bundle --output dist\mars_runtime_bundle.zip
```

The default bundle does not include raw H&M images. If dashboard product previews
must be distributed together, build a much larger bundle with images:

```powershell
python -m scripts.packaging.package_runtime_bundle `
  --output dist\mars_runtime_bundle_with_images.zip `
  --include-images
```

## Verified Scale

The raw rebuild path was verified at full scale with these row counts:

| Table | Rows |
| --- | ---: |
| Products | 50,000 |
| Users | 10,000 |
| Events | 1,000,000 |
| Sessions | 240,836 |
| Search queries | 193,064 |
| Train events | 799,998 |
| Valid events | 99,997 |
| Test events | 100,005 |

The regenerated `hm_products_master_clean_50k.csv` matched the existing final
catalog exactly by row count, columns, product order, and DataFrame values.
