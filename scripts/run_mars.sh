#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${USERPROFILE:-}" ]] && command -v cmd.exe >/dev/null 2>&1 && command -v wslpath >/dev/null 2>&1; then
  WIN_USERPROFILE="$(cmd.exe /C echo %USERPROFILE% 2>/dev/null | tr -d '\r')"
  if [[ -n "$WIN_USERPROFILE" ]]; then
    export USERPROFILE="$(wslpath -u "$WIN_USERPROFILE")"
  fi
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not running. Start Docker Desktop first, then rerun this script." >&2
  exit 1
fi

DATA_DIR="${MARS_HOST_DATA_DIR:-$ROOT_DIR/data}"
DATA_DIR_FOR_CHECK="$DATA_DIR"
if [[ "$DATA_DIR_FOR_CHECK" =~ ^[A-Za-z]:\\ ]] && command -v wslpath >/dev/null 2>&1; then
  DATA_DIR_FOR_CHECK="$(wslpath -u "$DATA_DIR_FOR_CHECK")"
elif [[ "$DATA_DIR_FOR_CHECK" != /* ]]; then
  DATA_DIR_FOR_CHECK="$ROOT_DIR/$DATA_DIR_FOR_CHECK"
fi

HAS_PROCESSED_DATA=0
if [[ -f "$DATA_DIR_FOR_CHECK/processed/manifest.json" ]] \
  && [[ -f "$DATA_DIR_FOR_CHECK/processed/products.parquet" ]] \
  && [[ -f "$DATA_DIR_FOR_CHECK/processed/search_queries.parquet" ]]; then
  HAS_PROCESSED_DATA=1
fi

HAS_REBUILD_INPUTS=0
if [[ -f "$DATA_DIR_FOR_CHECK/external/hm/processed/hm_products_master_clean_50k.csv" ]] \
  && [[ -f "$DATA_DIR_FOR_CHECK/external/hnm_search/raw/queries.csv" ]] \
  && [[ -f "$DATA_DIR_FOR_CHECK/external/hnm_search/raw/qrels.csv" ]]; then
  HAS_REBUILD_INPUTS=1
fi

if [[ "$HAS_PROCESSED_DATA" != "1" && "$HAS_REBUILD_INPUTS" != "1" ]]; then
  cat >&2 <<EOF
MARS runtime data was not found.

This GitHub repository does not include the large H&M data files.
Prepare one of the following before running:

1. Put a runtime data bundle under:
   $ROOT_DIR/data

2. Or point MARS_HOST_DATA_DIR to an existing data directory, for example:
   MARS_HOST_DATA_DIR=/mnt/f/롱스톤/mars/data bash scripts/run_mars.sh

Expected processed-data files include:
  data/processed/manifest.json
  data/processed/products.parquet
  data/processed/search_queries.parquet

Expected rebuild-input files include:
  data/external/hm/processed/hm_products_master_clean_50k.csv
  data/external/hnm_search/raw/queries.csv
  data/external/hnm_search/raw/qrels.csv
EOF
  exit 1
fi

echo "Using MARS data directory: $DATA_DIR"

docker compose up -d --build
docker compose ps

cat <<'EOF'

MARS is starting.

FastAPI:             http://localhost:8000
API Swagger:         http://localhost:8000/docs
Streamlit Dashboard: http://localhost:8501

Useful checks:
  docker compose logs -f api dashboard
  python -m scripts.checks.smoke_api --base-url http://localhost:8000 --timeout 240
EOF
