#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not running. Start Docker Desktop first, then rerun this script." >&2
  exit 1
fi

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
