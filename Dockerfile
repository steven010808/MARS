FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/src \
    MARS_CONFIG=/app/configs/config.yaml

WORKDIR /app

COPY pyproject.toml ./
COPY requirements.txt ./

RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY apps ./apps
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY README.md ./

RUN pip install --no-deps -e .

RUN mkdir -p data/raw data/processed artifacts/search artifacts/recsys artifacts/reports artifacts/registry logs docs

EXPOSE 8000 8501
