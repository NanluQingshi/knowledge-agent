# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

ARG POETRY_EXTRAS=loaders

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VIRTUALENVS_IN_PROJECT=true

WORKDIR /app

RUN pip install --no-cache-dir "poetry>=1.8,<2.0"

COPY pyproject.toml ./
RUN if [ -n "$POETRY_EXTRAS" ]; then \
        poetry install --no-ansi --no-root --only main -E "$POETRY_EXTRAS"; \
    else \
        poetry install --no-ansi --no-root --only main; \
    fi

COPY . .
RUN poetry install --no-ansi --only-root

FROM python:3.11-slim

ARG INSTALL_OCR=false

ENV HOME=/home/knowledge-agent \
    PATH=/app/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/app/.venv

WORKDIR /app

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends libgomp1; \
    if [ "$INSTALL_OCR" = "true" ]; then \
        apt-get install -y --no-install-recommends \
            tesseract-ocr \
            tesseract-ocr-chi-sim \
            tesseract-ocr-eng; \
    fi; \
    rm -rf /var/lib/apt/lists/*; \
    groupadd --gid 10001 knowledge-agent; \
    useradd --uid 10001 --gid knowledge-agent --create-home knowledge-agent

COPY --chown=knowledge-agent:knowledge-agent --from=builder /app /app

RUN mkdir -p /app/data \
    && chown -R knowledge-agent:knowledge-agent /app/data

USER knowledge-agent

EXPOSE 8000 7860

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=5).read()"]

CMD ["ka", "webui"]
