---- Base ----

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 
PYTHONUNBUFFERED=1 
PIP_NO_CACHE_DIR=1

WORKDIR /app

System deps

RUN apt-get update && apt-get install -y --no-install-recommends 
curl ca-certificates && 
rm -rf /var/lib/apt/lists/*

---- Deps layer ----

COPY backend/requirements.txt /tmp/requirements.txt RUN python -m venv /opt/venv && 
. /opt/venv/bin/activate && 
pip install -r /tmp/requirements.txt

ENV PATH="/opt/venv/bin:$PATH"

---- App layer ----

COPY backend/src /app COPY backend/.env.example /app/.env.example

EXPOSE 8081

HEALTHCHECK --interval=20s --timeout=5s --retries=5 
CMD curl -fsS http://127.0.0.1:8081/health || exit 1

CMD ["uvicorn", "src.predictive_api:app", "--host", "0.0.0.0", "--port", "8081"]
