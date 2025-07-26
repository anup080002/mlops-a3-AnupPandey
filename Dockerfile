# syntax=docker/dockerfile:1.6
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 1. Install deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy code + trained artefacts (models/* created in CI)
COPY src/ src/
COPY models/ models/

# 3. Default smoke‑test entrypoint
ENTRYPOINT ["python", "-m", "src.predict"]
