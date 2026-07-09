









FROM python:3.10-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.10-slim

# Runtime deps for opencv-headless (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libxcb1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only installed packages from builder (no pip cache, no build tools)
COPY --from=builder /install /usr/local

COPY src/ ./src/
RUN mkdir -p /app/models

RUN useradd -m appuser && chown -R appuser /app
USER appuser

CMD ["uvicorn", "src.web.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]