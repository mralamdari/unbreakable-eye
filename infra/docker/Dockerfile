# STAGE 1: Builder (Compiling dependencies)
FROM python:3.10-slim as builder

WORKDIR /app

# Install system build tools
RUN apt-get update && apt-get install -y gcc

# Install Python Manager
RUN pip install poetry

COPY pyproject.toml poetry.lock ./

# Export to requirements.txt (easier for slim builds)
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# STAGE 2: Runtime (The actual container)
FROM python:3.10-slim

WORKDIR /app

# Install OpenCV system dependencies (Critical for cv2-headless)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Your Code
COPY src/ ./src/
COPY models/ ./models/ 
# ^ Note: Only copy models if they are small. 
# For big models, mount a volume instead!

# Create non-root user
RUN useradd -m appuser
USER appuser

# Environment Variables (Defaults)
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the Server
CMD ["python", "-m", "src.server.main"]
