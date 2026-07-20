# =============================================================================
# DistilKit — Docker Image
#
# Build:
#   docker build -t distilkit .
#
# Run (web GUI):
#   docker run -p 7860:7860 distilkit
#
# Run (CLI — train ResNet18 → MiniCNN on CIFAR-10):
#   docker run --rm distilkit distilkit train --epochs 5
#
# Run (CLI — benchmark):
#   docker run --rm -v $(pwd)/checkpoints:/app/checkpoints distilkit \
#     distilkit benchmark --model checkpoints/student.onnx
#
# Attach a volume for persisted data/checkpoints:
#   docker run -p 7860:7860 -v distilkit_data:/app/data \
#     -v distilkit_checkpoints:/app/checkpoints distilkit
#
# Stats (approx):
#   Image size: ~1.8 GB (compressed ~800 MB)
#   Python: 3.12-slim, CPU-only PyTorch
# =============================================================================

FROM python:3.12-slim AS builder

LABEL org.opencontainers.image.title="DistilKit"
LABEL org.opencontainers.image.description="Knowledge Distillation Framework"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.licenses="MIT"

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install build-time system deps (only what's needed for pip wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies (skip .pyc compilation for smaller image)
RUN pip install --upgrade pip && \
    pip install --no-compile -r requirements.txt

# ── Second stage: runtime ────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONOPTIMIZE=2 \
    DEVICE=cpu

WORKDIR /app

# Runtime system deps (aria2c/wget/curl for faster dataset downloads)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    libomp-dev \
    aria2 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy only the installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Strip .pyc and __pycache__ from site-packages to save space
RUN find /usr/local/lib/python3.12/site-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Copy source code (no tests/ — not needed at runtime)
COPY src/ src/
COPY requirements.txt .
COPY pyproject.toml .
COPY README.md .

# Default directories for datasets and checkpoints
VOLUME ["/app/data", "/app/checkpoints"]

EXPOSE 7860

# Default: launch the web GUI
CMD ["python", "-m", "src.webapp"]
