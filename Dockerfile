# ── Build stage ──
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (saves ~1.5GB vs GPU version)
RUN pip install --no-cache-dir --prefix=/install \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (with all sub-deps)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install \
    medmnist>=3.0.0 \
    opacus>=1.4.0 \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    streamlit>=1.28.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    pyyaml>=6.0

# ── Runtime stage ──
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project source code
COPY config/ config/
COPY src/ src/
COPY api/ api/
COPY dashboard/ dashboard/
COPY scripts/ scripts/

# Create data directory for dataset downloads
RUN mkdir -p data experiments

# Default: start the API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
