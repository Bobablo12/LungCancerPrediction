# Multi-arch friendly base; 3.11 works with TF 2.20 wheels
FROM python:3.11-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=10000 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_VISIBLE_DEVICES=-1

WORKDIR /app

# System deps (Pillow image codecs, certificates)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg62-turbo \
    zlib1g \
    libpng16-16 \
    libtiff6 \
    libopenjp2-7 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt

# Copy app code and model
COPY . .

# Expose port for cloud platforms
EXPOSE 10000

# Default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
