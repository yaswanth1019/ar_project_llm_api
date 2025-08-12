FROM python:3.10-slim

# Install system dependencies and clean up in one layer
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies and clean pip cache
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Create necessary directories
RUN mkdir -p uploads responses models

# Download Vosk model, extract, and cleanup zip in one layer
RUN cd models && \
    wget -q -O vosk-model.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip && \
    unzip -q vosk-model.zip && \
    rm -f vosk-model.zip && \
    echo "Vosk model installed, zip file removed" && \
    ls -la vosk-model-small-en-us-0.15/

# Copy app files (do this after model download to leverage Docker cache)
COPY . .

# Set environment variables to reduce memory usage
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/hf_cache

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-10000}/health || exit 1

# Start the app with explicit port binding and immediate startup
CMD echo "Starting on port $PORT" && python -m uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --access-log --log-level info --timeout-keep-alive 30
