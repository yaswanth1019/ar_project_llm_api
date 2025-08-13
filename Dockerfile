# Use slim Python base image
FROM python:3.10-slim

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --no-deps \
    fastapi uvicorn[standard] python-multipart python-dotenv \
    && pip install --no-cache-dir --no-deps \
    langchain langchain-community langchain-core langchain-text-splitters langchain_groq \
    && pip install --no-cache-dir --no-deps \
    sentence-transformers scikit-learn \
    && pip install --no-cache-dir --no-deps \
    faiss-cpu gTTS vosk websockets soundfile \
    && pip cache purge

# Create necessary directories
RUN mkdir -p uploads responses models

# Copy app files
COPY . .

# Set environment variables for memory optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/hf_cache
ENV TOKENIZERS_PARALLELISM=false

# Expose port
EXPOSE $PORT

# Health check with longer timeout for model loading
HEALTHCHECK --interval=45s --timeout=45s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT:-10000}/health || exit 1

# Start the app
CMD python -m uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 60
