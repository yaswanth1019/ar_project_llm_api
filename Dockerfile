FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    espeak \
    espeak-data \
    libespeak1 \
    libespeak-dev \
    festival* \
    alsa-utils \
    pulseaudio \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Create necessary directories
RUN mkdir -p uploads responses models

# Download Vosk model
RUN if [ ! -d "models/vosk-model-small-en-us-0.15" ]; then \
    wget -O models/vosk-model-small-en-us-0.15.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip && \
    cd models && \
    unzip vosk-model-small-en-us-0.15.zip && \
    rm vosk-model-small-en-us-0.15.zip; \
    fi

# Tell Render this container listens on $PORT
EXPOSE 10000

# Start the app
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
