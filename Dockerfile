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

# Download Vosk model at build time (faster startup)
RUN wget -O /tmp/vosk.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip \
    && unzip /tmp/vosk.zip -d models \
    && rm /tmp/vosk.zip

# Optional: tell Docker this container listens on port 10000 (Render ignores this)
EXPOSE 10000

# Start app with dynamic $PORT from Render
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
