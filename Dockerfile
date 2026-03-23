FROM python:3.10-slim

# System libs required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/

# Create writable data/logs dirs (sessions saved here when S3 is not configured)
RUN mkdir -p data/sessions logs

EXPOSE 8000

# hand_landmarker.task is downloaded automatically by hand_detector.py on first run
CMD ["python", "src/capture/server.py"]
