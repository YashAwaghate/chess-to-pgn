FROM python:3.10-slim

# System libs required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgles2 \
        libegl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
# Install PyTorch CPU-only build first (smaller image, no CUDA)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/

# Create writable data/logs/models dirs (trained model can be mounted or downloaded at runtime)
RUN mkdir -p data/sessions logs models

EXPOSE 8000

# hand_landmarker.task is downloaded automatically by hand_detector.py on first run
CMD ["python", "src/capture/server.py"]
