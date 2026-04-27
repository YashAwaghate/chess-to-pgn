FROM python:3.10-slim

# System libs required by OpenCV (headless) and MediaPipe
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

# Install PyTorch CPU-only build first (smaller image, no CUDA)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Force-purge any cached opencv variants before installing requirements,
# so the version pinned in requirements.txt is the one that ends up on PATH.
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless || true

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source AND trained models
COPY src/ ./src/
COPY models/ ./models/

# Writable runtime dirs
RUN mkdir -p data/sessions logs

EXPOSE 8000

# hand_landmarker.task is downloaded automatically by hand_detector.py on first run
CMD ["python", "src/capture/server.py"]
