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
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU-only build first (smaller image, no CUDA)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Force-purge any cached opencv variants before installing requirements,
# so the version pinned in requirements.txt is the one that ends up on PATH.
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless || true

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import cv2, numpy as np; print('cv2', cv2.__version__, 'numpy', np.__version__); assert cv2.__version__.startswith('4.10.'), cv2.__version__"

# Copy application source AND trained models
COPY src/ ./src/
COPY models/ ./models/

# corner_detector.pth is downloaded at runtime from S3 via boto3 on first use.
# Set CORNER_MODEL_S3_KEY (e.g. "models/corner_detector.pth") + S3_BUCKET env vars
# in Railway to enable auto-calibration. No presigned URLs needed.

# Writable runtime dirs
RUN mkdir -p data/sessions logs

EXPOSE 8000

# hand_landmarker.task is downloaded automatically by hand_detector.py on first run
CMD ["python", "src/capture/server.py"]
