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

# Download corner detector model from S3 if not present in build context.
# Set CORNER_MODEL_URL as a Railway build variable:
#   https://<bucket>.s3.<region>.amazonaws.com/models/corner_detector.pth
ARG CORNER_MODEL_URL=""
RUN if [ -n "$CORNER_MODEL_URL" ]; then \
        echo "Downloading corner_detector.pth from S3..." && \
        curl -fsSL -o models/corner_detector.pth "$CORNER_MODEL_URL" && \
        echo "Downloaded $(du -sh models/corner_detector.pth | cut -f1)"; \
    else \
        echo "CORNER_MODEL_URL not set — skipping corner model download"; \
    fi

# Writable runtime dirs
RUN mkdir -p data/sessions logs

EXPOSE 8000

# hand_landmarker.task is downloaded automatically by hand_detector.py on first run
CMD ["python", "src/capture/server.py"]
