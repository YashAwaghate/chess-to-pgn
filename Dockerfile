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

# Download corner detector model from S3.
# Set CORNER_MODEL_URL as a Railway build variable (presigned S3 URL).
# Uses Python to avoid shell quoting issues with & in presigned URLs.
ARG CORNER_MODEL_URL=""
ENV CORNER_MODEL_URL=${CORNER_MODEL_URL}
RUN python3 -c "\
import os,sys,urllib.request; \
u=os.environ.get('CORNER_MODEL_URL',''); \
d='models/corner_detector.pth'; \
(print('CORNER_MODEL_URL not set, skipping') or sys.exit(0)) if not u else None; \
(print('corner_detector.pth already present') or sys.exit(0)) if os.path.exists(d) else None; \
print('Downloading corner_detector.pth...'); \
[urllib.request.urlretrieve(u,d), print(str(os.path.getsize(d)//1000000)+'MB')] if True else None; \
" || echo "WARNING: corner_detector.pth download failed — auto-calibrate will be unavailable"

# Writable runtime dirs
RUN mkdir -p data/sessions logs

EXPOSE 8000

# hand_landmarker.task is downloaded automatically by hand_detector.py on first run
CMD ["python", "src/capture/server.py"]
