import os
import cv2
import numpy as np
import time
import uuid
import base64
import logging
from enum import Enum
from fastapi import FastAPI, UploadFile, Form, File, Body
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys

# Add preprocessing path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(os.path.join(project_root, 'src', 'preprocessing'))
from process_board import determine_orientation, CANVAS_HEIGHT, BOARD_START_Y

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# App State
class CaptureState(Enum):
    CALIBRATING = "CALIBRATING"
    STATIC = "STATIC"
    MOVING = "MOVING"

class SessionState:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.state = CaptureState.CALIBRATING
        
        # Calculate session number based on existing output folders
        sessions_dir = os.path.join(project_root, "data", "sessions")
        os.makedirs(sessions_dir, exist_ok=True)
        session_num = len([d for d in os.listdir(sessions_dir) if os.path.isdir(os.path.join(sessions_dir, d))]) + 1
        
        self.game_id = f"SESSION_{session_num:03d}_{uuid.uuid4().hex[:4].upper()}"
        self.move_number = 0
        self.perspective_matrix = None
        self.rotation_angle = 0
        self.bg_frame = None
        self.last_motion_time = time.time()
        self.moving_start_time = time.time()
        self.motion_threshold = 2.0  # seconds
        self.reference_points = []
        self.prev_frame = None
        
# Global session instance (for single-user local deployment)
session = SessionState()

# Logging
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Models
class CalibrationData(BaseModel):
    points: list[dict] # [{"x": 100, "y": 200}, ...]
    image_b64: str

class FrameData(BaseModel):
    image_b64: str

def decode_image(b64_string):
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"

@app.get("/api/state")
def get_state():
    return {
        "state": session.state.value,
        "game_id": session.game_id,
        "move_number": session.move_number,
        "rotation_angle": session.rotation_angle,
        "calibrated": session.perspective_matrix is not None
    }

@app.post("/api/calibrate")
def calibrate(data: CalibrationData):
    try:
        pts = [[p['x'], p['y']] for p in data.points]
        sc_pts = np.array(pts, dtype="float32")
        frame = decode_image(data.image_b64)
        
        # Calculate transform
        dst_points = np.array([
            [0, BOARD_START_Y],
            [400, BOARD_START_Y],
            [400, BOARD_START_Y + 400],
            [0, BOARD_START_Y + 400]
        ], dtype="float32")
        
        session.perspective_matrix = cv2.getPerspectiveTransform(sc_pts, dst_points)
        logger.info("Perspective Matrix computed from web calibration.")
        
        warped = cv2.warpPerspective(frame, session.perspective_matrix, (400, CANVAS_HEIGHT))
        session.rotation_angle = determine_orientation(warped)
        logger.info(f"Orientation locked to: {session.rotation_angle}")
        
        session.state = CaptureState.STATIC
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        session.bg_frame = cv2.GaussianBlur(warped_gray, (21, 21), 0)
        session.prev_frame = session.bg_frame.copy()
        
        # Save setup image
        save_and_upload(warped)
        
        return {"status": "success", "rotation_angle": session.rotation_angle}
    except Exception as e:
        logger.error(f"Calibration error: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/api/reset")
def reset_session():
    session.reset()
    return {"status": "success", "message": "Session reset"}

@app.post("/api/process_frame")
def process_frame(data: FrameData):
    if session.state == CaptureState.CALIBRATING:
        return {"status": "calibrating"}
        
    frame = decode_image(data.image_b64)
    # Warp the frame first so we ONLY check motion on the chessboard ROI
    warped = cv2.warpPerspective(frame, session.perspective_matrix, (400, CANVAS_HEIGHT))
    
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if getattr(session, 'prev_frame', None) is None:
        session.prev_frame = gray_blur.copy()
    
    # Contrast against resting background (detects entry)
    bg_delta = cv2.absdiff(session.bg_frame, gray_blur)
    bg_thresh = cv2.threshold(bg_delta, 25, 255, cv2.THRESH_BINARY)[1]
    bg_thresh = cv2.dilate(bg_thresh, None, iterations=2)
    bg_contours, _ = cv2.findContours(bg_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_bg_area = max([cv2.contourArea(c) for c in bg_contours] + [0])
    
    # Contrast against previous frame (detects active movement)
    prev_delta = cv2.absdiff(session.prev_frame, gray_blur)
    prev_thresh = cv2.threshold(prev_delta, 25, 255, cv2.THRESH_BINARY)[1]
    prev_thresh = cv2.dilate(prev_thresh, None, iterations=2)
    prev_contours, _ = cv2.findContours(prev_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_prev_area = max([cv2.contourArea(c) for c in prev_contours] + [0])
    
    session.prev_frame = gray_blur.copy()
    
    # Thresholds
    trigger_area = 800       # Area differing from resting background to trigger MOVING
    movement_area = 100      # Area changing frame-to-frame to keep in MOVING 

    if session.state == CaptureState.STATIC:
        if max_bg_area > trigger_area:
            logger.info("Motion detected - STATE: MOVING")
            session.state = CaptureState.MOVING
            session.moving_start_time = time.time()
            session.last_motion_time = time.time()
            
    elif session.state == CaptureState.MOVING:
        if max_prev_area > movement_area:
            # Active movement happening frame-to-frame. Reset stabilization timer.
            session.last_motion_time = time.time()
            
            # Adaptive Background Update: 10 second timeout for runaway states
            time_since_move_started = time.time() - session.moving_start_time
            if time_since_move_started > 10.0:
                logger.warning("Moving state timed out (10s). Assuming camera shifted. Forcing re-stabilization.")
                session.bg_frame = gray_blur.copy()
                session.state = CaptureState.STATIC
        else:
            # Scene is still from frame-to-frame. Wait for stabilization threshold. 
            stable_duration = time.time() - session.last_motion_time
            if stable_duration >= session.motion_threshold:
                logger.info(f"Board stabilized. Capturing Move {session.move_number}.")
                session.state = CaptureState.STATIC
                
                # Capture the newly moved state
                save_and_upload(warped)
                
                # Lock the NEW board state as the background for future moves!
                session.bg_frame = gray_blur.copy()
    
    # Return a thumbnail of the mask for debugging in the UI
    mask_b64 = encode_image(bg_thresh)
    
    return {
        "state": session.state.value,
        "is_moving": session.state == CaptureState.MOVING,
        "mask_b64": mask_b64
    }

def save_and_upload(warped_frame):
    filename = f"{session.game_id}_{session.move_number:03d}.jpg"
    save_dir = os.path.join(project_root, "data", "sessions", session.game_id)
    os.makedirs(save_dir, exist_ok=True)
    local_path = os.path.join(save_dir, filename)
    cv2.imwrite(local_path, warped_frame)
    
    import json
    with open(os.path.join(save_dir, "rotation.json"), "w") as f:
        json.dump({"rotation": session.rotation_angle}, f)
        
    # AWS upload can be hooked up here later as requested
    logger.info(f"Saved locally: {local_path}")
    session.move_number += 1

# Mount the static files (the frontend UI)
static_dir = os.path.join(script_dir, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
