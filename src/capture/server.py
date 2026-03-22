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

# Add capture path so hand_detector is importable from here
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(os.path.join(project_root, 'src', 'preprocessing'))
sys.path.append(script_dir)
from process_board import determine_orientation, CANVAS_HEIGHT, BOARD_START_Y
from hand_detector import HandDetector

app = FastAPI()

# Single global HandDetector (MediaPipe model loaded once at startup)
hand_detector = HandDetector()

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
        self.board_corners = []       # 4 [x, y] points in original frame space
        self.hand_exit_time = None    # timestamp when hand last left the board
        self.cooldown_duration = 0.5  # seconds to wait after hand exits before capture
        
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
        session.board_corners = pts  # store original-frame corners for hand detection
        logger.info("Perspective Matrix computed from web calibration.")

        warped = cv2.warpPerspective(frame, session.perspective_matrix, (400, CANVAS_HEIGHT))
        session.rotation_angle = determine_orientation(warped)
        logger.info(f"Orientation locked to: {session.rotation_angle}")

        session.state = CaptureState.STATIC
        
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

    # Run hand detection on the original (un-warped) frame
    hand_result = hand_detector.process_frame(frame, session.board_corners)

    if session.state == CaptureState.STATIC:
        if hand_result.over_board:
            logger.info("Hand over board detected — STATE: MOVING")
            session.state = CaptureState.MOVING
            session.hand_exit_time = None

    elif session.state == CaptureState.MOVING:
        if hand_result.over_board:
            # Hand still present — reset any exit timer
            session.hand_exit_time = None
        else:
            # Hand has left the board
            if session.hand_exit_time is None:
                session.hand_exit_time = time.time()
            elif time.time() - session.hand_exit_time >= session.cooldown_duration:
                logger.info(f"Hand gone for {session.cooldown_duration}s — capturing move {session.move_number}.")
                session.state = CaptureState.STATIC
                session.hand_exit_time = None

                warped = cv2.warpPerspective(frame, session.perspective_matrix, (400, CANVAS_HEIGHT))
                save_and_upload(warped)

    # Return annotated debug frame (hand landmarks + board polygon)
    debug_b64 = encode_image(hand_result.annotated_frame)

    return {
        "state": session.state.value,
        "is_moving": session.state == CaptureState.MOVING,
        "mask_b64": debug_b64,
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
