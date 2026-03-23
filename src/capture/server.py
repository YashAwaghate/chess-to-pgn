import os
import cv2
import numpy as np
import time
import uuid
import base64
import json
import logging
from datetime import date
from enum import Enum
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(os.path.join(project_root, 'src', 'preprocessing'))
sys.path.append(script_dir)
from hand_detector import HandDetector

app = FastAPI()
hand_detector = HandDetector()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BOARD_SIZE = 400

_A1_ROTATION = {
    (0, 7): None,
    (7, 7): cv2.ROTATE_90_CLOCKWISE,
    (7, 0): cv2.ROTATE_180,
    (0, 0): cv2.ROTATE_90_COUNTERCLOCKWISE,
}
_A1_ANGLE_LABEL = {(0, 7): 0, (7, 7): 90, (7, 0): 180, (0, 0): 270}


class CaptureState(Enum):
    SETUP       = "SETUP"         # Step 0: enter game metadata
    CALIBRATING = "CALIBRATING"   # Step 1: click 4 board corners
    ORIENTATION = "ORIENTATION"   # Step 2: click a1 square
    STATIC      = "STATIC"
    MOVING      = "MOVING"


class SessionState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = CaptureState.SETUP
        sessions_dir = os.path.join(project_root, "data", "sessions")
        os.makedirs(sessions_dir, exist_ok=True)
        n = len([d for d in os.listdir(sessions_dir)
                 if os.path.isdir(os.path.join(sessions_dir, d))]) + 1
        self.game_id            = f"SESSION_{n:03d}_{uuid.uuid4().hex[:4].upper()}"
        self.move_number        = 0
        self.perspective_matrix = None
        self.rotation_angle     = 0
        self.rotation_code      = None
        self.board_corners      = []
        self.warped_setup_frame = None
        self.hand_exit_time     = None
        self.cooldown_duration  = 0.5
        self.game_info: dict    = {}   # populated by /api/setup


session = SessionState()

log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ──────────────────────────── Pydantic models ────────────────────────────────

class GameSetupData(BaseModel):
    white:        str
    black:        str
    event:        Optional[str] = ""
    site:         Optional[str] = ""
    game_date:    Optional[str] = ""   # ISO date string yyyy-mm-dd
    round:        Optional[str] = "-"
    time_control: Optional[str] = "Casual"
    notes:        Optional[str] = ""

class CalibrationData(BaseModel):
    points:    list[dict]
    image_b64: str

class OrientationData(BaseModel):
    col: int
    row: int

class FrameData(BaseModel):
    image_b64: str

class ResultData(BaseModel):
    result: str   # "1-0" | "0-1" | "1/2-1/2" | "*"


# ──────────────────────────── Helpers ────────────────────────────────────────

def decode_image(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",")[1]
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)

def encode_image(img: np.ndarray) -> str:
    _, buf = cv2.imencode('.jpg', img)
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

def apply_rotation(image: np.ndarray, code) -> np.ndarray:
    return image if code is None else cv2.rotate(image, code)

def snap_to_corner(col: int, row: int) -> tuple:
    corners = [(0, 0), (7, 0), (0, 7), (7, 7)]
    return min(corners, key=lambda c: abs(c[0] - col) + abs(c[1] - row))

def write_game_info():
    """Write (or update) game_info.json for the current session."""
    save_dir = os.path.join(project_root, "data", "sessions", session.game_id)
    os.makedirs(save_dir, exist_ok=True)
    data = dict(session.game_info)
    data["rotation"]    = session.rotation_angle
    data["total_moves"] = max(0, session.move_number - 1)
    with open(os.path.join(save_dir, "game_info.json"), "w") as f:
        json.dump(data, f, indent=2)


# ──────────────────────────── Endpoints ──────────────────────────────────────

@app.get("/api/state")
def get_state():
    return {
        "state":         session.state.value,
        "game_id":       session.game_id,
        "move_number":   session.move_number,
        "rotation_angle": session.rotation_angle,
        "calibrated":    session.perspective_matrix is not None,
        "game_info":     session.game_info,
    }


@app.post("/api/setup")
def setup_game(data: GameSetupData):
    """Step 0: save game metadata and move to CALIBRATING."""
    if session.state != CaptureState.SETUP:
        return JSONResponse(status_code=400, content={"message": "Not in SETUP state"})
    try:
        game_date = data.game_date or str(date.today())
        white     = data.white.strip() or "Player 1"
        black     = data.black.strip() or "Player 2"
        event     = data.event.strip() or f"{white} vs {black}"
        site      = data.site.strip()  or "—"

        session.game_info = {
            "game_id":      session.game_id,
            "white":        white,
            "black":        black,
            "event":        event,
            "site":         site,
            "date":         game_date,
            "round":        data.round or "-",
            "time_control": data.time_control or "Casual",
            "result":       "*",
            "notes":        data.notes or "",
        }
        session.state = CaptureState.CALIBRATING
        logger.info(f"Game setup: {white} vs {black} on {game_date}")
        return {"status": "success", "game_info": session.game_info}
    except Exception as e:
        logger.error(f"Setup error: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.post("/api/calibrate")
def calibrate(data: CalibrationData):
    """Step 1: compute perspective transform, return warped preview."""
    if session.state != CaptureState.CALIBRATING:
        return JSONResponse(status_code=400, content={"message": "Not in CALIBRATING state"})
    try:
        src   = np.array([[p['x'], p['y']] for p in data.points], dtype="float32")
        dst   = np.array([[0,0],[400,0],[400,400],[0,400]], dtype="float32")
        frame = decode_image(data.image_b64)

        session.perspective_matrix = cv2.getPerspectiveTransform(src, dst)
        session.board_corners      = [[p['x'], p['y']] for p in data.points]

        warped = cv2.warpPerspective(frame, session.perspective_matrix, (BOARD_SIZE, BOARD_SIZE))
        session.warped_setup_frame = warped.copy()
        session.state              = CaptureState.ORIENTATION

        logger.info("Perspective matrix computed — awaiting a1 click.")
        return {"status": "success", "warped_b64": encode_image(warped)}
    except Exception as e:
        logger.error(f"Calibration error: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.post("/api/set_orientation")
def set_orientation(data: OrientationData):
    """Step 2: user identifies a1. Rotate board so a1 is always bottom-left."""
    if session.state != CaptureState.ORIENTATION:
        return JSONResponse(status_code=400, content={"message": "Not in ORIENTATION state"})
    try:
        corner               = snap_to_corner(data.col, data.row)
        session.rotation_code  = _A1_ROTATION[corner]
        session.rotation_angle = _A1_ANGLE_LABEL[corner]

        rotated = apply_rotation(session.warped_setup_frame, session.rotation_code)
        save_and_upload(rotated)

        session.state = CaptureState.STATIC
        logger.info(f"Orientation set: {session.rotation_angle}°. Session active.")
        return {"status": "success", "rotation_angle": session.rotation_angle}
    except Exception as e:
        logger.error(f"Orientation error: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.post("/api/end_game")
def end_game(data: ResultData):
    """Record the final game result and flush game_info.json."""
    valid = {"1-0", "0-1", "1/2-1/2", "*"}
    if data.result not in valid:
        return JSONResponse(status_code=400, content={"message": f"Result must be one of {valid}"})
    session.game_info["result"] = data.result
    write_game_info()
    logger.info(f"Game ended: {data.result}")
    return {"status": "success", "result": data.result}


@app.post("/api/reset")
def reset_session():
    session.reset()
    return {"status": "success"}


@app.post("/api/process_frame")
def process_frame(data: FrameData):
    if session.state in (CaptureState.SETUP, CaptureState.CALIBRATING, CaptureState.ORIENTATION):
        return {"status": session.state.value.lower()}

    frame       = decode_image(data.image_b64)
    hand_result = hand_detector.process_frame(frame, session.board_corners)

    if session.state == CaptureState.STATIC:
        if hand_result.over_board:
            session.state          = CaptureState.MOVING
            session.hand_exit_time = None

    elif session.state == CaptureState.MOVING:
        if hand_result.over_board:
            session.hand_exit_time = None
        else:
            if session.hand_exit_time is None:
                session.hand_exit_time = time.time()
            elif time.time() - session.hand_exit_time >= session.cooldown_duration:
                logger.info(f"Capturing move {session.move_number}")
                session.state          = CaptureState.STATIC
                session.hand_exit_time = None

                warped  = cv2.warpPerspective(frame, session.perspective_matrix, (BOARD_SIZE, BOARD_SIZE))
                rotated = apply_rotation(warped, session.rotation_code)
                save_and_upload(rotated)

    return {
        "state":     session.state.value,
        "is_moving": session.state == CaptureState.MOVING,
        "mask_b64":  encode_image(hand_result.annotated_frame),
    }


def save_and_upload(frame: np.ndarray):
    filename = f"{session.game_id}_{session.move_number:03d}.jpg"
    save_dir = os.path.join(project_root, "data", "sessions", session.game_id)
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, filename), frame)
    write_game_info()
    logger.info(f"Saved: {filename}")
    session.move_number += 1


static_dir = os.path.join(script_dir, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
