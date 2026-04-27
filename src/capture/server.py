import os
# Suppress MediaPipe / TensorFlow C++ warnings (W0000 noise) before any imports
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import time
import uuid
import base64
import json
import logging
import sys
from datetime import date
from enum import Enum
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Load .env file when running locally (no-op in production where env vars are injected)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(os.path.join(project_root, 'src', 'preprocessing'))
sys.path.append(script_dir)
from hand_detector import HandDetector
from process_board import determine_orientation

# ── S3 (optional — only active when S3_BUCKET env var is set) ────────────────
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_PREFIX = os.getenv("S3_PREFIX", "sessions")
_s3_client = None
if S3_BUCKET:
    try:
        import boto3
        _s3_client = boto3.client(
            "s3",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
    except Exception as _e:
        print(f"[WARN] Could not initialise S3 client: {_e}. Running without S3.")

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
    SETUP           = "SETUP"           # Step 0: enter game metadata
    CALIBRATING     = "CALIBRATING"     # Step 1: click 4 board corners
    GRID_CORRECTION = "GRID_CORRECTION" # Step 2: adjust grid lines
    ORIENTATION     = "ORIENTATION"     # Step 3: click a1 square
    STATIC          = "STATIC"
    MOVING          = "MOVING"


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
        self.board_grid         = None  # {'x_lines': [...], 'y_lines': [...]}
        self.hand_exit_time     = None
        self.cooldown_duration  = 0.5
        self.save_raw: bool     = False
        self.game_info: dict         = {}   # populated by /api/setup
        self.last_activity_time      = time.time()


session = SessionState()

ACTIVITY_TIMEOUT = 60  # seconds — auto-reset if no activity in calibration/orientation

log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)
# Stream to stdout so Railway shows app logs in the same colour as uvicorn (not red)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Startup: log S3 configuration status clearly ─────────────────────────────
if S3_BUCKET and _s3_client:
    logger.info(f"S3 configured → s3://{S3_BUCKET}/{S3_PREFIX}/")
elif S3_BUCKET and not _s3_client:
    logger.warning("S3_BUCKET is set but boto3 client failed to initialise — uploads will be skipped.")
else:
    logger.warning("S3 not configured (S3_BUCKET env var missing) — images will be saved locally only.")


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
    save_raw:     Optional[bool] = False

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

class GridCorrectionData(BaseModel):
    x_lines: list[int]  # 9 values
    y_lines: list[int]  # 9 values


# ──────────────────────────── Helpers ────────────────────────────────────────

def decode_image(b64: str) -> np.ndarray:
    if not b64:
        raise ValueError("Empty image data received")
    if "," in b64:
        b64 = b64.split(",")[1]
    buf = np.frombuffer(base64.b64decode(b64), np.uint8).copy()
    if buf.size == 0:
        raise ValueError("Decoded image buffer is empty")
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed — invalid or unsupported image format")
    return img

def encode_image(img: np.ndarray) -> str:
    _, buf = cv2.imencode('.jpg', img)
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

def apply_rotation(image: np.ndarray, code) -> np.ndarray:
    return image if code is None else cv2.rotate(image, code)

def _rotate_grid(grid: dict, code) -> dict:
    """Rotate grid line coordinates to match a cv2.rotate() code applied to a 400×400 image."""
    if code is None:
        return grid
    xs = grid['x_lines']
    ys = grid['y_lines']
    SIZE = 400
    if code == cv2.ROTATE_90_CLOCKWISE:       # (x,y) -> (SIZE-y, x)
        return {'x_lines': [SIZE - y for y in reversed(ys)],
                'y_lines': list(xs)}
    if code == cv2.ROTATE_180:                # (x,y) -> (SIZE-x, SIZE-y)
        return {'x_lines': [SIZE - x for x in reversed(xs)],
                'y_lines': [SIZE - y for y in reversed(ys)]}
    if code == cv2.ROTATE_90_COUNTERCLOCKWISE: # (x,y) -> (y, SIZE-x)
        return {'x_lines': list(ys),
                'y_lines': [SIZE - x for x in reversed(xs)]}
    return grid

def snap_to_corner(col: int, row: int) -> tuple:
    corners = [(0, 0), (7, 0), (0, 7), (7, 7)]
    return min(corners, key=lambda c: abs(c[0] - col) + abs(c[1] - row))

def detect_board_grid(warped: np.ndarray) -> dict:
    """Uniform 8×8 grid — after perspective warp to 400×400, grid is evenly spaced."""
    H, W = warped.shape[:2]
    return {
        'x_lines': [round(i * W / 8) for i in range(9)],
        'y_lines': [round(i * H / 8) for i in range(9)],
    }


def write_game_info():
    """Write (or update) game_info.json — to S3 if configured, else locally."""
    data = dict(session.game_info)
    data["rotation_angle"] = session.rotation_angle
    data["total_moves"] = max(0, session.move_number - 1)
    if session.board_grid:
        data["board_grid"] = session.board_grid
    json_bytes = json.dumps(data, indent=2).encode()
    s3_key = f"{S3_PREFIX}/{session.game_id}/game_info.json"
    if not _s3_put(s3_key, json_bytes, "application/json"):
        # Fallback: local save (development / no S3)
        save_dir = os.path.join(project_root, "data", "sessions", session.game_id)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "game_info.json"), "w") as f:
            f.write(json_bytes.decode())


# ──────────────────────────── Endpoints ──────────────────────────────────────

@app.get("/api/debug/grid")
def debug_grid():
    """Returns the warped setup frame with the detected grid drawn on it — useful for verifying detection."""
    if session.warped_setup_frame is None or session.board_grid is None:
        return JSONResponse(status_code=400, content={"message": "No warped frame available"})
    img   = session.warped_setup_frame.copy()
    grid  = session.board_grid
    for x in grid['x_lines']:
        cv2.line(img, (x, 0), (x, img.shape[0]), (0, 255, 0), 2)
    for y in grid['y_lines']:
        cv2.line(img, (0, y), (img.shape[1], y), (0, 255, 0), 2)
    return {"debug_b64": encode_image(img)}


@app.get("/api/state")
def get_state():
    # Auto-reset stale calibration/orientation sessions on state check
    if (session.state in (CaptureState.CALIBRATING, CaptureState.ORIENTATION, CaptureState.GRID_CORRECTION)
            and time.time() - session.last_activity_time > ACTIVITY_TIMEOUT):
        logger.info("Activity timeout on state check — auto-resetting stale session")
        session.reset()
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
    if session.state not in (CaptureState.SETUP, CaptureState.CALIBRATING, CaptureState.ORIENTATION, CaptureState.GRID_CORRECTION):
        return JSONResponse(status_code=400, content={"message": "Game already in progress"})
    if session.state != CaptureState.SETUP:
        logger.info(f"Setup called from {session.state.value} — resetting stale session")
        session.reset()
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
        session.save_raw = bool(data.save_raw)
        session.state = CaptureState.CALIBRATING
        logger.info(f"Game setup: {white} vs {black} on {game_date} (save_raw={session.save_raw})")
        return {"status": "success", "game_info": session.game_info}
    except Exception as e:
        logger.error(f"Setup error: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.post("/api/calibrate")
def calibrate(data: CalibrationData):
    """Step 1: compute perspective transform, return warped preview."""
    session.last_activity_time = time.time()
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
        session.state              = CaptureState.GRID_CORRECTION

        grid = detect_board_grid(warped)
        session.board_grid = grid
        logger.info(f"Grid — x: {grid['x_lines']}, y: {grid['y_lines']}")

        return {"status": "success", "warped_b64": encode_image(warped), "grid": grid}
    except Exception as e:
        logger.error(f"Calibration error: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.post("/api/set_orientation")
def set_orientation(data: OrientationData):
    """Step 3: user identifies a1. Rotate board so a1 is always bottom-left, then start."""
    session.last_activity_time = time.time()
    if session.state != CaptureState.ORIENTATION:
        return JSONResponse(status_code=400, content={"message": "Not in ORIENTATION state"})
    try:
        corner               = snap_to_corner(data.col, data.row)
        session.rotation_code  = _A1_ROTATION[corner]
        session.rotation_angle = _A1_ANGLE_LABEL[corner]

        # Rotate the already grid-corrected warped frame
        rotated = apply_rotation(session.warped_setup_frame, session.rotation_code)
        session.warped_setup_frame = rotated

        # Re-apply rotation to the stored corrected grid lines
        if session.board_grid:
            session.board_grid = _rotate_grid(session.board_grid, session.rotation_code)

        # Save the initial board image (move 000) and go STATIC
        save_and_upload(session.warped_setup_frame)
        session.state = CaptureState.STATIC
        logger.info(f"Orientation set: {session.rotation_angle}°. Session active.")
        return {
            "status": "success",
            "rotation_angle": session.rotation_angle,
        }
    except Exception as e:
        logger.error(f"Orientation error: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.post("/api/confirm_grid")
def confirm_grid(data: GridCorrectionData):
    """Step 2: user corrects grid lines, then moves to orientation (a1 selection)."""
    session.last_activity_time = time.time()
    if session.state != CaptureState.GRID_CORRECTION:
        return JSONResponse(status_code=400, content={"message": "Not in GRID_CORRECTION state"})
    try:
        if len(data.x_lines) != 9 or len(data.y_lines) != 9:
            return JSONResponse(status_code=400, content={"message": "Need exactly 9 x and 9 y lines"})
        # Validate monotonically increasing
        for i in range(1, 9):
            if data.x_lines[i] <= data.x_lines[i - 1] or data.y_lines[i] <= data.y_lines[i - 1]:
                return JSONResponse(status_code=400, content={"message": "Lines must be monotonically increasing"})
        session.board_grid = {'x_lines': list(data.x_lines), 'y_lines': list(data.y_lines)}
        session.state = CaptureState.ORIENTATION
        logger.info(f"Grid confirmed. Orientation step next. Grid: x={data.x_lines}, y={data.y_lines}")

        # Auto-detect orientation suggestion on the corrected warped frame
        _ANGLE_TO_CORNER = {0: {"col": 0, "row": 7}, 90: {"col": 7, "row": 7},
                            180: {"col": 7, "row": 0}, 270: {"col": 0, "row": 0}}
        try:
            suggested_angle = determine_orientation(session.warped_setup_frame)
        except Exception:
            suggested_angle = 0
        suggested_a1 = {**_ANGLE_TO_CORNER[suggested_angle], "angle": suggested_angle}
        logger.info(f"Auto-detected orientation: {suggested_angle}°")

        return {
            "status": "success",
            "warped_b64": encode_image(session.warped_setup_frame),
            "grid": session.board_grid,
            "suggested_a1": suggested_a1,
        }
    except Exception as e:
        logger.error(f"Grid confirmation error: {e}")
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


@app.post("/api/new_game_same_calibration")
def new_game_same_calibration(data: GameSetupData):
    """Start a new game while keeping the existing calibration (perspective,
    grid, orientation). Resets game_id, move counter, and metadata."""
    if session.perspective_matrix is None:
        return JSONResponse(status_code=400,
                            content={"message": "No active calibration to reuse"})

    # Generate a fresh game_id, keep all calibration state intact
    sessions_dir = os.path.join(project_root, "data", "sessions")
    os.makedirs(sessions_dir, exist_ok=True)
    n = len([d for d in os.listdir(sessions_dir)
             if os.path.isdir(os.path.join(sessions_dir, d))]) + 1
    session.game_id = f"SESSION_{n:03d}_{uuid.uuid4().hex[:4].upper()}"
    session.move_number = 0
    session.hand_exit_time = None
    session.last_activity_time = time.time()

    game_date = data.game_date or str(date.today())
    white = data.white.strip() or "Player 1"
    black = data.black.strip() or "Player 2"
    event = data.event.strip() or f"{white} vs {black}"
    site = data.site.strip() or "—"

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

    # Save the existing warped setup frame as move 000 of the new game
    if session.warped_setup_frame is not None:
        save_and_upload(session.warped_setup_frame)

    session.state = CaptureState.STATIC
    logger.info(f"New game started with reused calibration: {session.game_id}")
    return {
        "status": "success",
        "game_id": session.game_id,
        "game_info": session.game_info,
    }


@app.post("/api/process_frame")
def process_frame(data: FrameData):
    # Auto-reset stale calibration/orientation sessions
    if (session.state in (CaptureState.CALIBRATING, CaptureState.ORIENTATION, CaptureState.GRID_CORRECTION)
            and time.time() - session.last_activity_time > ACTIVITY_TIMEOUT):
        logger.info("Activity timeout — auto-resetting stale session")
        session.reset()
    if session.state in (CaptureState.SETUP, CaptureState.CALIBRATING, CaptureState.ORIENTATION, CaptureState.GRID_CORRECTION):
        return {"status": session.state.value.lower()}
    session.last_activity_time = time.time()

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
                save_and_upload(rotated, raw_frame=frame)

    return {
        "state":     session.state.value,
        "is_moving": session.state == CaptureState.MOVING,
        "mask_b64":  encode_image(hand_result.annotated_frame),
    }


def _s3_put(s3_key: str, data: bytes, content_type: str = "image/jpeg") -> bool:
    """Upload bytes directly to S3 (no temp file). Returns True on success."""
    if not _s3_client or not S3_BUCKET:
        return False
    try:
        _s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=data, ContentType=content_type)
        logger.info(f"S3 ↑ s3://{S3_BUCKET}/{s3_key}")
        return True
    except Exception as e:
        logger.error(f"S3 upload failed for {s3_key}: {e}")
        return False


def save_and_upload(frame: np.ndarray, raw_frame: np.ndarray = None):
    filename = f"{session.move_number:03d}.jpg"

    # Always save warped+rotated image
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    img_bytes = buf.tobytes()
    s3_key = f"{S3_PREFIX}/{session.game_id}/warped/{filename}"
    if not _s3_put(s3_key, img_bytes):
        save_dir = os.path.join(project_root, "data", "sessions", session.game_id, "warped")
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, filename), frame)
        logger.info(f"Saved locally (warped): {filename}")

    # Opt-in: save raw camera frame
    if session.save_raw and raw_frame is not None:
        _, rbuf = cv2.imencode('.jpg', raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        raw_bytes = rbuf.tobytes()
        raw_s3_key = f"{S3_PREFIX}/{session.game_id}/raw/{filename}"
        if not _s3_put(raw_s3_key, raw_bytes):
            raw_dir = os.path.join(project_root, "data", "sessions", session.game_id, "raw")
            os.makedirs(raw_dir, exist_ok=True)
            cv2.imwrite(os.path.join(raw_dir, filename), raw_frame)
            logger.info(f"Saved locally (raw): {filename}")

    write_game_info()
    session.move_number += 1


# ──────────────────────────── Gallery API (S3 viewer) ────────────────────────

@app.get("/api/gallery/sessions")
def gallery_list_sessions():
    """List all session folders in S3 (or local fallback)."""
    if _s3_client and S3_BUCKET:
        try:
            paginator = _s3_client.get_paginator("list_objects_v2")
            sessions = set()
            for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"{S3_PREFIX}/", Delimiter="/"):
                for cp in page.get("CommonPrefixes", []):
                    folder = cp["Prefix"].rstrip("/").split("/")[-1]
                    sessions.add(folder)
            return {"source": "s3", "sessions": sorted(sessions, reverse=True)}
        except Exception as e:
            logger.error(f"S3 list error: {e}")
            return JSONResponse(status_code=500, content={"message": str(e)})

    # Fallback: local filesystem
    local_dir = os.path.join(project_root, "data", "sessions")
    if os.path.isdir(local_dir):
        folders = sorted(
            [d for d in os.listdir(local_dir) if os.path.isdir(os.path.join(local_dir, d))],
            reverse=True,
        )
        return {"source": "local", "sessions": folders}
    return {"source": "local", "sessions": []}


@app.get("/api/gallery/{session_id}")
def gallery_get_session(session_id: str):
    """Return game_info + warped/raw image lists for a session."""
    if _s3_client and S3_BUCKET:
        try:
            prefix = f"{S3_PREFIX}/{session_id}/"
            resp = _s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, MaxKeys=1000)
            warped_images = []
            raw_images = []
            flat_images = []
            game_info = None
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                name = key.split("/")[-1]
                if name == "game_info.json":
                    body = _s3_client.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
                    game_info = json.loads(body)
                elif name.endswith(".jpg"):
                    # Determine if in warped/ or raw/ subfolder or flat
                    rel = key[len(prefix):]
                    if rel.startswith("warped/"):
                        warped_images.append(name)
                    elif rel.startswith("raw/"):
                        raw_images.append(name)
                    else:
                        flat_images.append(name)
            # Backward compat: old flat sessions → treat as warped
            images = sorted(warped_images) if warped_images else sorted(flat_images)
            return {"session_id": session_id, "game_info": game_info,
                    "images": images, "raw_images": sorted(raw_images),
                    "has_subfolders": bool(warped_images)}
        except Exception as e:
            logger.error(f"S3 session read error: {e}")
            return JSONResponse(status_code=500, content={"message": str(e)})

    # Fallback: local
    local_dir = os.path.join(project_root, "data", "sessions", session_id)
    if not os.path.isdir(local_dir):
        return JSONResponse(status_code=404, content={"message": "Session not found"})
    warped_dir = os.path.join(local_dir, "warped")
    raw_dir = os.path.join(local_dir, "raw")
    if os.path.isdir(warped_dir):
        images = sorted(f for f in os.listdir(warped_dir) if f.endswith(".jpg"))
        raw_images = sorted(f for f in os.listdir(raw_dir) if f.endswith(".jpg")) if os.path.isdir(raw_dir) else []
        has_subfolders = True
    else:
        images = sorted(f for f in os.listdir(local_dir) if f.endswith(".jpg"))
        raw_images = []
        has_subfolders = False
    game_info = None
    info_path = os.path.join(local_dir, "game_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            game_info = json.load(f)
    return {"session_id": session_id, "game_info": game_info,
            "images": images, "raw_images": raw_images, "has_subfolders": has_subfolders}


@app.get("/api/gallery/{session_id}/{image_type}/{filename}")
def gallery_get_image_typed(session_id: str, image_type: str, filename: str):
    """Stream a single image from warped/ or raw/ subfolder."""
    if image_type not in ("warped", "raw"):
        return JSONResponse(status_code=400, content={"message": "image_type must be 'warped' or 'raw'"})
    if _s3_client and S3_BUCKET:
        try:
            key = f"{S3_PREFIX}/{session_id}/{image_type}/{filename}"
            obj = _s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            return Response(content=obj["Body"].read(), media_type="image/jpeg")
        except Exception:
            pass
    local_path = os.path.join(project_root, "data", "sessions", session_id, image_type, filename)
    if os.path.exists(local_path):
        with open(local_path, "rb") as f:
            return Response(content=f.read(), media_type="image/jpeg")
    return JSONResponse(status_code=404, content={"message": "Image not found"})


@app.get("/api/gallery/{session_id}/{filename}")
def gallery_get_image(session_id: str, filename: str):
    """Stream a single image — backward compat for old flat sessions or warped fallback."""
    if _s3_client and S3_BUCKET:
        # Try warped/ first, then flat
        for sub in ("warped/", ""):
            try:
                key = f"{S3_PREFIX}/{session_id}/{sub}{filename}"
                obj = _s3_client.get_object(Bucket=S3_BUCKET, Key=key)
                return Response(content=obj["Body"].read(), media_type="image/jpeg")
            except Exception:
                continue
        return JSONResponse(status_code=404, content={"message": "Image not found"})

    # Fallback: local — try warped/ then flat
    for sub in ("warped", ""):
        local_path = os.path.join(project_root, "data", "sessions", session_id, sub, filename)
        if os.path.exists(local_path):
            with open(local_path, "rb") as f:
                return Response(content=f.read(), media_type="image/jpeg")
    return JSONResponse(status_code=404, content={"message": "Image not found"})


# ──────────────────────────── PGN Generation API ─────────────────────────────

@app.post("/api/generate_pgn/{game_id}")
def generate_pgn_endpoint(game_id: str):
    """Process a captured session and generate PGN."""
    try:
        # Add project root to path for pipeline imports
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from src.pipeline.process_game import process_game_session

        model_path = os.path.join(project_root, "models", "chess_piece_classifier_v2.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(project_root, "models", "chess_piece_classifier.pth")
        if not os.path.exists(model_path):
            return JSONResponse(status_code=400, content={
                "message": "Model not found. Train the classifier first.",
                "model_path": model_path,
            })

        # Prefer local session dir when it exists (avoids needing S3 creds locally)
        local_dir = os.path.join(project_root, "data", "sessions", game_id)
        if os.path.isdir(local_dir):
            result = process_game_session(
                local_dir=local_dir,
                model_path=model_path,
            )
        else:
            result = process_game_session(
                game_id=game_id,
                model_path=model_path,
                s3_bucket=S3_BUCKET or None,
            )

        # Upload PGN to S3 if available
        pgn_str = result['pgn']
        if _s3_client and S3_BUCKET:
            pgn_key = f"{S3_PREFIX}/{game_id}/game.pgn"
            _s3_client.put_object(Bucket=S3_BUCKET, Key=pgn_key,
                                  Body=pgn_str.encode(), ContentType="text/plain")

        return {
            "pgn": pgn_str,
            "moves": result['moves'],
            "total_images": len(result['fen_sequence']),
            "errors": result['errors'],
            "skipped": result['skipped'],
        }
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"message": str(e)})
    except Exception as e:
        logging.exception("PGN generation failed")
        return JSONResponse(status_code=500, content={"message": str(e)})


# ──────────────────────────── Labeling API ───────────────────────────────────

@app.get("/api/labeling/sessions")
def labeling_list_sessions():
    """List available sessions with image counts for labeling."""
    sessions = []
    if _s3_client and S3_BUCKET:
        try:
            paginator = _s3_client.get_paginator("list_objects_v2")
            # List all game_info.json files to find sessions
            for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"{S3_PREFIX}/",
                                            Delimiter="/"):
                for prefix_obj in page.get("CommonPrefixes", []):
                    session_prefix = prefix_obj["Prefix"]
                    session_id = session_prefix.rstrip("/").split("/")[-1]

                    # Count warped images
                    img_count = 0
                    for img_page in paginator.paginate(Bucket=S3_BUCKET,
                                                        Prefix=f"{session_prefix}warped/"):
                        for obj in img_page.get("Contents", []):
                            if obj["Key"].lower().endswith((".jpg", ".jpeg", ".png")):
                                img_count += 1

                    # Check for game_info
                    has_info = False
                    try:
                        _s3_client.head_object(Bucket=S3_BUCKET,
                                               Key=f"{session_prefix}game_info.json")
                        has_info = True
                    except Exception:
                        pass

                    if has_info:
                        sessions.append({
                            "session_id": session_id,
                            "image_count": img_count,
                        })
        except Exception as e:
            logging.exception("Failed listing sessions for labeling")
    return {"sessions": sessions}


@app.get("/api/labeling/session/{game_id}/images")
def labeling_list_images(game_id: str):
    """List warped images and existing labels for a session."""
    images = []
    labels_exist = set()

    if _s3_client and S3_BUCKET:
        paginator = _s3_client.get_paginator("list_objects_v2")
        # List warped images
        for page in paginator.paginate(Bucket=S3_BUCKET,
                                        Prefix=f"{S3_PREFIX}/{game_id}/warped/"):
            for obj in page.get("Contents", []):
                filename = obj["Key"].split("/")[-1]
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    images.append(filename)

        # List existing labels
        for page in paginator.paginate(Bucket=S3_BUCKET,
                                        Prefix=f"{S3_PREFIX}/{game_id}/labels/"):
            for obj in page.get("Contents", []):
                label_name = obj["Key"].split("/")[-1].replace(".json", "")
                labels_exist.add(label_name)

    images.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)

    # Load game_info for grid
    game_info = {}
    if _s3_client and S3_BUCKET:
        try:
            resp = _s3_client.get_object(Bucket=S3_BUCKET,
                                          Key=f"{S3_PREFIX}/{game_id}/game_info.json")
            game_info = json.loads(resp["Body"].read().decode())
        except Exception:
            pass

    return {
        "images": images,
        "labeled": list(labels_exist),
        "game_info": game_info,
    }


@app.get("/api/labeling/session/{game_id}/image/{idx}/predictions")
def labeling_get_predictions(game_id: str, idx: str):
    """Get classifier predictions for a specific image (auto-fill for labeling)."""
    try:
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        model_path = os.path.join(project_root, "models", "chess_piece_classifier_v2.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(project_root, "models", "chess_piece_classifier.pth")
        if not os.path.exists(model_path):
            return JSONResponse(status_code=400, content={"message": "Model not trained yet"})

        # Load image
        if _s3_client and S3_BUCKET:
            for sub in ("warped/", ""):
                try:
                    key = f"{S3_PREFIX}/{game_id}/{sub}{idx}.jpg"
                    resp = _s3_client.get_object(Bucket=S3_BUCKET, Key=key)
                    data = resp["Body"].read()
                    arr = np.frombuffer(data, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    break
                except Exception:
                    img = None

        if img is None:
            return JSONResponse(status_code=404, content={"message": "Image not found"})

        # Load game_info for grid
        resp = _s3_client.get_object(Bucket=S3_BUCKET,
                                      Key=f"{S3_PREFIX}/{game_id}/game_info.json")
        game_info = json.loads(resp["Body"].read().decode())
        grid = game_info.get("board_grid", {
            "x_lines": [i * 50 for i in range(9)],
            "y_lines": [i * 50 for i in range(9)],
        })
        rotation = game_info.get("rotation_angle", 0)

        from src.preprocessing.process_board import crop_squares_from_grid
        from src.models.inference import ChessPieceClassifier

        patches = crop_squares_from_grid(img, grid, rotation)
        classifier = ChessPieceClassifier(model_path=model_path)
        predictions = classifier.predict_board(patches)

        # Convert to serializable format
        result = {}
        for sq, (piece, conf) in predictions.items():
            result[sq] = {"piece": piece, "confidence": round(conf, 4)}

        return {"predictions": result}
    except Exception as e:
        logging.exception("Prediction failed")
        return JSONResponse(status_code=500, content={"message": str(e)})


class LabelData(BaseModel):
    labels: dict  # {square_name: piece_class}


@app.post("/api/labeling/session/{game_id}/image/{idx}")
def labeling_save_labels(game_id: str, idx: str, data: LabelData):
    """Save manual labels for a specific image."""
    # Validate labels
    valid_pieces = {'empty', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'}
    for sq, piece in data.labels.items():
        if piece not in valid_pieces:
            return JSONResponse(status_code=400,
                                content={"message": f"Invalid piece '{piece}' for square {sq}"})

    label_json = json.dumps(data.labels, indent=2)

    if _s3_client and S3_BUCKET:
        key = f"{S3_PREFIX}/{game_id}/labels/{idx}.json"
        _s3_client.put_object(Bucket=S3_BUCKET, Key=key,
                              Body=label_json.encode(), ContentType="application/json")
    else:
        # Save locally
        label_dir = os.path.join(project_root, "data", "sessions", game_id, "labels")
        os.makedirs(label_dir, exist_ok=True)
        with open(os.path.join(label_dir, f"{idx}.json"), "w") as f:
            f.write(label_json)

    return {"status": "saved", "squares_labeled": len(data.labels)}


@app.get("/api/labeling/session/{game_id}/image/{idx}/labels")
def labeling_get_labels(game_id: str, idx: str):
    """Get existing labels for a specific image."""
    if _s3_client and S3_BUCKET:
        try:
            key = f"{S3_PREFIX}/{game_id}/labels/{idx}.json"
            resp = _s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            labels = json.loads(resp["Body"].read().decode())
            return {"labels": labels, "exists": True}
        except Exception:
            return {"labels": {}, "exists": False}

    local_path = os.path.join(project_root, "data", "sessions", game_id, "labels", f"{idx}.json")
    if os.path.exists(local_path):
        with open(local_path) as f:
            return {"labels": json.load(f), "exists": True}
    return {"labels": {}, "exists": False}


# ──────────────────────────── Static files (must be last) ────────────────────
static_dir = os.path.join(script_dir, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    # reload=True is fine locally; Railway/Docker will just run it once
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
