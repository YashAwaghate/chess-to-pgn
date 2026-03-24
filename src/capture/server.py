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
        self.board_grid         = None  # {'x_lines': [...], 'y_lines': [...]}
        self.hand_exit_time     = None
        self.cooldown_duration  = 0.5
        self.game_info: dict    = {}   # populated by /api/setup


session = SessionState()

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

def detect_board_grid(warped: np.ndarray) -> dict:
    """
    Detect the 9 vertical and 9 horizontal grid lines on a warped board image
    using gradient projection — robust to pieces blocking the lines.

    Strategy:
      • Compute Sobel gradients, sum their absolute values per column/row.
      • The chessboard alternating pattern creates strong, consistent peaks
        at every grid boundary across the full image width/height.
      • Pick the 9 peaks with correct spacing via non-maximum suppression.
      • Extrapolate if fewer than 9 are found.

    Returns {'x_lines': [9 ints], 'y_lines': [9 ints]}.
    """
    H, W = warped.shape[:2]

    def uniform():
        return {
            'x_lines': [round(i * W / 8) for i in range(9)],
            'y_lines': [round(i * H / 8) for i in range(9)],
        }

    # ── Gradient profiles ────────────────────────────────────────────────────
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)   # vertical edges
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)   # horizontal edges

    col_profile = np.sum(np.abs(gx), axis=0)   # W values — peaks at vertical lines
    row_profile = np.sum(np.abs(gy), axis=1)   # H values — peaks at horizontal lines

    def find_9_lines(profile, dim):
        # Smooth to suppress piece-edge noise while keeping grid peaks
        smooth_k = max(3, dim // 60)
        kernel   = np.ones(smooth_k) / smooth_k
        smooth   = np.convolve(profile, kernel, mode='same')

        # Minimum expected spacing: assume grid occupies at least 50% of dim
        min_spacing = dim // (8 * 3)   # generous lower bound

        # Collect all local maxima above a noise threshold
        noise_floor = np.mean(smooth) + 0.3 * np.std(smooth)
        candidates  = []
        for i in range(1, dim - 1):
            if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1] \
                    and smooth[i] > noise_floor:
                candidates.append((float(smooth[i]), i))

        if len(candidates) < 2:
            return None   # signal too weak, fall back to uniform

        # Non-maximum suppression: keep strongest peaks with min spacing
        candidates.sort(key=lambda x: -x[0])
        selected = []
        for _, pos in candidates:
            if all(abs(pos - s) >= min_spacing for s in selected):
                selected.append(pos)

        selected.sort()
        if len(selected) < 2:
            return None

        # Estimate grid spacing from the detected peaks
        diffs   = np.diff(selected)
        valid   = diffs[diffs > min_spacing * 0.5]
        if len(valid) == 0:
            return None
        spacing = float(np.median(valid))

        # Extrapolate forward and backward to reach 9 lines
        result = list(map(float, selected))
        for _ in range(20):
            if len(result) >= 9:
                break
            # Prefer filling toward the edges of the image
            gap_front = result[0]
            gap_back  = dim - result[-1]
            if gap_front > gap_back and gap_front > spacing * 0.4:
                result.insert(0, result[0] - spacing)
            elif gap_back > spacing * 0.4:
                result.append(result[-1] + spacing)
            else:
                break

        # Trim extras (keep the 9 that best cover [0, dim])
        while len(result) > 9:
            if abs(result[0]) > abs(result[-1] - dim):
                result.pop(0)
            else:
                result.pop()
        # Last resort pad
        while len(result) < 9:
            result.append(result[-1] + spacing)

        return [int(round(v)) for v in sorted(result)[:9]]

    x_lines = find_9_lines(col_profile, W)
    y_lines = find_9_lines(row_profile, H)

    if x_lines is None or y_lines is None:
        logger.warning("Grid detection fell back to uniform grid.")
        return uniform()

    logger.info(f"Grid x: {x_lines}")
    logger.info(f"Grid y: {y_lines}")
    return {'x_lines': x_lines, 'y_lines': y_lines}


def write_game_info():
    """Write (or update) game_info.json — to S3 if configured, else locally."""
    data = dict(session.game_info)
    data["rotation"]    = session.rotation_angle
    data["total_moves"] = max(0, session.move_number - 1)
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

        grid = detect_board_grid(warped)
        session.board_grid = grid
        logger.info(f"Grid detected — x: {grid['x_lines']}, y: {grid['y_lines']}")
        return {"status": "success", "warped_b64": encode_image(warped), "grid": grid}
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


def save_and_upload(frame: np.ndarray):
    filename = f"{session.game_id}_{session.move_number:03d}.jpg"
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    img_bytes = buf.tobytes()
    s3_key = f"{S3_PREFIX}/{session.game_id}/{filename}"
    if not _s3_put(s3_key, img_bytes):
        # Fallback: local save (development / no S3)
        save_dir = os.path.join(project_root, "data", "sessions", session.game_id)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, filename), frame)
        logger.info(f"Saved locally: {filename}")
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
    """Return game_info + image list for a session."""
    if _s3_client and S3_BUCKET:
        try:
            prefix = f"{S3_PREFIX}/{session_id}/"
            resp = _s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
            images = []
            game_info = None
            for obj in resp.get("Contents", []):
                name = obj["Key"].split("/")[-1]
                if name.endswith(".jpg"):
                    images.append(name)
                elif name == "game_info.json":
                    body = _s3_client.get_object(Bucket=S3_BUCKET, Key=obj["Key"])["Body"].read()
                    game_info = json.loads(body)
            return {"session_id": session_id, "game_info": game_info, "images": sorted(images)}
        except Exception as e:
            logger.error(f"S3 session read error: {e}")
            return JSONResponse(status_code=500, content={"message": str(e)})

    # Fallback: local
    local_dir = os.path.join(project_root, "data", "sessions", session_id)
    if not os.path.isdir(local_dir):
        return JSONResponse(status_code=404, content={"message": "Session not found"})
    images = sorted(f for f in os.listdir(local_dir) if f.endswith(".jpg"))
    game_info = None
    info_path = os.path.join(local_dir, "game_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            game_info = json.load(f)
    return {"session_id": session_id, "game_info": game_info, "images": images}


@app.get("/api/gallery/{session_id}/{filename}")
def gallery_get_image(session_id: str, filename: str):
    """Stream a single image (from S3 or local)."""
    if _s3_client and S3_BUCKET:
        try:
            key = f"{S3_PREFIX}/{session_id}/{filename}"
            obj = _s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            return Response(content=obj["Body"].read(), media_type="image/jpeg")
        except Exception as e:
            return JSONResponse(status_code=404, content={"message": str(e)})

    # Fallback: local
    local_path = os.path.join(project_root, "data", "sessions", session_id, filename)
    if os.path.exists(local_path):
        with open(local_path, "rb") as f:
            return Response(content=f.read(), media_type="image/jpeg")
    return JSONResponse(status_code=404, content={"message": "Image not found"})


# ──────────────────────────── Static files (must be last) ────────────────────
static_dir = os.path.join(script_dir, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    # reload=True is fine locally; Railway/Docker will just run it once
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
