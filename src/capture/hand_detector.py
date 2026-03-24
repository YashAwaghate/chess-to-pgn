import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from dataclasses import dataclass
from typing import List, Optional
import urllib.request
import os
import time

# Hand landmark connections for drawing (indices into the 21-point landmark list)
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (0,9),(9,10),(10,11),(11,12),     # middle
    (0,13),(13,14),(14,15),(15,16),   # ring
    (0,17),(17,18),(18,19),(19,20),   # pinky
    (5,9),(9,13),(13,17),             # palm knuckles
]


@dataclass
class HandDetectionResult:
    hand_present: bool
    over_board: bool
    landmarks_px: List[tuple]          # (x, y) pixel coords for every detected landmark
    annotated_frame: Optional[np.ndarray] = None


class HandDetector:
    """
    Wraps the MediaPipe Hand Landmarker (Tasks API, mediapipe>=0.10) to detect
    whether a human hand is present and over the chess board region.
    """

    _MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    _MODEL_FILENAME = "hand_landmarker.task"

    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        model_path = self._ensure_model()

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self._start_time = time.time()

    def _ensure_model(self) -> str:
        """Download the .task model file if not already present."""
        model_path = os.path.join(os.path.dirname(__file__), self._MODEL_FILENAME)
        if not os.path.exists(model_path):
            print(f"Downloading hand landmark model → {model_path} ...")
            urllib.request.urlretrieve(self._MODEL_URL, model_path)
            print("Download complete.")
        return model_path

    def process_frame(self, frame_bgr: np.ndarray,
                      board_corners_px=None) -> HandDetectionResult:
        """
        Detect hands in a BGR frame.

        Parameters
        ----------
        frame_bgr        : BGR numpy array from the camera (NOT warped).
        board_corners_px : (4, 2) array-like of [x, y] corner pixels in the
                           original frame that define the board polygon.

        Returns
        -------
        HandDetectionResult
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        annotated = frame_bgr.copy()
        landmarks_px: List[tuple] = []

        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
                landmarks_px.extend(pts)

                # Draw connections
                for a, b in _HAND_CONNECTIONS:
                    cv2.line(annotated, pts[a], pts[b], (0, 200, 255), 2)
                # Draw landmark dots
                for px, py in pts:
                    cv2.circle(annotated, (px, py), 4, (0, 255, 200), -1)

        hand_present = len(landmarks_px) > 0
        over_board = False

        # Check only fingertips (4,8,12,16,20) + palm center per hand.
        # Require ≥2 key points inside the board polygon to trigger over_board.
        _FINGERTIP_INDICES = [4, 8, 12, 16, 20]
        if hand_present and board_corners_px is not None:
            corners = np.array(board_corners_px, dtype=np.float32).reshape(-1, 1, 2)
            for hand_lms in result.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
                palm = ((pts[0][0] + pts[9][0]) // 2, (pts[0][1] + pts[9][1]) // 2)
                key_pts = [pts[i] for i in _FINGERTIP_INDICES] + [palm]
                inside = sum(
                    1 for px, py in key_pts
                    if cv2.pointPolygonTest(corners, (float(px), float(py)), False) >= 0
                )
                if inside >= 2:
                    over_board = True
                    break

            # Draw board polygon on debug frame
            cv2.polylines(annotated,
                          [np.array(board_corners_px, dtype=np.int32).reshape(-1, 1, 2)],
                          isClosed=True, color=(0, 255, 0), thickness=2)

        return HandDetectionResult(
            hand_present=hand_present,
            over_board=over_board,
            landmarks_px=landmarks_px,
            annotated_frame=annotated,
        )

    def close(self):
        self._detector.close()
