import os
import cv2
import numpy as np
import time
import logging
import uuid
from enum import Enum
import sys
import boto3
from botocore.exceptions import NoCredentialsError

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(os.path.join(project_root, 'src', 'preprocessing'))
sys.path.append(script_dir)
from process_board import determine_orientation, CANVAS_HEIGHT, BOARD_START_Y
from hand_detector import HandDetector

class CaptureState(Enum):
    CALIBRATING = 1
    STATIC = 2
    MOVING = 3

class ChessSessionManager:
    def __init__(self, s3_bucket="yash-chess-capstone-2026", upload=True, camera_index=0):
        self.state = CaptureState.CALIBRATING
        self.camera_index = camera_index
        self.s3_bucket = s3_bucket
        self.upload = upload
        self.game_id = f"GAME_{uuid.uuid4().hex[:8].upper()}"
        self.move_number = 0
        self.rotation_angle = 0
        self.perspective_matrix = None
        self.reference_points = []
        
        # Hand detection
        self.hand_detector = HandDetector()
        self.hand_exit_time = None    # timestamp when hand last left the board
        self.cooldown_duration = 0.5  # seconds after hand exits before capture
        
        # AWS Setup
        if self.upload:
            self.s3_client = boto3.client('s3')
            
        # Logging Setup
        log_dir = os.path.join(project_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"session_{self.game_id}.log")
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler(log_file),
                                logging.StreamHandler()
                            ])
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Started new session: {self.game_id}")
        
    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.reference_points) < 4:
                self.reference_points.append((x, y))
                self.logger.info(f"Clicked point: ({x}, {y})")

    def calibrate(self, display):
        """Allows user to click 4 corners (TL, TR, BR, BL) to establish fixed ROI."""
        for i, pt in enumerate(self.reference_points):
            cv2.circle(display, pt, 5, (0, 0, 255), -1)
            cv2.putText(display, str(i+1), (pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
        cv2.putText(display, "Click 4 corners (TL, TR, BR, BL), then press 'C'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Chess Session Manager", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and len(self.reference_points) == 4:
            pts = np.array(self.reference_points, dtype="float32")

            dst_points = np.array([
                [0, BOARD_START_Y],
                [400, BOARD_START_Y],
                [400, BOARD_START_Y + 400],
                [0, BOARD_START_Y + 400]
            ], dtype="float32")

            self.perspective_matrix = cv2.getPerspectiveTransform(pts, dst_points)
            self.logger.info("Calibration successful. Perspective Matrix computed.")

            # Use `display` (which IS the current frame copy) for orientation detection
            warped = cv2.warpPerspective(display, self.perspective_matrix, (400, CANVAS_HEIGHT))
            self.rotation_angle = determine_orientation(warped)
            self.logger.info(f"Orientation locked to: {self.rotation_angle} degrees.")

            self.state = CaptureState.STATIC

            # Save the '00' setup image
            self.save_and_upload(warped)
            
    def process_hand(self, frame):
        """
        Runs hand detection and updates the state machine.
        Returns the annotated debug frame (landmarks + board polygon drawn).
        """
        hand_result = self.hand_detector.process_frame(frame, self.reference_points)

        if self.state == CaptureState.STATIC:
            if hand_result.over_board:
                self.logger.info("Hand over board — STATE: MOVING")
                self.state = CaptureState.MOVING
                self.hand_exit_time = None

        elif self.state == CaptureState.MOVING:
            if hand_result.over_board:
                self.hand_exit_time = None
            else:
                if self.hand_exit_time is None:
                    self.hand_exit_time = time.time()
                elif time.time() - self.hand_exit_time >= self.cooldown_duration:
                    self.logger.info(f"Hand gone for {self.cooldown_duration}s — capturing move {self.move_number}.")
                    self.state = CaptureState.STATIC
                    self.hand_exit_time = None
                    warped = cv2.warpPerspective(frame, self.perspective_matrix, (400, CANVAS_HEIGHT))
                    self.save_and_upload(warped)

        return hand_result.annotated_frame

    def save_and_upload(self, warped_frame):
        """Saves the rectified snapshot locally and uploads to S3."""
        filename = f"{self.game_id}_{self.move_number:03d}.jpg"
        save_dir = os.path.join(project_root, "data", "sessions", self.game_id)
        os.makedirs(save_dir, exist_ok=True)
        local_path = os.path.join(save_dir, filename)
        
        # The frame is already rectified, we optionally rotate it physically here so that 
        # White is always at the bottom if the user prefers viewing standard images natively.
        # However, preserving the captured perspective is fine since segmentation handles rotation.
        # Let's save it exactly as captured and maintain the rotation_json alongside.
        cv2.imwrite(local_path, warped_frame)
        self.logger.info(f"Saved local image: {local_path}")
        
        # Save rotation context for pipeline
        import json
        with open(os.path.join(save_dir, "rotation.json"), "w") as f:
            json.dump({"rotation": self.rotation_angle}, f)
        
        if self.upload:
            s3_key = f"custom-dataset/{self.game_id}/{filename}"
            try:
                self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
                self.logger.info(f"Uploaded S3 => s3://{self.s3_bucket}/{s3_key}")
            except Exception as e:
                self.logger.error(f"S3 Upload failed: {e}")
                
        self.move_number += 1

    def run(self):
        self.logger.info(f"Initializing camera feed (Index {self.camera_index})...")
        cap = cv2.VideoCapture(self.camera_index)
        cv2.namedWindow("Chess Session Manager")
        cv2.setMouseCallback("Chess Session Manager", self.click_event)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                self.logger.error("Failed to read camera frame.")
                break
                
            display = frame.copy()
                
            if self.state == CaptureState.CALIBRATING:
                self.calibrate(display)
            else:
                annotated = self.process_hand(frame)

                # Overlay UI info on the annotated debug frame
                status_color = (0, 0, 255) if self.state == CaptureState.MOVING else (0, 255, 0)
                cv2.putText(annotated, f"STATE: {self.state.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(annotated, f"ROTATION: {self.rotation_angle}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated, f"MOVES CAPTURED: {self.move_number - 1}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Chess Session Manager", annotated)
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.logger.info("Quitting session.")
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.hand_detector.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chess Session Manager")
    parser.add_argument("--no-upload", action="store_true", help="Disable AWS S3 uploads for local testing")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    
    args = parser.parse_args()
    
    upload = not args.no_upload
    app = ChessSessionManager(upload=upload, camera_index=args.camera)
    app.run()
