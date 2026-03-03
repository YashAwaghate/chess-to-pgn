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

# Add preprocessing path to import determine_orientation
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(os.path.join(project_root, 'src', 'preprocessing'))
from process_board import determine_orientation, CANVAS_HEIGHT, BOARD_START_Y

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
        
        # Motion detection params
        self.last_motion_time = time.time()
        self.motion_threshold = 2.0  # seconds
        self.bg_frame = None
        
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
            
            # The destination should match our existing 400x500 padded target from process_board
            # We want the 400x400 board to fit perfectly starting at Y=100
            dst_points = np.array([
                [0, BOARD_START_Y],
                [400, BOARD_START_Y],
                [400, BOARD_START_Y + 400],
                [0, BOARD_START_Y + 400]
            ], dtype="float32")
            
            self.perspective_matrix = cv2.getPerspectiveTransform(pts, dst_points)
            self.logger.info("Calibration successful. Perspective Matrix computed.")
            
            # Now, test the orientation on this reference frame!
            warped = cv2.warpPerspective(frame, self.perspective_matrix, (400, CANVAS_HEIGHT))
            self.rotation_angle = determine_orientation(warped)
            self.logger.info(f"Orientation locked to: {self.rotation_angle} degrees.")
            
            self.state = CaptureState.STATIC
            self.bg_frame = self.preprocess_for_motion(display)
            
            # Save the '00' setup image
            self.save_and_upload(warped)
            
    def preprocess_for_motion(self, frame):
        """Grayscales and blurs frame for differencing."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (21, 21), 0)

    def detect_motion(self, frame):
        """Updates the state transitioning between MOVING and STATIC."""
        gray = self.preprocess_for_motion(frame)
        
        # Absolute difference between current frame and our reference BG
        frame_delta = cv2.absdiff(self.bg_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        is_moving = False
        for c in contours:
            if cv2.contourArea(c) > 500: # Motion threshold area
                is_moving = True
                break
                
        if is_moving:
            if self.state != CaptureState.MOVING:
                self.logger.info("Motion detected! State -> MOVING")
            self.state = CaptureState.MOVING
            self.last_motion_time = time.time()
            
            # Slowly update background to adapt to lighting changes
            cv2.accumulateWeighted(gray, self.bg_frame.astype("float"), 0.05)
            self.bg_frame = cv2.convertScaleAbs(self.bg_frame)
        else:
            # If we were moving and now stabilized
            if self.state == CaptureState.MOVING:
                stable_duration = time.time() - self.last_motion_time
                if stable_duration >= self.motion_threshold:
                    self.logger.info(f"Board stabilized for {self.motion_threshold}s. Capturing move!")
                    self.state = CaptureState.STATIC
                    # Capture the finalized move
                    warped = cv2.warpPerspective(frame, self.perspective_matrix, (400, CANVAS_HEIGHT))
                    self.save_and_upload(warped)
                    
                    # Update background to perfectly match the new static state
                    self.bg_frame = gray
                    
        return thresh # Return visual difference for UI

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
                motion_mask = self.detect_motion(frame)
                
                # Draw grid bounds for visualization
                rectified_pts = np.array([
                    [0, BOARD_START_Y], [400, BOARD_START_Y], 
                    [400, BOARD_START_Y + 400], [0, BOARD_START_Y + 400]
                ], dtype="float32")
                inv_M = cv2.invert(self.perspective_matrix)[1]
                if inv_M is not None:
                    board_contour = cv2.perspectiveTransform(np.array([rectified_pts]), inv_M)
                    cv2.polylines(display, [np.int32(board_contour)], True, (0, 255, 0), 2)
                
                # Overlay UI info
                status_color = (0, 0, 255) if self.state == CaptureState.MOVING else (0, 255, 0)
                cv2.putText(display, f"STATE: {self.state.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(display, f"ROTATION: {self.rotation_angle}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display, f"MOVES CAPTURED: {self.move_number-1}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("Chess Session Manager", display)
                cv2.imshow("Motion Mask", motion_mask)
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.logger.info("Quitting session.")
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chess Session Manager")
    parser.add_argument("--no-upload", action="store_true", help="Disable AWS S3 uploads for local testing")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    
    args = parser.parse_args()
    
    upload = not args.no_upload
    app = ChessSessionManager(upload=upload, camera_index=args.camera)
    app.run()
