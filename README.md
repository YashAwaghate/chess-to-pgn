# Chess to PGN

This project aims to convert chess board images to PGN notation using computer vision and deep learning.

## Directory Structure

- `/data/raw`: Contains the raw ChessReD images.
- `/data/processed`: Contains warped 400x400 top-down crops of the chess boards.
- `/src/preprocessing`: Scripts for Canny edge detection and Hough transform.
- `/src/models`: Residual CNN training scripts.
- `/notebooks`: Jupyter notebooks for rapid experimentation with OpenCV.
