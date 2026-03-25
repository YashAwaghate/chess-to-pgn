# Chess-to-PGN

Automated OTB chess game recorder: webcam → board capture → piece classification → PGN.

## Project Structure

```
src/
├── capture/                    # Live capture system (FastAPI)
│   ├── server.py               # Main server, state machine, all API endpoints
│   ├── hand_detector.py        # MediaPipe hand detection
│   └── static/                 # Frontend
│       ├── index.html          # Main capture UI
│       ├── app.js              # Capture logic (calibration, orientation, grid correction)
│       ├── styles.css
│       ├── viewer.html         # S3 image gallery
│       └── labeling.html       # Manual square labeling tool
├── models/                     # ML classifier
│   ├── classifier.py           # CVCHESS-inspired residual CNN (13 classes)
│   ├── prepare_chessred.py     # ChessReD dataset → training patches
│   ├── train.py                # Training script
│   └── inference.py            # Prediction wrapper
├── pipeline/                   # Image → PGN processing
│   ├── fen_generator.py        # 64 predictions → FEN
│   ├── move_detector.py        # FEN diff → legal move (python-chess)
│   ├── pgn_generator.py        # Moves → PGN file
│   └── process_game.py         # End-to-end session processor
└── preprocessing/
    └── process_board.py        # Board segmentation, grid-aware cropping, orientation detection
```

## Capture State Machine

SETUP → CALIBRATING → ORIENTATION → GRID_CORRECTION → STATIC ↔ MOVING

- **CALIBRATING**: User drags 4 corners on webcam to frame the board
- **ORIENTATION**: User picks where a1 is (4-button picker)
- **GRID_CORRECTION**: User drags 18 grid lines to align with actual board squares
- **STATIC/MOVING**: Hand detection triggers transitions; captures on 0.5s cooldown after hand leaves

## Key Technical Details

- Board images are warped to 400×400px via `cv2.getPerspectiveTransform`
- Corrected grid (9 x_lines + 9 y_lines) saved in `game_info.json` for accurate 64-square cropping
- Classifier: 13 classes — empty, P, N, B, R, Q, K, p, n, b, r, q, k
- Move detector has fuzzy matching (≥62/64 squares) to tolerate classifier errors
- S3 structure: `sessions/{game_id}/warped/`, `sessions/{game_id}/raw/`, `game_info.json`, `labels/`

## Current Status (March 2026)

- Capture system: **complete and deployed on Railway**
- ML pipeline code: **built, not yet trained** — needs ChessReD dataset download + training
- Next: download ChessReD → prepare patches → train → test end-to-end → iterate

## Commands

```bash
# Run server locally
python src/capture/server.py

# Prepare ChessReD training data
python -m src.models.prepare_chessred --chessred_dir /path/to/ChessReD

# Train classifier
python -m src.models.train --data_dir data/chessred_patches --epochs 30

# Process a game session to PGN
python -m src.pipeline.process_game --game_id <id> --output game.pgn
```
