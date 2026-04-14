"""
Diagnostic tool: visualise classifier predictions on a single board image.

Saves an 8×8 grid image where each cell shows the square patch with its
predicted piece label and confidence overlaid. Border colour indicates
confidence: green ≥0.8, yellow 0.5–0.8, red <0.5.

Usage:
    python scripts/diagnose_session.py --local_dir data/sessions/SESSION_001_C364 --frame 0
    python scripts/diagnose_session.py --local_dir data/sessions/SESSION_001_C364 --frame 0 --classifier pretrained
"""

import os
import sys
import json
import re
import argparse
import cv2
import numpy as np

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.process_board import crop_squares_from_grid
from src.models.inference import ChessPieceClassifier, PretrainedBoardClassifier


CELL = 100  # output cell size in pixels


def _border_color(conf: float):
    if conf >= 0.8:
        return (0, 200, 0)    # green
    if conf >= 0.5:
        return (0, 180, 220)  # yellow
    return (0, 0, 220)        # red


def make_diagnosis_grid(patches: dict, predictions: dict) -> np.ndarray:
    files = "abcdefgh"
    ranks = "87654321"
    canvas = np.zeros((8 * CELL, 8 * CELL, 3), dtype=np.uint8)

    for ri, rank in enumerate(ranks):
        for fi, f in enumerate(files):
            sq = f"{f}{rank}"
            patch = patches.get(sq)
            piece, conf = predictions.get(sq, ('?', 0.0))

            cell = np.zeros((CELL, CELL, 3), dtype=np.uint8)
            if patch is not None:
                cell = cv2.resize(patch, (CELL, CELL))

            # Coloured border
            color = _border_color(conf)
            thickness = 3
            cv2.rectangle(cell, (0, 0), (CELL - 1, CELL - 1), color, thickness)

            # Piece label (large, centred)
            label = piece if piece != 'empty' else '·'
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale, thick = 1.4, 2
            (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
            tx = (CELL - tw) // 2
            ty = (CELL + th) // 2 - 4
            cv2.putText(cell, label, (tx, ty), font, scale, (255, 255, 255), thick + 1,
                        cv2.LINE_AA)
            cv2.putText(cell, label, (tx, ty), font, scale, (0, 0, 0), thick, cv2.LINE_AA)

            # Confidence (small, bottom)
            conf_str = f"{conf:.0%}"
            cv2.putText(cell, conf_str, (4, CELL - 5), cv2.FONT_HERSHEY_PLAIN, 0.9,
                        (200, 200, 200), 1, cv2.LINE_AA)

            # Square name (top-left, tiny)
            cv2.putText(cell, sq, (3, 12), cv2.FONT_HERSHEY_PLAIN, 0.8,
                        (180, 180, 180), 1, cv2.LINE_AA)

            y0, x0 = ri * CELL, fi * CELL
            canvas[y0:y0 + CELL, x0:x0 + CELL] = cell

    return canvas


def _frame_number(fname):
    stem = os.path.splitext(fname)[0]
    m = re.search(r'(\d+)$', stem)
    return int(m.group(1)) if m else 0


def load_frame(local_dir: str, frame_idx: int):
    info_path = os.path.join(local_dir, 'game_info.json')
    with open(info_path) as f:
        game_info = json.load(f)

    warped_dir = os.path.join(local_dir, 'warped')
    img_dir = warped_dir if os.path.isdir(warped_dir) else local_dir

    files = sorted(
        [fn for fn in os.listdir(img_dir)
         if re.search(r'\d+\.(jpg|jpeg|png)$', fn, re.IGNORECASE)],
        key=_frame_number,
    )

    if frame_idx >= len(files):
        raise IndexError(f"Frame {frame_idx} out of range (session has {len(files)} images)")

    img_path = os.path.join(img_dir, files[frame_idx])
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {img_path}")

    print(f"Loaded frame {frame_idx}: {files[frame_idx]}")
    return img, game_info


def main():
    parser = argparse.ArgumentParser(description='Visualise classifier predictions on one board image')
    parser.add_argument('--local_dir', required=True, help='Local session directory')
    parser.add_argument('--frame', type=int, default=0, help='Frame index (0-based)')
    parser.add_argument('--output', help='Output image path (default: auto-named in cwd)')
    parser.add_argument('--classifier', default='patch', choices=['patch', 'pretrained'])
    parser.add_argument('--model_path', default='models/chess_piece_classifier.pth')
    parser.add_argument('--force_uniform_grid', action='store_true',
                        help='Override board_grid with uniform 50px grid (useful for diagnosing misaligned grids)')
    args = parser.parse_args()

    img, game_info = load_frame(args.local_dir, args.frame)

    uniform_grid = {'x_lines': [i * 50 for i in range(9)],
                    'y_lines': [i * 50 for i in range(9)]}
    if args.force_uniform_grid:
        grid = uniform_grid
        print("Using forced uniform grid (0,50,100,...,400)")
    else:
        grid = game_info.get('board_grid', uniform_grid)
        print(f"Using grid from game_info: x={grid['x_lines']}, y={grid['y_lines']}")
    rotation = game_info.get('rotation_angle', 0)

    patches = crop_squares_from_grid(img, grid, rotation)

    if args.classifier == 'pretrained':
        pretrained_path = os.path.join(
            os.path.dirname(args.model_path), '..', 'src', 'models', 'pretrained', 'checkpoint.ckpt')
        pretrained_path = os.path.normpath(pretrained_path)
        clf = PretrainedBoardClassifier(checkpoint_path=pretrained_path)
        print(f"Using pretrained ResNeXt classifier")
    else:
        clf = ChessPieceClassifier(model_path=args.model_path)
        print(f"Using patch classifier from {args.model_path}")

    predictions = clf.predict_board(patches)

    # Print FEN and summary stats
    from src.pipeline.fen_generator import predictions_to_fen, fen_position_only
    fen = fen_position_only(predictions_to_fen(predictions))
    print(f"\nFEN: {fen}")

    non_empty = [(sq, p, c) for sq, (p, c) in predictions.items() if p != 'empty']
    print(f"Non-empty squares ({len(non_empty)}): "
          + ", ".join(f"{sq}={p}({c:.0%})" for sq, p, c in sorted(non_empty)))

    low_conf = [(sq, p, c) for sq, (p, c) in predictions.items() if c < 0.5]
    if low_conf:
        print(f"Low-confidence squares: "
              + ", ".join(f"{sq}={p}({c:.0%})" for sq, p, c in sorted(low_conf)))

    grid_vis = make_diagnosis_grid(patches, predictions)

    session_name = os.path.basename(os.path.normpath(args.local_dir))
    out_path = args.output or f"diagnosis_{session_name}_frame{args.frame}_{args.classifier}.jpg"
    cv2.imwrite(out_path, grid_vis)
    print(f"\nSaved diagnosis image → {out_path}")


if __name__ == '__main__':
    main()
