"""
Prepare ChessReD dataset for patch-based classification (CVCHESS approach).

Approach (arXiv 2511.11522 — "CVCHESS: Computer Vision Chess"):
  For each board image:
    1. Warp to 400×400 using annotated corner homography
    2. Map each annotated piece to its grid cell via bbox centroid → (row//50, col//50)
    3. Crop 64 uniform 50×50 patches; label pieces from annotations, remainder as 'empty'
    4. Save to {output_dir}/{split}/{class_name}/{img_id}_{row}_{col}.jpg

Only images that have corner annotations are used (1442 train / 330 val in ChessReD).
Splits are taken directly from annotations.json to match the official dataset split.

Dataset: ChessReD — Chess Recognition Dataset (Masouris, 2023)
  https://data.4tu.nl/articles/dataset/ChessReD_Chess_Recognition_Dataset/19600827

Usage:
  python -m src.models.prepare_chessred --chessred_dir data --output_dir data/chessred_patches
"""

import os
import cv2
import json
import argparse
import shutil
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm


BOARD_SIZE = 400
SQUARE_SIZE = 50   # 400 / 8

# ChessReD category_id → folder name used in ImageFolder directories.
# On Windows, directory names are case-insensitive, so 'B' and 'b' would collide.
# Black pieces use a trailing underscore (e.g. 'b_') to stay distinct on all platforms.
# inference.py maps these folder names back to standard FEN characters.
CATEGORY_TO_CLASS = {
    0: 'P',      # white-pawn
    1: 'R',      # white-rook
    2: 'N',      # white-knight
    3: 'B',      # white-bishop
    4: 'Q',      # white-queen
    5: 'K',      # white-king
    6: 'p_',     # black-pawn
    7: 'r_',     # black-rook
    8: 'n_',     # black-knight
    9: 'b_',     # black-bishop
    10: 'q_',    # black-queen
    11: 'k_',    # black-king
    12: 'empty',
}

ALL_CLASSES = list(CATEGORY_TO_CLASS.values())


def warp_board(img, corners: dict) -> np.ndarray:
    """Perspective-warp image to BOARD_SIZE×BOARD_SIZE using annotated corners.

    corners dict has keys: top_left, top_right, bottom_right, bottom_left
    each value is [x, y] in the original image.
    """
    src = np.float32([
        corners['top_left'],
        corners['top_right'],
        corners['bottom_right'],
        corners['bottom_left'],
    ])
    dst = np.float32([
        [0, 0],
        [BOARD_SIZE - 1, 0],
        [BOARD_SIZE - 1, BOARD_SIZE - 1],
        [0, BOARD_SIZE - 1],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (BOARD_SIZE, BOARD_SIZE)), M


def bbox_centroid_to_grid(bbox, M) -> tuple:
    """Map a bbox [x, y, w, h] in original coords to a warped grid cell (row, col).

    Returns (row, col) where row 0 = top of board, col 0 = left of board.
    Returns None if centroid falls outside the board after warp.
    """
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2

    # Apply homography to centroid
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    warped_pt = cv2.perspectiveTransform(pt, M)[0][0]
    wx, wy = warped_pt

    if not (0 <= wx < BOARD_SIZE and 0 <= wy < BOARD_SIZE):
        return None

    row = int(wy // SQUARE_SIZE)
    col = int(wx // SQUARE_SIZE)
    return (min(row, 7), min(col, 7))


def crop_patch(warped_img, row: int, col: int) -> np.ndarray:
    """Crop a SQUARE_SIZE×SQUARE_SIZE patch at grid position (row, col)."""
    y1 = row * SQUARE_SIZE
    y2 = (row + 1) * SQUARE_SIZE
    x1 = col * SQUARE_SIZE
    x2 = (col + 1) * SQUARE_SIZE
    return warped_img[y1:y2, x1:x2]


def prepare_dataset(chessred_dir: str, output_dir: str):
    ann_path = os.path.join(chessred_dir, 'annotations.json')
    if not os.path.exists(ann_path):
        print(f"ERROR: annotations.json not found at {ann_path}")
        return

    print(f"Loading annotations from {ann_path} ...")
    with open(ann_path) as f:
        data = json.load(f)

    # Build lookup tables
    images_by_id = {img['id']: img for img in data['images']}
    corners_by_id = {c['image_id']: c['corners'] for c in data['annotations']['corners']}

    # Group piece annotations by image_id
    pieces_by_image = defaultdict(list)
    for p in data['annotations']['pieces']:
        pieces_by_image[p['image_id']].append(p)

    # Official train / val splits (use 'val' for validation, skip 'test')
    splits = data['splits']
    split_ids = {
        'train': set(splits['train']['image_ids']),
        'val':   set(splits['val']['image_ids']),
    }

    # Only process images that have corner annotations
    cornered_ids = set(corners_by_id.keys())
    split_cornered = {
        name: sorted(ids & cornered_ids)
        for name, ids in split_ids.items()
    }
    for name, ids in split_cornered.items():
        print(f"  {name}: {len(ids)} images with corners "
              f"(of {len(split_ids[name])} total in split)")

    # Clean and recreate output directories
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    for split in ['train', 'val']:
        for cls in ALL_CLASSES:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    class_counts = Counter()
    skipped = 0

    for split_name, img_ids in split_cornered.items():
        print(f"\nProcessing {split_name} split ({len(img_ids)} images)...")

        for img_id in tqdm(img_ids, desc=f'  {split_name}'):
            img_info = images_by_id[img_id]

            # Resolve image path: annotations use "images/0/G000_IMG000.jpg"
            # actual location:          "ChessRed_images/0/G000_IMG000.jpg"
            rel_path = img_info['path'].replace('images/', 'ChessRed_images/', 1)
            img_path = os.path.join(chessred_dir, rel_path)

            if not os.path.exists(img_path):
                skipped += 1
                continue

            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue

            corners = corners_by_id[img_id]
            try:
                warped, M = warp_board(img, corners)
            except Exception:
                skipped += 1
                continue

            # Build grid label map: (row, col) → class_name
            grid_labels = {}  # default: empty

            for piece in pieces_by_image[img_id]:
                cell = bbox_centroid_to_grid(piece['bbox'], M)
                if cell is None:
                    continue
                row, col = cell
                class_name = CATEGORY_TO_CLASS.get(piece['category_id'], 'empty')
                # If two pieces map to same cell, prefer non-empty
                if (row, col) not in grid_labels or class_name != 'empty':
                    grid_labels[(row, col)] = class_name

            # Save all 64 patches
            for row in range(8):
                for col in range(8):
                    class_name = grid_labels.get((row, col), 'empty')
                    patch = crop_patch(warped, row, col)

                    out_path = os.path.join(
                        output_dir, split_name, class_name,
                        f'{img_id:05d}_{row}{col}.jpg'
                    )
                    cv2.imwrite(out_path, patch)
                    class_counts[class_name] += 1

    # Stats
    print("\n" + "-" * 40)
    print(f"Skipped images: {skipped}")
    print("\n--- Class Distribution ---")
    total = sum(class_counts.values())
    for cls in ALL_CLASSES:
        count = class_counts.get(cls, 0)
        pct = count / total * 100 if total else 0
        print(f"  {cls:6s}: {count:7,d}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':6s}: {total:7,d}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare ChessReD patches for CVCHESS-style training')
    parser.add_argument('--chessred_dir', default='data',
                        help='Directory containing annotations.json and ChessRed_images/')
    parser.add_argument('--output_dir', default='data/chessred_patches',
                        help='Output directory for patch images')
    args = parser.parse_args()

    prepare_dataset(args.chessred_dir, args.output_dir)
