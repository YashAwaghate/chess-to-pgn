"""
Patch preparation V2 — labels each grid cell by ChessReD's canonical
`chessboard_position` field, not by bbox-centroid projection.

Why this matters
----------------
The V1 script (prepare_chessred.py) assigns labels by projecting each
piece's bbox centroid through the perspective warp and bucketing into
a 50×50 grid cell. For tall pieces under tilted cameras the centroid
sometimes falls into the *next* grid cell, so the training label and
the canonical square disagree.

Diagnostic on 20 test images (1,280 squares):
  - classifier vs V1 (bbox-centroid) labels: 91.3 %
  - classifier vs V2 (chessboard_position) labels: 82.7 %
  - labels disagree on: 14.7 % of squares

In other words, 8.6 / 17.3 percentage points of the end-to-end accuracy
loss is a *label-alignment* issue, not a perceptual one.

V2 instead says: for the piece annotated at chessboard_position='a1',
the warped grid cell at (row=7, col=0) is that piece — full stop.
(We verified on 20 starting-position frames that ChessReD's
`corners.top_left` consistently maps to a8 → warp-pixel (0,0), so
the chessboard_position → grid mapping is:
     row = 8 - int(rank),   col = 'abcdefgh'.index(file))

Usage:
    python -m src.models.prepare_chessred_v2 \
        --chessred_dir data --output_dir data/chessred_patches_v2
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
PATCH_PADDING = 6  # extra pixels on each side; gives 62×62 crops → resized to 50×50 at train time

# ChessReD category_id → ImageFolder-safe folder name
CATEGORY_TO_CLASS = {
    0: 'P', 1: 'R', 2: 'N', 3: 'B', 4: 'Q', 5: 'K',
    6: 'p_', 7: 'r_', 8: 'n_', 9: 'b_', 10: 'q_', 11: 'k_',
    12: 'empty',
}
ALL_CLASSES = list(CATEGORY_TO_CLASS.values())


def warp_board(img, corners: dict) -> np.ndarray:
    src = np.float32([corners['top_left'], corners['top_right'],
                      corners['bottom_right'], corners['bottom_left']])
    dst = np.float32([[0, 0], [BOARD_SIZE - 1, 0],
                      [BOARD_SIZE - 1, BOARD_SIZE - 1], [0, BOARD_SIZE - 1]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (BOARD_SIZE, BOARD_SIZE))


def square_to_grid(square: str) -> tuple:
    """Map chessboard_position ('a8'..'h1') to warped grid cell (row, col).

    Assumes corners.top_left == a8 (verified on 20 starting-position frames).
    """
    file_idx = 'abcdefgh'.index(square[0])
    rank = int(square[1])
    return (8 - rank, file_idx)


def crop_patch(warped_img, row: int, col: int) -> np.ndarray:
    y1 = max(0, row * SQUARE_SIZE - PATCH_PADDING)
    y2 = min(BOARD_SIZE, (row + 1) * SQUARE_SIZE + PATCH_PADDING)
    x1 = max(0, col * SQUARE_SIZE - PATCH_PADDING)
    x2 = min(BOARD_SIZE, (col + 1) * SQUARE_SIZE + PATCH_PADDING)
    return warped_img[y1:y2, x1:x2]


def prepare_dataset(chessred_dir: str, output_dir: str):
    with open(os.path.join(chessred_dir, 'annotations.json')) as f:
        data = json.load(f)

    images_by_id = {img['id']: img for img in data['images']}
    corners_by_id = {c['image_id']: c['corners']
                     for c in data['annotations']['corners']}

    pieces_by_image = defaultdict(list)
    for p in data['annotations']['pieces']:
        pieces_by_image[p['image_id']].append(p)

    splits = data['splits']
    split_ids = {
        'train': set(splits['train']['image_ids']),
        'val':   set(splits['val']['image_ids']),
    }

    cornered_ids = set(corners_by_id.keys())
    split_cornered = {
        name: sorted(ids & cornered_ids) for name, ids in split_ids.items()
    }
    for name, ids in split_cornered.items():
        print(f"  {name}: {len(ids)} images with corners "
              f"(of {len(split_ids[name])} total in split)")

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
            rel_path = img_info['path'].replace('images/', 'ChessRed_images/', 1)
            img_path = os.path.join(chessred_dir, rel_path)

            if not os.path.exists(img_path):
                skipped += 1
                continue
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue

            try:
                warped = warp_board(img, corners_by_id[img_id])
            except Exception:
                skipped += 1
                continue

            # Build label map using chessboard_position (canonical scheme)
            grid_labels = {}
            for piece in pieces_by_image[img_id]:
                sq = piece['chessboard_position']
                if not (len(sq) == 2 and sq[0] in 'abcdefgh' and sq[1] in '12345678'):
                    continue
                row, col = square_to_grid(sq)
                class_name = CATEGORY_TO_CLASS.get(piece['category_id'], 'empty')
                grid_labels[(row, col)] = class_name

            # Save every cell (filling empties)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--chessred_dir', default='data')
    parser.add_argument('--output_dir', default='data/chessred_patches_v2')
    args = parser.parse_args()
    prepare_dataset(args.chessred_dir, args.output_dir)
