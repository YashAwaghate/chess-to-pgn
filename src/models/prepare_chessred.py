"""
Prepare the ChessReD dataset for per-square classification training.

ChessReD (Chess Recognition Dataset):
  - ~10,800 board images with FEN annotations
  - Images are rectified chessboard photos
  - Available from 4TU.ResearchData

This script:
  1. Reads board images and their FEN annotations
  2. Crops each image into 64 square patches (uniform 50px grid on 400×400)
  3. Labels each patch from the FEN
  4. Saves to data/chessred_patches/{train,val}/{class_name}/
  5. Prints class distribution stats

Usage:
  python -m src.models.prepare_chessred --chessred_dir /path/to/ChessReD --output_dir data/chessred_patches
"""

import os
import cv2
import json
import argparse
import random
import shutil
from collections import Counter
from tqdm import tqdm

# The 13 classes matching classifier.py
CLASS_NAMES = [
    'empty',
    'P', 'N', 'B', 'R', 'Q', 'K',
    'p', 'n', 'b', 'r', 'q', 'k',
]

BOARD_SIZE = 400
SQUARE_SIZE = 50


def fen_to_board(fen_position: str) -> list:
    """Convert FEN position string to 8×8 list of piece chars.

    Returns list of 8 rows (rank 8 first), each row is a list of 8 chars.
    Empty squares are 'empty'.
    """
    rows = fen_position.split('/')
    board = []
    for row in rows:
        rank = []
        for ch in row:
            if ch.isdigit():
                rank.extend(['empty'] * int(ch))
            else:
                rank.append(ch)
        board.append(rank)
    return board


def crop_uniform_grid(img, target_size=400):
    """Resize image to target_size×target_size and crop 64 uniform patches."""
    h, w = img.shape[:2]
    if h != target_size or w != target_size:
        img = cv2.resize(img, (target_size, target_size))

    patches = []
    for row in range(8):
        row_patches = []
        for col in range(8):
            y1 = row * SQUARE_SIZE
            y2 = (row + 1) * SQUARE_SIZE
            x1 = col * SQUARE_SIZE
            x2 = (col + 1) * SQUARE_SIZE
            patch = img[y1:y2, x1:x2]
            row_patches.append(patch)
        patches.append(row_patches)
    return patches


def find_chessred_structure(chessred_dir):
    """Detect ChessReD directory structure and return list of (image_path, fen) pairs.

    ChessReD has multiple possible structures. This function handles:
    1. CSV/JSON annotation files with image paths and FENs
    2. Directory-based structure with FEN in filename or metadata
    """
    pairs = []

    # Try: annotations.json or similar
    for ann_name in ['annotations.json', 'metadata.json', 'labels.json']:
        ann_path = os.path.join(chessred_dir, ann_name)
        if os.path.exists(ann_path):
            with open(ann_path) as f:
                data = json.load(f)
            if isinstance(data, list):
                for entry in data:
                    img_path = os.path.join(chessred_dir, entry.get('image', entry.get('filename', '')))
                    fen = entry.get('fen', entry.get('FEN', ''))
                    if os.path.exists(img_path) and fen:
                        # Extract just the position part of FEN
                        fen_position = fen.split(' ')[0]
                        pairs.append((img_path, fen_position))
            elif isinstance(data, dict):
                for img_name, fen in data.items():
                    img_path = os.path.join(chessred_dir, img_name)
                    if os.path.exists(img_path):
                        fen_position = fen.split(' ')[0] if isinstance(fen, str) else fen.get('fen', '').split(' ')[0]
                        if fen_position:
                            pairs.append((img_path, fen_position))
            if pairs:
                return pairs

    # Try: CSV annotation file
    for csv_name in ['annotations.csv', 'metadata.csv', 'labels.csv']:
        csv_path = os.path.join(chessred_dir, csv_name)
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                header = f.readline().strip().split(',')
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        row_dict = dict(zip(header, parts))
                        img_name = row_dict.get('image', row_dict.get('filename', parts[0]))
                        fen = row_dict.get('fen', row_dict.get('FEN', parts[-1]))
                        img_path = os.path.join(chessred_dir, img_name)
                        if not os.path.exists(img_path):
                            # Try subdirectories
                            for sub in os.listdir(chessred_dir):
                                candidate = os.path.join(chessred_dir, sub, img_name)
                                if os.path.exists(candidate):
                                    img_path = candidate
                                    break
                        if os.path.exists(img_path) and fen:
                            fen_position = fen.split(' ')[0]
                            pairs.append((img_path, fen_position))
            if pairs:
                return pairs

    # Try: directory walk — look for images alongside .txt or .fen files
    for root, dirs, files in os.walk(chessred_dir):
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_file in image_files:
            base = os.path.splitext(img_file)[0]
            # Look for FEN in companion text file
            for ext in ['.txt', '.fen']:
                fen_file = os.path.join(root, base + ext)
                if os.path.exists(fen_file):
                    with open(fen_file) as f:
                        fen = f.read().strip().split(' ')[0]
                    if '/' in fen:  # Basic FEN validation
                        pairs.append((os.path.join(root, img_file), fen))
                    break

    return pairs


def prepare_dataset(chessred_dir, output_dir, val_split=0.2, seed=42):
    """Main preparation function."""
    print(f"Scanning ChessReD directory: {chessred_dir}")
    pairs = find_chessred_structure(chessred_dir)

    if not pairs:
        print("ERROR: Could not find any image+FEN pairs in the ChessReD directory.")
        print("Expected one of:")
        print("  - annotations.json / metadata.json with image paths and FEN strings")
        print("  - annotations.csv with image,fen columns")
        print("  - Image files with companion .txt/.fen files containing FEN strings")
        return

    print(f"Found {len(pairs)} board images with FEN annotations")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - val_split))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Create output directories
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for split in ['train', 'val']:
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    class_counts = Counter()

    for split_name, split_pairs in [('train', train_pairs), ('val', val_pairs)]:
        print(f"\nProcessing {split_name} split...")
        for idx, (img_path, fen_position) in enumerate(tqdm(split_pairs)):
            img = cv2.imread(img_path)
            if img is None:
                print(f"  Warning: could not read {img_path}, skipping")
                continue

            board = fen_to_board(fen_position)
            if len(board) != 8 or any(len(r) != 8 for r in board):
                print(f"  Warning: invalid FEN '{fen_position}' for {img_path}, skipping")
                continue

            patches = crop_uniform_grid(img)

            for row in range(8):
                for col in range(8):
                    piece = board[row][col]
                    if piece not in CLASS_NAMES:
                        print(f"  Warning: unknown piece '{piece}' in FEN, skipping square")
                        continue

                    patch = patches[row][col]
                    # Resize to 50×50 if not already
                    if patch.shape[0] != 50 or patch.shape[1] != 50:
                        patch = cv2.resize(patch, (50, 50))

                    files = "abcdefgh"
                    ranks = "87654321"
                    square_name = f"{files[col]}{ranks[row]}"

                    out_path = os.path.join(
                        output_dir, split_name, piece,
                        f"{idx:05d}_{square_name}.jpg"
                    )
                    cv2.imwrite(out_path, patch)
                    class_counts[piece] += 1

    # Print stats
    print("\n--- Class Distribution ---")
    total = sum(class_counts.values())
    for cls in CLASS_NAMES:
        count = class_counts.get(cls, 0)
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {cls:6s}: {count:7d}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':6s}: {total:7d}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare ChessReD dataset for training')
    parser.add_argument('--chessred_dir', required=True, help='Path to ChessReD dataset root')
    parser.add_argument('--output_dir', default='data/chessred_patches', help='Output directory')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    prepare_dataset(args.chessred_dir, args.output_dir, args.val_split, args.seed)
