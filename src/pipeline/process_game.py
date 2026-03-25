"""
End-to-end pipeline: process a captured game session into PGN.

  S3 session folder → load images → crop squares → classify → FEN → moves → PGN

Usage:
  python -m src.pipeline.process_game --game_id <id> --model_path models/chess_piece_classifier.pth
  python -m src.pipeline.process_game --local_dir /path/to/session --model_path models/chess_piece_classifier.pth
"""

import os
import io
import json
import re
import argparse
import cv2
import numpy as np

from src.preprocessing.process_board import crop_squares_from_grid
from src.models.inference import ChessPieceClassifier
from src.pipeline.fen_generator import predictions_to_fen, fen_position_only
from src.pipeline.move_detector import detect_moves_sequence
from src.pipeline.pgn_generator import generate_pgn, save_pgn


def load_session_from_local(session_dir: str) -> dict:
    """Load a game session from a local directory.

    Returns dict with 'game_info', 'images' (list of numpy arrays in order).
    """
    # Load game_info.json
    info_path = os.path.join(session_dir, 'game_info.json')
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"game_info.json not found in {session_dir}")

    with open(info_path) as f:
        game_info = json.load(f)

    # Find warped images — check warped/ subfolder first, then root
    warped_dir = os.path.join(session_dir, 'warped')
    if os.path.isdir(warped_dir):
        img_dir = warped_dir
    else:
        img_dir = session_dir

    # Collect image files sorted by name (00.jpg, 01.jpg, ...)
    image_files = sorted([
        f for f in os.listdir(img_dir)
        if re.match(r'^\d+\.(jpg|jpeg|png)$', f, re.IGNORECASE)
    ], key=lambda x: int(os.path.splitext(x)[0]))

    images = []
    for img_file in image_files:
        img = cv2.imread(os.path.join(img_dir, img_file))
        if img is not None:
            images.append(img)

    return {'game_info': game_info, 'images': images}


def load_session_from_s3(game_id: str, bucket: str = None) -> dict:
    """Load a game session from S3.

    Returns dict with 'game_info', 'images' (list of numpy arrays in order).
    """
    import boto3
    from dotenv import load_dotenv
    load_dotenv()

    bucket = bucket or os.getenv('S3_BUCKET_NAME', 'chess-capture-bucket')
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'us-east-1'),
    )

    prefix = f"sessions/{game_id}/"

    # Load game_info.json
    info_key = f"{prefix}game_info.json"
    resp = s3.get_object(Bucket=bucket, Key=info_key)
    game_info = json.loads(resp['Body'].read().decode())

    # List warped images
    warped_prefix = f"{prefix}warped/"
    paginator = s3.get_paginator('list_objects_v2')

    image_keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=warped_prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            filename = key.split('/')[-1]
            if re.match(r'^\d+\.(jpg|jpeg|png)$', filename, re.IGNORECASE):
                image_keys.append(key)

    # Fallback: check root prefix (old flat structure)
    if not image_keys:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                filename = key.split('/')[-1]
                if re.match(r'^\d+\.(jpg|jpeg|png)$', filename, re.IGNORECASE):
                    image_keys.append(key)

    # Sort by numeric filename
    image_keys.sort(key=lambda k: int(os.path.splitext(k.split('/')[-1])[0]))

    images = []
    for key in image_keys:
        resp = s3.get_object(Bucket=bucket, Key=key)
        data = resp['Body'].read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)

    return {'game_info': game_info, 'images': images}


def process_game_session(game_id: str = None, local_dir: str = None,
                         model_path: str = 'models/chess_piece_classifier.pth',
                         s3_bucket: str = None, result: str = '*') -> dict:
    """Full pipeline: session images → PGN.

    Parameters
    ----------
    game_id    : str — S3 session ID (used if local_dir is None)
    local_dir  : str — local session directory (takes priority over game_id)
    model_path : str — path to trained classifier checkpoint
    s3_bucket  : str — S3 bucket name (optional, uses env var default)
    result     : str — game result for PGN header

    Returns
    -------
    dict with 'pgn', 'moves', 'fen_sequence', 'errors', 'skipped'
    """
    # 1. Load session
    if local_dir:
        session = load_session_from_local(local_dir)
    elif game_id:
        session = load_session_from_s3(game_id, s3_bucket)
    else:
        raise ValueError("Provide either game_id or local_dir")

    game_info = session['game_info']
    images = session['images']

    if not images:
        raise ValueError("No images found in session")

    # 2. Get grid and rotation from game_info
    grid = game_info.get('board_grid')
    if not grid:
        # Fallback: uniform 50px grid
        grid = {
            'x_lines': [i * 50 for i in range(9)],
            'y_lines': [i * 50 for i in range(9)],
        }

    rotation = game_info.get('rotation_angle', 0)

    # Use result from game_info if available and not overridden
    if result == '*' and game_info.get('result'):
        result = game_info['result']

    # 3. Load classifier
    classifier = ChessPieceClassifier(model_path=model_path)
    print(f"Loaded classifier from {model_path}")

    # 4. Process each image → FEN
    fen_sequence = []
    print(f"Processing {len(images)} board images...")

    for i, img in enumerate(images):
        patches = crop_squares_from_grid(img, grid, rotation)
        predictions = classifier.predict_board(patches)
        fen = predictions_to_fen(predictions, move_number=i + 1)
        fen_pos = fen_position_only(fen)
        fen_sequence.append(fen_pos)

        if i % 10 == 0:
            print(f"  Image {i:3d}: {fen_pos[:30]}...")

    # 5. Detect moves from FEN sequence
    print("Detecting moves...")
    move_result = detect_moves_sequence(fen_sequence)

    moves = move_result['moves']
    errors = move_result['errors']
    skipped = move_result['skipped']

    print(f"  Detected {len(moves)} moves, {len(errors)} errors, {skipped} identical frames skipped")

    # 6. Generate PGN
    pgn = generate_pgn(moves, game_info, result)

    return {
        'pgn': pgn,
        'moves': moves,
        'fen_sequence': fen_sequence,
        'errors': errors,
        'skipped': skipped,
    }


def main():
    parser = argparse.ArgumentParser(description='Process a chess game session into PGN')
    parser.add_argument('--game_id', help='S3 session game ID')
    parser.add_argument('--local_dir', help='Local session directory')
    parser.add_argument('--model_path', default='models/chess_piece_classifier.pth')
    parser.add_argument('--s3_bucket', help='S3 bucket name')
    parser.add_argument('--result', default='*', help='Game result (1-0, 0-1, 1/2-1/2, *)')
    parser.add_argument('--output', help='Output PGN file path')
    args = parser.parse_args()

    result = process_game_session(
        game_id=args.game_id,
        local_dir=args.local_dir,
        model_path=args.model_path,
        s3_bucket=args.s3_bucket,
        result=args.result,
    )

    print("\n--- Generated PGN ---")
    print(result['pgn'])

    if args.output:
        save_pgn(result['pgn'], args.output)
        print(f"\nSaved to {args.output}")

    if result['errors']:
        print(f"\n--- {len(result['errors'])} Errors ---")
        for err in result['errors']:
            print(f"  Image {err['index']}: {err['reason']}")


if __name__ == '__main__':
    main()
