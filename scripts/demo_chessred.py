#!/usr/bin/env python3
"""
End-to-end demo using the ChessReD dataset.

Loads ChessReD game images and annotations, runs the pretrained ResNeXt-101
classifier through the full pipeline, and produces:
  - demo_output/accuracy_report.txt  — per-frame accuracy vs ground truth
  - demo_output/demo_game.pgn        — reconstructed PGN from detected moves

Usage:
    python scripts/demo_chessred.py [--game_id 0] [--split test] [--max_moves 40]
"""

import argparse
import datetime
import json
import os
import sys

import cv2

# Add project root to path so src.* imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.models.inference import PretrainedBoardClassifier, ChessPieceClassifier
from src.pipeline.fen_generator import predictions_to_fen, fen_position_only
from src.pipeline.move_detector import detect_moves_sequence_with_feedback
from src.pipeline.pgn_generator import generate_pgn, save_pgn

# ChessReD category_id → FEN piece character
_CAT_ID_TO_PIECE = {
    0: 'P', 1: 'R', 2: 'N', 3: 'B', 4: 'Q', 5: 'K',   # white
    6: 'p', 7: 'r', 8: 'n', 9: 'b', 10: 'q', 11: 'k',  # black
    # 12 = empty, omitted
}

# All 64 squares in FEN order (rank 8 → 1, file a → h)
_ALL_SQUARES = [f"{f}{r}" for r in "87654321" for f in "abcdefgh"]


# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------

def build_gt_index(annotations: dict) -> dict:
    """Index piece annotations by image_id.

    Returns {image_id: {square: piece_char}} — empty squares omitted.
    """
    index = {}
    for ann in annotations['annotations']['pieces']:
        img_id = ann['image_id']
        piece = _CAT_ID_TO_PIECE.get(ann['category_id'])
        if piece is None:
            continue  # empty square
        square = ann['chessboard_position']
        if img_id not in index:
            index[img_id] = {}
        index[img_id][square] = piece
    return index


def gt_to_fen_position(square_map: dict) -> str:
    """Convert {square: piece_char} ground truth to a FEN position string."""
    rows = []
    for rank in "87654321":
        empty = 0
        row = ""
        for f in "abcdefgh":
            piece = square_map.get(f"{f}{rank}")
            if piece is None:
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                row += piece
        if empty:
            row += str(empty)
        rows.append(row)
    return "/".join(rows)


# ---------------------------------------------------------------------------
# Accuracy comparison
# ---------------------------------------------------------------------------

def compare_to_gt(predictions: dict, gt_square_map: dict) -> tuple:
    """Compare predictions to ground truth.

    Returns (correct_count, errors) where errors is a list of
    (square, gt_piece, pred_piece, confidence).
    """
    correct = 0
    errors = []
    for sq in _ALL_SQUARES:
        pred_piece, conf = predictions.get(sq, ('empty', 0.0))
        gt_piece = gt_square_map.get(sq, None)
        gt_str = gt_piece if gt_piece else 'empty'
        if pred_piece == gt_str:
            correct += 1
        else:
            errors.append((sq, gt_str, pred_piece, conf))
    return correct, errors


# ---------------------------------------------------------------------------
# Game selection
# ---------------------------------------------------------------------------

def pick_game(annotations: dict, split: str, game_id: int | None) -> tuple:
    """Return (game_id, sorted list of image dicts) for the chosen game.

    If game_id is None, picks the first complete game in the split.
    """
    split_ids = set(annotations['splits'][split]['image_ids'])
    id_to_img = {img['id']: img for img in annotations['images']}

    # Group split images by game_id
    games: dict[int, list] = {}
    for img_id in split_ids:
        img = id_to_img[img_id]
        gid = img['game_id']
        games.setdefault(gid, []).append(img)

    if game_id is not None:
        if game_id not in games:
            raise ValueError(
                f"game_id={game_id} not found in '{split}' split. "
                f"Available game IDs (first 10): {sorted(games)[:10]}"
            )
        frames = sorted(games[game_id], key=lambda x: x['move_id'])
        return game_id, frames

    # Pick the game with the most frames for a richer demo
    best_gid = max(games, key=lambda g: len(games[g]))
    frames = sorted(games[best_gid], key=lambda x: x['move_id'])
    return best_gid, frames


# ---------------------------------------------------------------------------
# Patch cropping helper (for patch-based classifier)
# ---------------------------------------------------------------------------

def _crop_squares_from_corners(bgr: np.ndarray, corners: dict) -> dict:
    """Warp the board to 400x400 and return 64 square patches (50x50 each).

    Parameters
    ----------
    bgr : np.ndarray  Full camera image
    corners : dict    {'top_left': [x,y], 'top_right': [x,y],
                       'bottom_left': [x,y], 'bottom_right': [x,y]}

    Returns
    -------
    dict {square_name: np.ndarray (50x50 BGR patch)}
    """
    src = np.array([
        corners['top_left'],
        corners['top_right'],
        corners['bottom_right'],
        corners['bottom_left'],
    ], dtype=np.float32)
    # Warp to 400x400 (top-left = a8, top-right = h8, bottom-right = h1, bottom-left = a1)
    BOARD_SIZE = 400
    SQ = BOARD_SIZE // 8  # 50
    dst = np.array([
        [0, 0],
        [BOARD_SIZE, 0],
        [BOARD_SIZE, BOARD_SIZE],
        [0, BOARD_SIZE],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(bgr, M, (BOARD_SIZE, BOARD_SIZE))

    patches = {}
    for row_i, rank in enumerate("87654321"):
        for col_i, file in enumerate("abcdefgh"):
            r0, r1 = row_i * SQ, (row_i + 1) * SQ
            c0, c1 = col_i * SQ, (col_i + 1) * SQ
            patches[f"{file}{rank}"] = warped[r0:r1, c0:c1]
    return patches


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_demo(args):
    output_dir = "demo_output"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 62)
    print("  ChessReD End-to-End Demo")
    print("=" * 62)

    # Load annotations
    ann_path = "data/annotations.json"
    images_base = "data/ChessRed_images"
    ckpt_path = args.checkpoint

    print(f"\nLoading annotations from {ann_path} ...")
    with open(ann_path) as f:
        annotations = json.load(f)

    gt_index = build_gt_index(annotations)
    print(f"  Indexed GT for {len(gt_index)} images")

    # Select game
    chosen_gid, frames = pick_game(annotations, args.split, args.game_id)
    frames = frames[:args.max_moves + 1]  # +1 because N frames → N-1 moves
    print(f"\nSelected game_id={chosen_gid} from '{args.split}' split")
    print(f"  Frames: {len(frames)} ({len(frames)-1} potential moves)")

    # Load classifier
    use_patch = (args.classifier == 'patch')
    if use_patch:
        print(f"\nLoading patch classifier from {args.patch_checkpoint} ...")
        clf = ChessPieceClassifier(model_path=args.patch_checkpoint)
        print(f"  Device: {clf.device}")
        # Build corner index for square cropping
        corner_index = {c['image_id']: c['corners'] for c in annotations['annotations']['corners']}
        print(f"  Corner annotations: {len(corner_index)} images")
    else:
        print(f"\nLoading pretrained classifier from {ckpt_path} ...")
        clf = PretrainedBoardClassifier(checkpoint_path=ckpt_path)
        print(f"  Device: {clf.device}")

    # Run inference frame by frame
    print(f"\nRunning inference on {len(frames)} frames ...")
    print("-" * 62)
    print(f"{'Frame':>5}  {'Image':>22}  {'Correct':>7}  {'Acc%':>6}  {'Status'}")
    print("-" * 62)

    frame_results = []
    all_predictions = []   # raw {sq: (piece, conf)} per frame — used for feedback detection

    for i, img_meta in enumerate(frames):
        img_id = img_meta['id']
        game_id_dir = str(img_meta['game_id'])
        img_path = os.path.join(images_base, game_id_dir, img_meta['file_name'])

        # Load image
        if not os.path.exists(img_path):
            print(f"{i:>5}  {img_meta['file_name']:>22}  {'—':>7}  {'—':>6}  MISSING")
            all_predictions.append(None)
            frame_results.append(None)
            continue

        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"{i:>5}  {img_meta['file_name']:>22}  {'—':>7}  {'—':>6}  READ ERROR")
            all_predictions.append(None)
            frame_results.append(None)
            continue

        # Predict — keep raw predictions (with confidence) for the feedback loop
        if use_patch:
            corners = corner_index.get(img_id)
            if corners is None:
                print(f"{i:>5}  {img_meta['file_name']:>22}  {'—':>7}  {'—':>6}  NO CORNERS")
                all_predictions.append(None)
                frame_results.append(None)
                continue
            patches = _crop_squares_from_corners(bgr, corners)
            predictions = clf.predict_board(patches)
        else:
            predictions = clf.predict_full_board(bgr)
        pred_pos = fen_position_only(predictions_to_fen(predictions, move_number=i + 1))
        all_predictions.append(predictions)

        # Compare to GT
        gt_sq = gt_index.get(img_id, {})
        gt_pos = gt_to_fen_position(gt_sq)
        correct, errors = compare_to_gt(predictions, gt_sq)
        pct = 100.0 * correct / 64

        status = "OK" if pct >= 90 else ("~" if pct >= 75 else "!!")
        print(f"{i:>5}  {img_meta['file_name']:>22}  {correct:>3}/64  {pct:>5.1f}%  {status}")

        frame_results.append({
            'frame': i,
            'img_id': img_id,
            'file': img_meta['file_name'],
            'move_id': img_meta['move_id'],
            'correct': correct,
            'pct': pct,
            'gt_pos': gt_pos,
            'pred_pos': pred_pos,
            'errors': errors,
        })

    # Move detection with feedback loop + PGN
    valid_preds = [p for p in all_predictions if p is not None]
    print(f"\nDetecting moves with feedback loop ({len(valid_preds)} frames, max_adjustments={args.max_adjustments}) ...")
    seq_result = detect_moves_sequence_with_feedback(
        valid_preds,
        max_adjustments=args.max_adjustments,
        consensus_window=args.consensus_window,
        consensus_force_sync=not args.no_consensus_force_sync,
    )
    moves = seq_result['moves']
    move_tags = seq_result['move_tags']
    det_errors = seq_result['errors']
    resyncs = seq_result.get('resyncs', [])
    sure_count = sum(1 for t in move_tags if t in ('sure',))
    unsure_count = sum(1 for t in move_tags if 'unsure' in t)
    consensus_count = sum(1 for t in move_tags if t.startswith('consensus_'))
    print(f"  Detected: {len(moves)} moves  |  Sure: {sure_count}  |  Unsure: {unsure_count}  |  Consensus: {consensus_count}  |  Failed: {len(det_errors)}  |  Re-syncs: {len(resyncs)}  |  Skipped: {seq_result['skipped']}")

    game_info = {
        'event': f'ChessReD Demo — Game {chosen_gid}',
        'site': 'chess-to-pgn / ChessReD',
        'date': datetime.date.today().isoformat(),
        'round': '-',
        'white': 'White',
        'black': 'Black',
    }
    pgn_str = generate_pgn(moves, game_info, result='*', move_tags=move_tags)
    pgn_path = os.path.join(output_dir, "demo_game.pgn")
    save_pgn(pgn_str, pgn_path)
    print(f"  PGN saved: {pgn_path}")

    # Build accuracy report
    valid_results = [r for r in frame_results if r is not None]
    avg_acc = sum(r['pct'] for r in valid_results) / len(valid_results) if valid_results else 0
    above_90 = sum(1 for r in valid_results if r['pct'] >= 90)

    lines = []
    lines.append("ChessReD End-to-End Demo — Accuracy Report")
    lines.append("=" * 62)
    lines.append(f"Date:                {datetime.date.today()}")
    lines.append(f"Split:               {args.split}")
    lines.append(f"Game ID:             {chosen_gid}")
    lines.append(f"Frames processed:    {len(valid_results)}")
    lines.append(f"Moves detected:      {len(moves)}  (sure={sure_count}, unsure={unsure_count}, consensus={consensus_count})")
    lines.append(f"Move detect errors:  {len(det_errors)}")
    lines.append(f"Consensus re-syncs:  {len(resyncs)}")
    lines.append(f"Avg square accuracy: {avg_acc:.1f}%")
    lines.append(f"Frames >=90% acc:    {above_90}/{len(valid_results)}")
    lines.append(f"Max adjustments:     {args.max_adjustments}")
    lines.append(f"Consensus window:    {args.consensus_window}")
    lines.append(f"PGN:                 {pgn_path}")
    lines.append("")
    lines.append("Per-Frame Results:  [sure] = exact match  [unsure] = feedback correction applied")
    lines.append("-" * 62)

    # Build a per-frame move map from the sequence result
    # move_tags aligns with moves[], but frames may have been skipped
    move_idx = 0
    for i, r in enumerate(frame_results):
        if r is None:
            lines.append(f"Frame {i:2d}: [SKIPPED - image not found]")
            continue
        if i == 0:
            move_tag_str = ""
            move_str = "—"
        elif move_idx < len(moves):
            move_str = moves[move_idx]
            tag = move_tags[move_idx] if move_idx < len(move_tags) else 'failed'
            move_tag_str = f" [{tag}]"
            # Show what squares were adjusted if unsure
            adj = seq_result['adjustments'][move_idx] if move_idx < len(seq_result['adjustments']) else []
            if adj:
                adj_str = ", ".join(f"{sq}:{o}->{c}({cf:.0%})" for sq, o, c, cf in adj)
                move_tag_str += f" adjusted: {adj_str}"
            move_idx += 1
        else:
            move_str = "failed"
            move_tag_str = " [failed]"

        lines.append(
            f"Frame {i:2d} (id={r['img_id']:5d}, move_id={r['move_id']:3d}): "
            f"{r['correct']:2d}/64 = {r['pct']:5.1f}%  move={move_str}{move_tag_str}"
        )
        lines.append(f"  GT:   {r['gt_pos']}")
        lines.append(f"  Pred: {r['pred_pos']}")
        if r['errors']:
            err_parts = [f"{sq}:{gt}->{pr}({conf:.0%})" for sq, gt, pr, conf in r['errors'][:6]]
            suffix = f" (+{len(r['errors'])-6} more)" if len(r['errors']) > 6 else ""
            lines.append(f"  Classifier errors [{len(r['errors'])}]: {', '.join(err_parts)}{suffix}")

    if det_errors:
        lines.append("")
        lines.append("Move Detection Failures (board state held at last good position):")
        lines.append("-" * 62)
        for e in det_errors:
            lines.append(f"  Frame {e['index']}: {e['reason']}")

    if resyncs:
        lines.append("")
        lines.append("Consensus Re-sync Events:")
        lines.append("-" * 62)
        for rs in resyncs:
            lines.append(
                f"  Frame {rs['frame']:3d}: force-synced board  "
                f"(window={rs['window_size']}, diff={rs['diff_from_prev']} sq)"
            )
            lines.append(f"    Consensus pos: {rs['consensus_pos']}")

    report_str = "\n".join(lines)
    report_path = os.path.join(output_dir, "accuracy_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_str)

    # Final summary
    print()
    print("=" * 62)
    print("  RESULTS")
    print("=" * 62)
    print(f"  Avg square accuracy:   {avg_acc:.1f}%")
    print(f"  Frames >=90% accuracy: {above_90}/{len(valid_results)}")
    print(f"  Moves detected:        {len(moves)}  (sure={sure_count}, unsure={unsure_count}, consensus={consensus_count})")
    print(f"  Move failures:         {len(det_errors)}")
    print(f"  Consensus re-syncs:    {len(resyncs)}")
    print(f"  Accuracy report:       {report_path}")
    print(f"  PGN:                   {pgn_path}")
    print("=" * 62)

    if moves:
        print("\nPGN preview:")
        print(pgn_str[:500])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChessReD end-to-end demo')
    parser.add_argument('--game_id', type=int, default=None,
                        help='ChessReD game ID to use (default: longest game in split)')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to use (default: test)')
    parser.add_argument('--max_moves', type=int, default=40,
                        help='Max frames to process (default: 40)')
    parser.add_argument('--max_adjustments', type=int, default=6,
                        help='Max squares feedback loop may correct per move (default: 6)')
    parser.add_argument('--consensus_window', type=int, default=5,
                        help='Consecutive failures before consensus re-sync (default: 5)')
    parser.add_argument('--no_consensus_force_sync', action='store_true',
                        help='Disable hard board re-sync when consensus exceeds budget')
    parser.add_argument('--checkpoint', default='src/models/pretrained/checkpoint.ckpt',
                        help='Path to full-board model checkpoint (default: src/models/pretrained/checkpoint.ckpt)')
    parser.add_argument('--classifier', default='pretrained', choices=['pretrained', 'patch'],
                        help='Which classifier to use: pretrained (ResNeXt-101) or patch (ChessPieceCNN) (default: pretrained)')
    parser.add_argument('--patch_checkpoint', default='models/chess_piece_classifier.pth',
                        help='Path to patch classifier checkpoint (default: models/chess_piece_classifier.pth)')
    args = parser.parse_args()
    run_demo(args)
