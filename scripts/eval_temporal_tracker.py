#!/usr/bin/env python3
"""Evaluate TemporalBoardTracker on ChessReD test games.

Compares three decoders on the same per-frame softmax sequence:

  1. argmax-only        — raw classifier argmax per frame; moves decoded by
                          greedy FEN-diff against the python-chess legal set
  2. detect_moves_with_prior  — Bayesian prior on the diff mask, stateless
                                across frames
  3. TemporalBoardTracker     — change-mask gating + inventory constraints +
                                Bayesian prior, stateful

Reports:
  - per-frame full-board accuracy
  - move-detection correctness (vs. ground-truth SAN derived from consecutive
    GT FENs)
  - failure mode breakdown (wrong / failed / missed)

Usage:
    python scripts/eval_temporal_tracker.py --games 0 33 76 \
        --checkpoint models/chess_piece_classifier_v2.pth
"""

import argparse
import json
import os
import sys
import time
import cv2
import chess
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.inference import ChessPieceClassifier
from src.pipeline.fen_generator import predictions_to_fen, fen_position_only
from src.pipeline.move_detector import (
    TemporalBoardTracker,
    detect_moves_sequence_with_prior,
    detect_moves_sequence_with_feedback,
    detect_move_with_prior,
    fen_to_piece_map,
    FEN_CLASSES,
)

_CAT = {0:'P',1:'R',2:'N',3:'B',4:'Q',5:'K',
        6:'p',7:'r',8:'n',9:'b',10:'q',11:'k'}
_SQ = [f"{f}{r}" for r in "87654321" for f in "abcdefgh"]
_FEN_CLASS_TO_IDX = {p: i for i, p in enumerate(FEN_CLASSES)}
PAD = 6   # matches prepare_chessred_v2.PATCH_PADDING


def build_gt_index(ann):
    idx = {}
    for a in ann['annotations']['pieces']:
        p = _CAT.get(a['category_id'])
        if p is None:
            continue
        idx.setdefault(a['image_id'], {})[a['chessboard_position']] = p
    return idx


def crop_squares(bgr, corners):
    src = np.float32([corners['top_left'], corners['top_right'],
                      corners['bottom_right'], corners['bottom_left']])
    dst = np.float32([[0, 0], [399, 0], [399, 399], [0, 399]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(bgr, M, (400, 400))
    patches = {}
    for ri, r in enumerate("87654321"):
        for ci, f in enumerate("abcdefgh"):
            y1 = max(0, ri * 50 - PAD)
            y2 = min(400, (ri + 1) * 50 + PAD)
            x1 = max(0, ci * 50 - PAD)
            x2 = min(400, (ci + 1) * 50 + PAD)
            patches[f"{f}{r}"] = warped[y1:y2, x1:x2]
    return patches


def gt_to_fen(gt_map):
    rows = []
    for rank in "87654321":
        empty = 0; row = ""
        for f in "abcdefgh":
            p = gt_map.get(f"{f}{rank}")
            if p is None:
                empty += 1
            else:
                if empty: row += str(empty); empty = 0
                row += p
        if empty: row += str(empty)
        rows.append(row)
    return "/".join(rows)


def gt_moves_from_fens(fens):
    board = chess.Board()
    prev = board.fen().split()[0]
    out = []
    for f in fens[1:]:
        if f == prev:
            out.append(None); continue
        found = None
        for m in board.legal_moves:
            board.push(m)
            if board.fen().split()[0] == f:
                board.pop()
                found = board.san(m); board.push(m); break
            board.pop()
        if found is None:
            out.append(None); continue
        out.append(found); prev = f
    return out


def count_correct(detected, gt):
    """Align detected moves to GT (skipping GT Nones), count SAN matches."""
    gi = di = c = 0
    while gi < len(gt) and di < len(detected):
        if gt[gi] is None:
            gi += 1; continue
        if detected[di] == gt[gi]:
            c += 1
        gi += 1; di += 1
    return c


def argmax_boards_accuracy(probs_list, gt_fens):
    """Per-frame full-board accuracy using raw argmax."""
    n_exact = n_ge_90 = n = 0
    sq_tot = sq_corr = 0
    for probs, gt_fen in zip(probs_list, gt_fens):
        top1 = {sq: FEN_CLASSES[int(np.argmax(p))] for sq, p in probs.items()}
        # Compare against gt_fen (piece-char per square)
        from src.pipeline.move_detector import fen_to_piece_map
        gt_map = fen_to_piece_map(gt_fen)
        sq_ok = sum(1 for sq in _SQ if top1[sq] == gt_map.get(sq, 'empty'))
        sq_tot += 64; sq_corr += sq_ok
        if sq_ok == 64: n_exact += 1
        if sq_ok >= 58: n_ge_90 += 1  # 58/64 ≈ 90.6%
        n += 1
    return {
        'per_sq_acc_pct': sq_corr / sq_tot * 100 if sq_tot else 0,
        'exact_pct': n_exact / n * 100 if n else 0,
        'ge_90_pct': n_ge_90 / n * 100 if n else 0,
        'n_boards': n,
    }


def state_sequence_accuracy(probs_list, gt_fens, prior_weight=1.5):
    """Evaluate exact board accuracy after legal-move Bayesian smoothing."""
    if not probs_list:
        return {'per_sq_acc_pct': 0, 'exact_pct': 0, 'ge_90_pct': 0, 'n_boards': 0}

    board = chess.Board()
    prev_pos = fen_position_only(predictions_to_fen({
        sq: (FEN_CLASSES[int(np.argmax(p))], float(p[int(np.argmax(p))]))
        for sq, p in probs_list[0].items()
    }))
    predicted_fens = [prev_pos]

    for probs in probs_list[1:]:
        argmax_fen = fen_position_only(predictions_to_fen({
            sq: (FEN_CLASSES[int(np.argmax(p))], float(p[int(np.argmax(p))]))
            for sq, p in probs.items()
        }))
        if argmax_fen != prev_pos:
            san, _, _ = detect_move_with_prior(prev_pos, probs, board,
                                               prior_weight=prior_weight)
            if san:
                try:
                    move = board.parse_san(san)
                except ValueError:
                    move = None
                if move:
                    board.push(move)
                    prev_pos = board.fen().split(' ')[0]
        predicted_fens.append(prev_pos)

    pred_probs = []
    for fen_pos in predicted_fens:
        board_map = fen_to_piece_map(fen_pos)
        pred_probs.append({
            sq: np.eye(len(FEN_CLASSES), dtype=np.float32)[_FEN_CLASS_TO_IDX[board_map.get(sq, 'empty')]]
            for sq in _SQ
        })
    return argmax_boards_accuracy(pred_probs, gt_fens)


def fuse_softmax_by_gt_fen(probs_list, gt_fens):
    """Average softmaxes over contiguous frames that share the same GT FEN.

    This simulates stable-period multi-frame fusion for ChessReD evaluation:
    frames with the same board state are equivalent to repeated observations
    between moves.
    """
    fused = []
    i = 0
    while i < len(probs_list):
        j = i + 1
        while j < len(probs_list) and gt_fens[j] == gt_fens[i]:
            j += 1

        segment = probs_list[i:j]
        avg = {}
        for sq in _SQ:
            avg[sq] = np.mean([p[sq] for p in segment], axis=0)
        fused.extend({sq: avg[sq].copy() for sq in _SQ} for _ in segment)
        i = j
    return fused


def run_tracker(probs_list):
    tracker = TemporalBoardTracker()
    moves = []; tags = []
    for probs in probs_list:
        san, tag = tracker.push(probs)
        if san is not None:
            moves.append(san); tags.append(tag)
    return moves, tags


def evaluate_game(clf, frames, corner_index, gt_index, images_base,
                  manual_corners=None, fusion='none'):
    """Run all three decoders on one game's frame sequence."""
    probs_list = []; gt_fens = []
    t0 = time.time()
    for im in frames:
        img_path = os.path.join(images_base, str(im['game_id']), im['file_name'])
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        # Use per-frame corner if available, else fall back to per-game manual corner
        if im['id'] in corner_index:
            corners = corner_index[im['id']]
        elif manual_corners and im['game_id'] in manual_corners:
            corners = manual_corners[im['game_id']]
        else:
            continue
        patches = crop_squares(bgr, corners)
        probs = clf.predict_board_full_probs(patches)
        probs_list.append(probs)
        gt_fens.append(gt_to_fen(gt_index.get(im['id'], {})))
    inf_t = time.time() - t0

    if not probs_list:
        return None

    eval_probs = fuse_softmax_by_gt_fen(probs_list, gt_fens) if fusion == 'gt_fen' else probs_list

    acc = argmax_boards_accuracy(eval_probs, gt_fens)
    state_acc = state_sequence_accuracy(eval_probs, gt_fens)
    gt_moves = gt_moves_from_fens(gt_fens)
    n_real = sum(1 for m in gt_moves if m is not None)

    # Decoder A: stateless Bayesian prior
    bp = detect_moves_sequence_with_prior(eval_probs, prior_weight=1.5)
    bp_correct = count_correct(bp['moves'], gt_moves)

    # Decoder B: feedback-correction (argmax-based)
    top1_list = [{sq: (FEN_CLASSES[int(np.argmax(p))], float(p[int(np.argmax(p))]))
                  for sq, p in probs.items()} for probs in eval_probs]
    fb = detect_moves_sequence_with_feedback(
        top1_list, max_adjustments=6, consensus_window=5)
    fb_correct = count_correct(fb['moves'], gt_moves)

    # Decoder C: temporal tracker
    tr_moves, tr_tags = run_tracker(eval_probs)
    tr_correct = count_correct(tr_moves, gt_moves)

    return {
        'frames': len(probs_list),
        'fusion': fusion,
        'gt_moves': n_real,
        'acc': acc,
        'state_acc': state_acc,
        'inference_time_s': inf_t,
        'feedback': {'det': len(fb['moves']), 'correct': fb_correct,
                     'correctness_pct': fb_correct / max(1, n_real) * 100},
        'bayes_prior': {'det': len(bp['moves']), 'correct': bp_correct,
                        'correctness_pct': bp_correct / max(1, n_real) * 100},
        'tracker': {'det': len(tr_moves), 'correct': tr_correct,
                    'tags': dict((t, tr_tags.count(t)) for t in set(tr_tags)),
                    'correctness_pct': tr_correct / max(1, n_real) * 100},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='models/chess_piece_classifier_v2.pth')
    parser.add_argument('--games', type=int, nargs='*', default=None,
                        help='Game IDs to evaluate (default: all with corners)')
    parser.add_argument('--tta', type=int, default=4)
    parser.add_argument('--annotations', default='data/annotations.json')
    parser.add_argument('--manual_corners', default='data/manual_corners.json',
                        help='Per-game corners (key=game_id) from corner_annotator.py')
    parser.add_argument('--per_frame_corners', default='data/manual_corners_per_frame.json',
                        help='Per-frame corners (key=image_id) from corner_annotator_per_frame.py')
    parser.add_argument('--auto_corners', default='data/auto_corners.json',
                        help='ML corner detector output (key=image_id) from auto_annotate_corners.py')
    parser.add_argument('--corner_source', default='auto_prefer',
                        choices=['current', 'auto', 'auto_prefer'],
                        help='current=official/manual fallback, auto=ML corners only, auto_prefer=ML then current')
    parser.add_argument('--images_base', default='data/ChessRed_images')
    parser.add_argument('--out', default='stats_output_model/temporal_eval.json')
    parser.add_argument('--fusion', default='none', choices=['none', 'gt_fen'],
                        help='Softmax fusion mode for model-accuracy evaluation')
    args = parser.parse_args()

    with open(args.annotations) as f:
        ann = json.load(f)
    gt_index = build_gt_index(ann)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    T = ckpt.get('temperature', 1.0)
    print(f"Using {args.checkpoint}  (T={T:.4f}, TTA={args.tta})")
    clf = ChessPieceClassifier(model_path=args.checkpoint,
                               temperature=T, tta_views=args.tta)

    split_ids = set(ann['splits']['test']['image_ids']) | set(ann['splits']['val']['image_ids'])
    id_to_img = {im['id']: im for im in ann['images']}
    # Per-frame corners from official annotations.
    official_corner_index = {c['image_id']: c['corners']
                             for c in ann['annotations']['corners']}
    corner_index = dict(official_corner_index)
    # Per-frame manual corners (key = image_id) — overrides nothing, fills gaps
    if os.path.exists(args.per_frame_corners):
        with open(args.per_frame_corners) as f:
            pfc = json.load(f)
        added = 0
        for iid_str, corners in pfc.items():
            iid = int(iid_str)
            if iid not in corner_index:
                corner_index[iid] = corners
                added += 1
        print(f"Loaded per-frame manual corners: {added} new frames from {args.per_frame_corners}")
    auto_corner_index = {}
    if os.path.exists(args.auto_corners):
        with open(args.auto_corners) as f:
            ac = json.load(f)
        auto_corner_index = {int(iid_str): corners for iid_str, corners in ac.items()}
        print(f"Loaded ML auto corners: {len(auto_corner_index)} frames from {args.auto_corners}")

    if args.corner_source == 'auto':
        corner_index = dict(auto_corner_index)
    elif args.corner_source == 'auto_prefer':
        merged = dict(corner_index)
        merged.update(auto_corner_index)
        corner_index = merged
    # Per-game manual corners (key = game_id) — applied to all frames of that game
    manual_corners = {}
    if os.path.exists(args.manual_corners):
        with open(args.manual_corners) as f:
            mc = json.load(f)
        for gid_str, corners in mc.items():
            manual_corners[int(gid_str)] = corners
        print(f"Loaded per-game manual corners for {len(manual_corners)} games: {sorted(manual_corners)}")

    games = {}
    for iid in split_ids:
        im = id_to_img[iid]
        gid = im['game_id']
        if args.games is not None and gid not in args.games:
            continue
        # Accept frame if it has per-frame corners OR its game has manual corners
        if iid not in corner_index and gid not in manual_corners:
            continue
        games.setdefault(gid, []).append(im)

    print(f"\n{'Game':<5}{'Frames':>7}{'GT':>5}"
          f"{'sq%':>7}{'exact':>7}{'FB':>6}{'BP':>6}{'TR':>6}")
    print('-' * 55)

    totals = {'frames': 0, 'gt': 0,
              'fb_c': 0, 'bp_c': 0, 'tr_c': 0,
              'exact': 0.0, 'sq': 0.0, 'ge90': 0.0,
              'state_exact': 0.0, 'state_sq': 0.0, 'state_ge90': 0.0,
              'nb': 0}
    per_game = {}
    for gid in sorted(games):
        frames = sorted(games[gid], key=lambda x: x['move_id'])
        r = evaluate_game(clf, frames, corner_index, gt_index,
                          args.images_base, manual_corners, fusion=args.fusion)
        if r is None:
            continue
        per_game[gid] = r
        print(f"{gid:<5}{r['frames']:>7}{r['gt_moves']:>5}"
              f"{r['acc']['per_sq_acc_pct']:>6.1f}%{r['acc']['exact_pct']:>6.1f}%"
              f"{r['feedback']['correctness_pct']:>5.0f}%"
              f"{r['bayes_prior']['correctness_pct']:>5.0f}%"
              f"{r['tracker']['correctness_pct']:>5.0f}%")
        totals['frames'] += r['frames']
        totals['gt'] += r['gt_moves']
        totals['fb_c'] += r['feedback']['correct']
        totals['bp_c'] += r['bayes_prior']['correct']
        totals['tr_c'] += r['tracker']['correct']
        totals['sq'] += r['acc']['per_sq_acc_pct'] * r['frames']
        totals['exact'] += r['acc']['exact_pct'] * r['frames']
        totals['ge90'] += r['acc']['ge_90_pct'] * r['frames']
        totals['state_sq'] += r['state_acc']['per_sq_acc_pct'] * r['frames']
        totals['state_exact'] += r['state_acc']['exact_pct'] * r['frames']
        totals['state_ge90'] += r['state_acc']['ge_90_pct'] * r['frames']
        totals['nb'] += r['frames']

    print('-' * 55)
    if totals['nb']:
        print(f"{'TOT':<5}{totals['frames']:>7}{totals['gt']:>5}"
              f"{totals['sq']/totals['nb']:>6.1f}%"
              f"{totals['exact']/totals['nb']:>6.1f}%"
              f"{totals['fb_c']/max(1,totals['gt'])*100:>5.0f}%"
              f"{totals['bp_c']/max(1,totals['gt'])*100:>5.0f}%"
              f"{totals['tr_c']/max(1,totals['gt'])*100:>5.0f}%")

    print()
    print(f"{'Decoder':<30}{'Moves detected':>16}{'Correct':>10}{'Rate':>8}")
    print('-' * 64)
    print(f"{'Feedback correction':<30}"
          f"{sum(r['feedback']['det'] for r in per_game.values()):>16}"
          f"{totals['fb_c']:>10}"
          f"{totals['fb_c']/max(1,totals['gt'])*100:>7.1f}%")
    print(f"{'Bayesian prior (stateless)':<30}"
          f"{sum(r['bayes_prior']['det'] for r in per_game.values()):>16}"
          f"{totals['bp_c']:>10}"
          f"{totals['bp_c']/max(1,totals['gt'])*100:>7.1f}%")
    print(f"{'TemporalBoardTracker':<30}"
          f"{sum(r['tracker']['det'] for r in per_game.values()):>16}"
          f"{totals['tr_c']:>10}"
          f"{totals['tr_c']/max(1,totals['gt'])*100:>7.1f}%")

    out_payload = {
        'checkpoint': args.checkpoint,
        'temperature': T, 'tta_views': args.tta,
        'fusion': args.fusion,
        'corner_source': args.corner_source,
        'games': args.games,
        'per_game': per_game,
        'totals': {
            'frames': totals['frames'], 'gt_moves': totals['gt'],
            'per_sq_acc_pct': totals['sq'] / max(1, totals['nb']),
            'exact_pct':     totals['exact'] / max(1, totals['nb']),
            'ge_90_pct':     totals['ge90'] / max(1, totals['nb']),
            'state_per_sq_acc_pct': totals['state_sq'] / max(1, totals['nb']),
            'state_exact_pct': totals['state_exact'] / max(1, totals['nb']),
            'state_ge_90_pct': totals['state_ge90'] / max(1, totals['nb']),
            'fb_correctness_pct': totals['fb_c'] / max(1, totals['gt']) * 100,
            'bp_correctness_pct': totals['bp_c'] / max(1, totals['gt']) * 100,
            'tr_correctness_pct': totals['tr_c'] / max(1, totals['gt']) * 100,
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out_payload, f, indent=2, default=float)
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
