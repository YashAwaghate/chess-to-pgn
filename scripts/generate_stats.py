#!/usr/bin/env python3
"""
Generate statistics and visualizations for the chess-to-pgn pipeline.

Evaluates our patch-based ChessPieceCNN classifier on available test games
(those with corner annotations for square cropping) and produces:
  - stats_output/accuracy_per_frame.png    — per-frame square accuracy over game
  - stats_output/game_phase_accuracy.png   — opening / middlegame / endgame accuracy
  - stats_output/move_detection.png        — move detection breakdown per game
  - stats_output/pgn_quality.png           — pie chart + detection rate
  - stats_output/accuracy_vs_phase.png     — scatter: accuracy vs game phase
  - stats_output/summary_stats.txt         — text summary

Usage:
    python scripts/generate_stats.py
    python scripts/generate_stats.py --games 0 33 76
    python scripts/generate_stats.py --max_moves 100
"""

import argparse
import datetime
import json
import os
import sys
from collections import defaultdict

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.inference import ChessPieceClassifier
from src.pipeline.fen_generator import predictions_to_fen, fen_position_only
from src.pipeline.move_detector import detect_moves_sequence_with_feedback
from src.pipeline.pgn_generator import generate_pgn

_CAT_ID_TO_PIECE = {
    0: 'P', 1: 'R', 2: 'N', 3: 'B', 4: 'Q', 5: 'K',
    6: 'p', 7: 'r', 8: 'n', 9: 'b', 10: 'q', 11: 'k',
}
_ALL_SQUARES = [f"{f}{r}" for r in "87654321" for f in "abcdefgh"]
_START_MAP = {
    'a1': 'R', 'b1': 'N', 'c1': 'B', 'd1': 'Q', 'e1': 'K', 'f1': 'B', 'g1': 'N', 'h1': 'R',
    'a2': 'P', 'b2': 'P', 'c2': 'P', 'd2': 'P', 'e2': 'P', 'f2': 'P', 'g2': 'P', 'h2': 'P',
    'a7': 'p', 'b7': 'p', 'c7': 'p', 'd7': 'p', 'e7': 'p', 'f7': 'p', 'g7': 'p', 'h7': 'p',
    'a8': 'r', 'b8': 'n', 'c8': 'b', 'd8': 'q', 'e8': 'k', 'f8': 'b', 'g8': 'n', 'h8': 'r',
}

PHASE_COLORS = {'Opening': '#27ae60', 'Middlegame': '#e67e22', 'Endgame': '#8e44ad'}
MODEL_COLOR  = '#2980b9'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_gt_index(annotations):
    index = {}
    for ann in annotations['annotations']['pieces']:
        img_id = ann['image_id']
        piece = _CAT_ID_TO_PIECE.get(ann['category_id'])
        if piece is None:
            continue
        index.setdefault(img_id, {})[ann['chessboard_position']] = piece
    return index


def compare_to_gt(predictions, gt_square_map):
    correct = 0
    for sq in _ALL_SQUARES:
        pred_piece, _ = predictions.get(sq, ('empty', 0.0))
        gt_piece = gt_square_map.get(sq)
        if pred_piece == (gt_piece if gt_piece else 'empty'):
            correct += 1
    return correct


def game_phase(gt_sq):
    """Return (changed_count, phase_label) based on squares changed from start."""
    changed = sum(1 for sq in _ALL_SQUARES if gt_sq.get(sq) != _START_MAP.get(sq))
    if changed < 8:
        label = 'Opening'
    elif changed < 20:
        label = 'Middlegame'
    else:
        label = 'Endgame'
    return changed, label


def crop_squares(bgr, corners):
    src = np.array([
        corners['top_left'], corners['top_right'],
        corners['bottom_right'], corners['bottom_left'],
    ], dtype=np.float32)
    BOARD_SIZE = 400
    SQ = 50
    dst = np.array([[0,0],[BOARD_SIZE,0],[BOARD_SIZE,BOARD_SIZE],[0,BOARD_SIZE]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(bgr, M, (BOARD_SIZE, BOARD_SIZE))
    patches = {}
    for row_i, rank in enumerate("87654321"):
        for col_i, file in enumerate("abcdefgh"):
            r0, r1 = row_i*SQ, (row_i+1)*SQ
            c0, c1 = col_i*SQ, (col_i+1)*SQ
            patches[f"{file}{rank}"] = warped[r0:r1, c0:c1]
    return patches


# ---------------------------------------------------------------------------
# Per-game evaluation
# ---------------------------------------------------------------------------

def run_on_game(game_id, frames, gt_index, corner_index, clf, images_base, max_moves):
    frames = sorted(frames, key=lambda x: x['move_id'])[:max_moves + 1]
    per_frame = []
    all_preds = []

    for img_meta in frames:
        img_id = img_meta['id']
        img_path = os.path.join(images_base, str(img_meta['game_id']), img_meta['file_name'])
        gt_sq = gt_index.get(img_id, {})
        changed, phase_label = game_phase(gt_sq)

        corners = corner_index.get(img_id)
        if corners is None or not os.path.exists(img_path):
            all_preds.append(None)
            continue

        bgr = cv2.imread(img_path)
        if bgr is None:
            all_preds.append(None)
            continue

        patches = crop_squares(bgr, corners)
        preds = clf.predict_board(patches)
        correct = compare_to_gt(preds, gt_sq)

        per_frame.append({
            'move_id':    img_meta['move_id'],
            'correct':    correct,
            'pct':        100.0 * correct / 64,
            'changed':    changed,
            'phase':      phase_label,
        })
        all_preds.append(preds)

    # Move detection using the feedback pipeline
    valid_preds = [p for p in all_preds if p is not None]
    seq = detect_moves_sequence_with_feedback(
        valid_preds, max_adjustments=6, consensus_window=5, consensus_force_sync=True
    )

    return {
        'game_id':     game_id,
        'per_frame':   per_frame,
        'moves':       seq['moves'],
        'move_tags':   seq['move_tags'],
        'det_errors':  seq['errors'],
        'resyncs':     seq.get('resyncs', []),
        'adjustments': seq.get('adjustments', []),
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def make_charts(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    game_ids = [r['game_id'] for r in results]

    # ── 1. Per-frame accuracy line chart ──────────────────────────────────
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4.5 * n), squeeze=False)
    for ax, gr in zip(axes[:, 0], results):
        frames = gr['per_frame']
        if not frames:
            ax.set_title(f'Game {gr["game_id"]} — no data'); continue
        xs  = [f['move_id'] for f in frames]
        ys  = [f['pct']     for f in frames]
        cols = [PHASE_COLORS[f['phase']] for f in frames]
        # Shade by phase
        for i in range(len(xs) - 1):
            ax.axvspan(xs[i], xs[i+1], alpha=0.06, color=PHASE_COLORS[frames[i]['phase']])
        ax.plot(xs, ys, color=MODEL_COLOR, linewidth=1.5, zorder=3)
        ax.scatter(xs, ys, c=cols, s=28, zorder=4)
        ax.axhline(90, color='gray', linestyle='--', linewidth=1)
        avg = np.mean(ys)
        ax.axhline(avg, color=MODEL_COLOR, linestyle=':', linewidth=1.5,
                   label=f'Avg {avg:.1f}%')
        ax.set_title(f'Game {gr["game_id"]} — Square Accuracy per Frame  (avg {avg:.1f}%)',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Move number'); ax.set_ylabel('Square accuracy (%)')
        ax.set_ylim(0, 107)
        # Phase legend proxies
        from matplotlib.patches import Patch
        legend_els = [Patch(facecolor=c, label=p, alpha=0.7)
                      for p, c in PHASE_COLORS.items()]
        legend_els.append(plt.Line2D([0],[0], color='gray', linestyle='--', label='90% target'))
        ax.legend(handles=legend_els, fontsize=9, loc='lower left')
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    p = os.path.join(output_dir, 'accuracy_per_frame.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")

    # ── 2. Game-phase accuracy bar chart ──────────────────────────────────
    phase_buckets = defaultdict(list)
    for gr in results:
        for f in gr['per_frame']:
            phase_buckets[f['phase']].append(f['pct'])

    phases = ['Opening', 'Middlegame', 'Endgame']
    means  = [np.mean(phase_buckets[p]) if phase_buckets[p] else 0 for p in phases]
    stds   = [np.std(phase_buckets[p])  if phase_buckets[p] else 0 for p in phases]
    counts = [len(phase_buckets[p]) for p in phases]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(phases, means, yerr=stds, capsize=7,
                  color=[PHASE_COLORS[p] for p in phases], width=0.5, alpha=0.88)
    for bar, mean, std, n in zip(bars, means, stds, counts):
        ax.text(bar.get_x() + bar.get_width()/2, mean + std + 1.5,
                f'{mean:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.axhline(90, color='gray', linestyle='--', linewidth=1.5, label='90% target')
    ax.set_ylim(0, 115)
    ax.set_ylabel('Avg square accuracy (%)', fontsize=12)
    ax.set_title('Square Accuracy by Game Phase\n(ChessPieceCNN patch classifier)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    p = os.path.join(output_dir, 'game_phase_accuracy.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")

    # ── 3. Move detection stacked bar ─────────────────────────────────────
    sure_c    = [sum(1 for t in gr['move_tags'] if t == 'sure')                        for gr in results]
    unsure_c  = [sum(1 for t in gr['move_tags'] if 'unsure' in t and 'consensus' not in t) for gr in results]
    cons_c    = [sum(1 for t in gr['move_tags'] if t.startswith('consensus'))           for gr in results]
    fail_c    = [len(gr['det_errors'])                                                  for gr in results]
    potential = [len(gr['per_frame']) - 1                                               for gr in results]

    x = np.arange(len(game_ids))
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x, sure_c,   label='Sure (exact match)',          color='#27ae60')
    b2 = ax.bar(x, unsure_c, bottom=sure_c,                        label='Unsure (feedback corrected)', color='#f39c12')
    b3 = ax.bar(x, cons_c,   bottom=[a+b for a,b in zip(sure_c, unsure_c)], label='Consensus re-sync', color='#e74c3c')
    b4 = ax.bar(x, fail_c,   bottom=[a+b+c for a,b,c in zip(sure_c, unsure_c, cons_c)], label='Failed', color='#bdc3c7')
    # Detection-rate label on top
    for xi, sc, uc, cc, fc, pot in zip(x, sure_c, unsure_c, cons_c, fail_c, potential):
        detected = sc + uc + cc
        ax.text(xi, detected + fc + 0.8, f'{detected}/{pot}\n({100*detected/max(pot,1):.0f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([f'Game {g}' for g in game_ids], fontsize=11)
    ax.set_ylabel('Number of moves', fontsize=12)
    ax.set_title('Move Detection Breakdown per Game\n(ChessPieceCNN + feedback pipeline)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    p = os.path.join(output_dir, 'move_detection.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")

    # ── 4. PGN quality summary ────────────────────────────────────────────
    total_sure    = sum(sure_c)
    total_unsure  = sum(unsure_c)
    total_cons    = sum(cons_c)
    total_fail    = sum(fail_c)
    total_det     = total_sure + total_unsure + total_cons
    total_pot     = sum(potential)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Pie — move quality
    labels_pie  = ['Sure\n(exact)', 'Unsure\n(feedback)', 'Consensus\n(re-sync)', 'Failed']
    sizes_pie   = [total_sure, total_unsure, total_cons, total_fail]
    colors_pie  = ['#27ae60', '#f39c12', '#e74c3c', '#bdc3c7']
    nonzero     = [(l, s, c) for l, s, c in zip(labels_pie, sizes_pie, colors_pie) if s > 0]
    wedges, texts, autos = axes[0].pie(
        [s for _,s,_ in nonzero],
        labels=[l for l,_,_ in nonzero],
        colors=[c for _,_,c in nonzero],
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 10},
    )
    axes[0].set_title(f'Move Detection Quality\n{total_det} detected  /  {total_pot} potential',
                      fontsize=11, fontweight='bold')

    # Bar — per-game detection rate
    rates = [(sc+uc+cc) / max(pot,1) * 100
             for sc,uc,cc,pot in zip(sure_c, unsure_c, cons_c, potential)]
    bars = axes[1].bar([f'Game {g}' for g in game_ids], rates, color=MODEL_COLOR, alpha=0.82)
    for bar, rate in zip(bars, rates):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{rate:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[1].axhline(100, color='gray', linestyle='--', linewidth=1)
    axes[1].set_ylim(0, 120)
    axes[1].set_ylabel('Move detection rate (%)', fontsize=12)
    axes[1].set_title('Move Detection Rate per Game', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.suptitle('PGN Accuracy — ChessPieceCNN Pipeline', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    p = os.path.join(output_dir, 'pgn_quality.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")

    # ── 5. Accuracy vs game phase scatter ─────────────────────────────────
    all_changed = [f['changed'] for gr in results for f in gr['per_frame']]
    all_pct     = [f['pct']     for gr in results for f in gr['per_frame']]
    all_phase   = [f['phase']   for gr in results for f in gr['per_frame']]

    fig, ax = plt.subplots(figsize=(11, 6))
    for phase, color in PHASE_COLORS.items():
        xs = [c for c, ph in zip(all_changed, all_phase) if ph == phase]
        ys = [p for p, ph in zip(all_pct,     all_phase) if ph == phase]
        ax.scatter(xs, ys, alpha=0.2, s=18, c=color, label=phase)
    # Overall trend line
    if len(all_changed) > 5:
        z = np.polyfit(all_changed, all_pct, 1)
        xs_trend = np.linspace(min(all_changed), max(all_changed), 200)
        ax.plot(xs_trend, np.poly1d(z)(xs_trend), 'k-', linewidth=2.5, label='Trend', zorder=5)
    ax.axhline(90, color='gray', linestyle='--', linewidth=1.5, label='90% target')
    ax.axvline(8,  color=PHASE_COLORS['Opening'],     linestyle=':', linewidth=1.5, alpha=0.8)
    ax.axvline(20, color=PHASE_COLORS['Middlegame'],  linestyle=':', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Squares changed from starting position  (0 = opening, 32+ = endgame)', fontsize=11)
    ax.set_ylabel('Square accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs Game Phase\n(ChessPieceCNN patch classifier)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    p = os.path.join(output_dir, 'accuracy_vs_phase.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")

    # ── Text summary ──────────────────────────────────────────────────────
    lines = ["Chess-to-PGN — ChessPieceCNN Evaluation Summary",
             "=" * 58, f"Date: {datetime.date.today()}",
             f"Games: {game_ids}", ""]
    for gr in results:
        pcts = [f['pct'] for f in gr['per_frame']]
        if not pcts:
            lines.append(f"Game {gr['game_id']}: no data"); continue
        avg   = np.mean(pcts)
        above = sum(1 for p in pcts if p >= 90)
        tags  = gr['move_tags']
        sc = sum(1 for t in tags if t == 'sure')
        uc = sum(1 for t in tags if 'unsure' in t and 'consensus' not in t)
        cc = sum(1 for t in tags if t.startswith('consensus'))
        fc = len(gr['det_errors'])
        pot = len(gr['per_frame']) - 1
        det = sc + uc + cc
        lines += [
            f"Game {gr['game_id']}:",
            f"  Frames:          {len(pcts)}",
            f"  Avg accuracy:    {avg:.1f}%",
            f"  Frames >=90%:    {above}/{len(pcts)} ({100*above/len(pcts):.1f}%)",
            f"  Min/Max:         {min(pcts):.1f}% / {max(pcts):.1f}%",
            f"  Moves detected:  {det}/{pot} ({100*det/max(pot,1):.0f}%)",
            f"    Sure:          {sc}",
            f"    Unsure:        {uc}",
            f"    Consensus:     {cc}",
            f"    Failed:        {fc}",
            f"  Re-syncs:        {len(gr['resyncs'])}",
        ]
        pgn = generate_pgn(gr['moves'], {
            'event': f'ChessReD Game {gr["game_id"]}', 'site': 'chess-to-pgn',
            'date': datetime.date.today().isoformat(), 'round': '-',
            'white': 'White', 'black': 'Black',
        }, result='*', move_tags=gr['move_tags'])
        lines += [f"  PGN preview:", f"    {pgn[:300]}", ""]

    all_pcts = [f['pct'] for gr in results for f in gr['per_frame']]
    lines += [
        "OVERALL",
        "-" * 40,
        f"  Total frames:       {len(all_pcts)}",
        f"  Avg accuracy:       {np.mean(all_pcts):.1f}%",
        f"  Frames >=90%:       {sum(1 for p in all_pcts if p>=90)}/{len(all_pcts)} ({100*sum(1 for p in all_pcts if p>=90)/len(all_pcts):.1f}%)",
        f"  Total moves det:    {total_det}/{total_pot} ({100*total_det/max(total_pot,1):.1f}%)",
        f"  Sure:               {total_sure}",
        f"  Feedback-corrected: {total_unsure}",
        f"  Consensus:          {total_cons}",
        f"  Failed:             {total_fail}",
    ]
    # Phase breakdown
    for phase in phases:
        ph_pcts = [f['pct'] for gr in results for f in gr['per_frame'] if f['phase'] == phase]
        if ph_pcts:
            lines.append(f"  {phase} avg:  {np.mean(ph_pcts):.1f}%  (n={len(ph_pcts)})")

    summary_path = os.path.join(output_dir, 'summary_stats.txt')
    with open(summary_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print(f"  Saved: {summary_path}")
    print('\n'.join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, nargs='+', default=[0, 33, 76])
    parser.add_argument('--split',            default='test')
    parser.add_argument('--max_moves',        type=int, default=100)
    parser.add_argument('--patch_checkpoint', default='models/chess_piece_classifier.pth')
    parser.add_argument('--output_dir',       default='stats_output')
    args = parser.parse_args()

    print("Loading annotations ...")
    with open('data/annotations.json') as f:
        annotations = json.load(f)

    gt_index     = build_gt_index(annotations)
    corner_index = {c['image_id']: c['corners'] for c in annotations['annotations']['corners']}
    id_to_img    = {img['id']: img for img in annotations['images']}
    split_ids    = set(annotations['splits'][args.split]['image_ids'])

    game_frames = defaultdict(list)
    for img_id in split_ids:
        img = id_to_img[img_id]
        game_frames[img['game_id']].append(img)

    print(f"Loading patch classifier from {args.patch_checkpoint} ...")
    clf = ChessPieceClassifier(model_path=args.patch_checkpoint)
    print(f"  Device: {clf.device}")

    all_results = []
    for game_id in args.games:
        if game_id not in game_frames:
            print(f"Game {game_id} not in '{args.split}' split — skipping."); continue
        frames = game_frames[game_id]
        print(f"\nProcessing game {game_id}  ({len(frames)} frames) ...")
        gr = run_on_game(game_id, frames, gt_index, corner_index, clf,
                         'data/ChessRed_images', args.max_moves)
        all_results.append(gr)
        pcts = [f['pct'] for f in gr['per_frame']]
        if pcts:
            print(f"  Avg accuracy: {np.mean(pcts):.1f}%  |  "
                  f">=90%: {sum(1 for p in pcts if p>=90)}/{len(pcts)}")
        print(f"  Moves detected: {len(gr['moves'])}  "
              f"(sure={sum(1 for t in gr['move_tags'] if t=='sure')}, "
              f"unsure={sum(1 for t in gr['move_tags'] if 'unsure' in t and 'consensus' not in t)}, "
              f"consensus={sum(1 for t in gr['move_tags'] if t.startswith('consensus'))})")

    if not all_results:
        print("No results — check game IDs and corner availability."); return

    print(f"\nGenerating charts -> {args.output_dir}/")
    make_charts(all_results, args.output_dir)


if __name__ == '__main__':
    main()
