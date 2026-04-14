"""
Detect chess moves by diffing consecutive FEN positions and validating
against legal moves using python-chess.
"""

import chess
from typing import Optional

from src.pipeline.fen_generator import predictions_to_fen, fen_position_only


def fen_to_piece_map(fen_position: str) -> dict:
    """Convert FEN position string to {square_name: piece_char} dict.

    Empty squares are omitted.
    """
    pieces = {}
    files = "abcdefgh"
    ranks = fen_position.split('/')

    for rank_idx, rank_str in enumerate(ranks):
        rank_num = 8 - rank_idx  # FEN starts from rank 8
        col = 0
        for ch in rank_str:
            if ch.isdigit():
                col += int(ch)
            else:
                square = f"{files[col]}{rank_num}"
                pieces[square] = ch
                col += 1
    return pieces


def diff_positions(prev_fen_pos: str, curr_fen_pos: str) -> dict:
    """Find squares that changed between two FEN positions.

    Returns dict with:
      'cleared': list of (square, old_piece) — squares that lost a piece
      'placed': list of (square, new_piece) — squares that gained a piece
      'changed': list of (square, old_piece, new_piece) — squares where piece changed
    """
    prev = fen_to_piece_map(prev_fen_pos)
    curr = fen_to_piece_map(curr_fen_pos)

    all_squares = set(list(prev.keys()) + list(curr.keys()))
    cleared = []
    placed = []
    changed = []

    for sq in all_squares:
        old = prev.get(sq)
        new = curr.get(sq)
        if old == new:
            continue
        if old and not new:
            cleared.append((sq, old))
        elif not old and new:
            placed.append((sq, new))
        elif old and new and old != new:
            changed.append((sq, old, new))

    return {'cleared': cleared, 'placed': placed, 'changed': changed}


def detect_move(prev_fen_pos: str, curr_fen_pos: str, board: chess.Board,
                fuzzy_threshold: int = 62) -> Optional[str]:
    """Find the legal move that transforms the board from prev to curr position.

    Parameters
    ----------
    prev_fen_pos : str — FEN position part only (e.g. 'rnbqkbnr/pppppppp/...')
    curr_fen_pos : str — FEN position part only
    board        : chess.Board — current game state (tracks castling, en passant, turns)

    Returns
    -------
    str or None — SAN notation of the detected move (e.g. 'e4', 'Nf3', 'O-O'),
                  or None if no legal move matches.
    """
    if prev_fen_pos == curr_fen_pos:
        return None  # No change detected

    # Try each legal move: apply it and check if the resulting position matches
    best_move = None
    best_score = -1

    for move in board.legal_moves:
        board.push(move)
        resulting_pos = board.fen().split(' ')[0]
        board.pop()

        if resulting_pos == curr_fen_pos:
            return board.san(move)

        # Fuzzy matching: count how many squares match
        score = _position_similarity(resulting_pos, curr_fen_pos)
        if score > best_score:
            best_score = score
            best_move = move

    # If no exact match, use the best fuzzy match if it's close enough
    if best_move and best_score >= fuzzy_threshold:
        return board.san(best_move)

    return None


def _position_similarity(fen_pos_a: str, fen_pos_b: str) -> int:
    """Count number of matching squares between two FEN positions (out of 64)."""
    map_a = fen_to_piece_map(fen_pos_a)
    map_b = fen_to_piece_map(fen_pos_b)

    matching = 0
    files = "abcdefgh"
    for rank in range(1, 9):
        for f in files:
            sq = f"{f}{rank}"
            piece_a = map_a.get(sq, 'empty')
            piece_b = map_b.get(sq, 'empty')
            if piece_a == piece_b:
                matching += 1

    return matching


def detect_move_with_feedback(
    prev_fen_pos: str,
    predictions: dict,
    board: chess.Board,
    max_adjustments: int = 3,
) -> tuple:
    """Try to detect a legal move, with feedback-driven correction on failure.

    Instead of giving up when the predicted FEN doesn't match any legal move,
    this function asks: "which legal move best explains the predictions, even
    if the classifier got a few squares wrong?"  It finds the legal move whose
    resulting position differs from the predicted FEN in the fewest squares,
    and only accepts it if those differing squares all had low confidence.

    Parameters
    ----------
    prev_fen_pos  : str  — FEN position before the move
    predictions   : dict {square: (piece_class, confidence)}
    board         : chess.Board — current game state
    max_adjustments : int — max number of low-confidence squares allowed to
                      differ between the predicted FEN and the matched legal
                      position (default 3)

    Returns
    -------
    (san_move, tag, adjustments)
      san_move    : str or None — SAN move if found, else None
      tag         : 'sure' | 'unsure' | 'failed'
                    'sure'   — exact FEN match, no corrections needed
                    'unsure' — match found after correcting 1–max_adjustments
                               low-confidence squares
                    'failed' — no legal move found within adjustment budget
      adjustments : list of (square, original_pred, corrected_pred, confidence)
                    squares the feedback loop had to flip to find a legal move
    """
    curr_fen_pos = fen_position_only(predictions_to_fen(predictions))

    # --- Pass 1: try exact / fuzzy match (existing logic) ---
    san = detect_move(prev_fen_pos, curr_fen_pos, board)
    if san:
        return san, 'sure', []

    # --- Pass 2: feedback loop over all legal moves ---
    # For each legal move, compute the resulting position and count how many
    # squares differ from our prediction — and what the confidence was there.
    best = None   # (n_mismatches, max_mismatch_conf, move_obj, mismatches)

    for move in board.legal_moves:
        board.push(move)
        legal_pos = board.fen().split(' ')[0]
        board.pop()

        # Find squares where our prediction disagrees with this legal outcome
        legal_map = fen_to_piece_map(legal_pos)
        mismatches = []
        for sq in _ALL_SQUARES:
            pred_piece = predictions.get(sq, ('empty', 0.0))[0]
            legal_piece = legal_map.get(sq, 'empty')
            if pred_piece != legal_piece:
                conf = predictions.get(sq, ('empty', 0.0))[1]
                mismatches.append((sq, pred_piece, legal_piece, conf))

        n = len(mismatches)
        if n == 0:
            # Exact match — shouldn't happen since detect_move already tried,
            # but handle gracefully
            return board.san(move), 'sure', []

        if n > max_adjustments:
            continue  # Too many corrections needed, skip

        # Among candidates with ≤ max_adjustments mismatches, prefer the one
        # where the mismatch squares had the lowest maximum confidence
        # (i.e. the model was least sure about those squares — most plausible flip)
        max_conf = max(c for _, _, _, c in mismatches)
        if best is None or n < best[0] or (n == best[0] and max_conf < best[1]):
            best = (n, max_conf, move, mismatches)

    if best is not None:
        _, _, move_obj, mismatches = best
        san = board.san(move_obj)
        adjustments = [(sq, pred, corr, conf) for sq, pred, corr, conf in mismatches]
        return san, 'unsure', adjustments

    return None, 'failed', []


# All 64 squares in FEN order — needed by detect_move_with_feedback
_ALL_SQUARES = [f"{f}{r}" for r in "87654321" for f in "abcdefgh"]


def detect_moves_sequence(fen_positions: list, fuzzy_threshold: int = 62) -> dict:
    """Process a sequence of FEN positions and detect all moves.

    Parameters
    ----------
    fen_positions : list of str — FEN position strings in chronological order

    Returns
    -------
    dict with:
      'moves': list of str — SAN notation moves
      'board': chess.Board — final board state
      'errors': list of dict — {index, prev_fen, curr_fen, reason}
      'skipped': int — number of consecutive identical positions skipped
    """
    if len(fen_positions) < 2:
        return {'moves': [], 'board': chess.Board(), 'errors': [], 'skipped': 0}

    board = chess.Board()
    moves = []
    errors = []
    skipped = 0

    prev_pos = fen_positions[0]

    for i in range(1, len(fen_positions)):
        curr_pos = fen_positions[i]

        # Skip identical positions (no move detected)
        if curr_pos == prev_pos:
            skipped += 1
            continue

        san = detect_move(prev_pos, curr_pos, board, fuzzy_threshold=fuzzy_threshold)

        if san:
            move = board.parse_san(san)
            board.push(move)
            moves.append(san)
            prev_pos = board.fen().split(' ')[0]  # Use actual board state, not classifier output
        else:
            errors.append({
                'index': i,
                'prev_fen': prev_pos,
                'curr_fen': curr_pos,
                'reason': 'no_legal_move_match',
            })
            # Still update prev_pos to curr_pos to try to recover
            prev_pos = curr_pos

    return {
        'moves': moves,
        'board': board,
        'errors': errors,
        'skipped': skipped,
    }


def compute_consensus_predictions(frames: list) -> dict:
    """Majority-vote across K frames' predictions for each of 64 squares.

    For each square, the winning piece class is the one seen most often across
    the given frames. Ties are broken by highest total confidence for that class.

    Parameters
    ----------
    frames : list of dict {square: (piece_class, confidence)}

    Returns
    -------
    dict {square: (winning_piece, vote_fraction)}
      vote_fraction is votes_for_winner / len(frames) — acts as confidence.
    """
    if not frames:
        return {}

    n = len(frames)
    consensus = {}
    for sq in _ALL_SQUARES:
        vote_counts: dict = {}
        vote_conf: dict = {}
        for preds in frames:
            piece, conf = preds.get(sq, ('empty', 0.0))
            vote_counts[piece] = vote_counts.get(piece, 0) + 1
            vote_conf[piece] = vote_conf.get(piece, 0.0) + conf

        # Pick winner: most votes, tie-break by highest total confidence
        winner = max(vote_counts, key=lambda p: (vote_counts[p], vote_conf[p]))
        consensus[sq] = (winner, vote_counts[winner] / n)

    return consensus


def detect_moves_sequence_with_feedback(
    frames_predictions: list,
    max_adjustments: int = 3,
    fuzzy_threshold: int = 62,
    consensus_window: int = 5,
    consensus_force_sync: bool = True,
) -> dict:
    """Process a sequence of per-frame predictions and detect all moves,
    with feedback-driven correction, 'unsure' flagging, and consensus re-sync.

    Three layers of resilience:
      1. Exact / fuzzy FEN match  → tag 'sure'
      2. Feedback correction (≤max_adjustments low-conf squares flipped) → 'unsure'
      3. Consensus re-sync: after consensus_window consecutive failures, majority-
         vote across failing frames to estimate the true board position, then:
           a. Try feedback detection against the consensus → 'consensus_sure/unsure'
           b. If still no match and consensus_force_sync=True, hard-reset the board
              to the consensus position (no move emitted, logged as resync event)

    Parameters
    ----------
    frames_predictions  : list of dict {square: (piece_class, confidence)}
    max_adjustments     : int — max squares feedback loop may correct per move
    fuzzy_threshold     : int — passed to underlying detect_move
    consensus_window    : int — consecutive failures before triggering consensus
    consensus_force_sync: bool — whether to force-sync board on consensus mismatch

    Returns
    -------
    dict with:
      'moves'        : list of str — SAN moves (all detected, including unsure)
      'move_tags'    : list of str — 'sure'|'unsure'|'consensus_sure'|
                       'consensus_unsure' per move
      'board'        : chess.Board — final board state
      'errors'       : list of dict — failed frame info
      'adjustments'  : list of list — per-move correction details
      'resyncs'      : list of dict — consensus force-sync events
      'skipped'      : int — frames with no position change
    """
    if len(frames_predictions) < 2:
        return {
            'moves': [], 'move_tags': [], 'board': chess.Board(),
            'errors': [], 'adjustments': [], 'resyncs': [], 'skipped': 0,
        }

    board = chess.Board()
    moves = []
    move_tags = []
    all_adjustments = []
    errors = []
    resyncs = []
    skipped = 0

    prev_pos = fen_position_only(predictions_to_fen(frames_predictions[0]))
    failure_buffer = []   # accumulates failed-frame predictions for consensus

    for i in range(1, len(frames_predictions)):
        preds = frames_predictions[i]
        curr_pos = fen_position_only(predictions_to_fen(preds))

        if curr_pos == prev_pos:
            skipped += 1
            continue

        # --- Normal feedback detection ---
        san, tag, adjustments = detect_move_with_feedback(
            prev_pos, preds, board, max_adjustments=max_adjustments
        )

        if san:
            move = board.parse_san(san)
            board.push(move)
            moves.append(san)
            move_tags.append(tag)
            all_adjustments.append(adjustments)
            prev_pos = board.fen().split(' ')[0]
            failure_buffer.clear()
            continue

        # --- Detection failed — buffer this frame ---
        failure_buffer.append(preds)
        errors.append({
            'index': i,
            'reason': 'no_legal_move_within_adjustment_budget',
            'adjustments_tried': adjustments,
        })

        # --- Consensus re-sync trigger ---
        if len(failure_buffer) < consensus_window:
            continue

        consensus_preds = compute_consensus_predictions(failure_buffer)
        consensus_pos = fen_position_only(predictions_to_fen(consensus_preds))
        diff_from_board = 64 - _position_similarity(prev_pos, consensus_pos)

        # Try feedback detection against the consensus position
        c_san, c_tag, c_adj = detect_move_with_feedback(
            prev_pos, consensus_preds, board, max_adjustments=max_adjustments
        )

        if c_san:
            move = board.parse_san(c_san)
            board.push(move)
            moves.append(c_san)
            move_tags.append(f'consensus_{c_tag}')
            all_adjustments.append(c_adj)
            prev_pos = board.fen().split(' ')[0]
            failure_buffer.clear()
        elif consensus_force_sync:
            # Hard re-sync: trust the majority-vote position over the stale board
            turn_char = 'w' if board.turn == chess.WHITE else 'b'
            sync_fen = f"{consensus_pos} {turn_char} - - 0 {board.fullmove_number}"
            try:
                board.set_fen(sync_fen)
                prev_pos = consensus_pos
                resyncs.append({
                    'frame': i,
                    'consensus_pos': consensus_pos,
                    'diff_from_prev': diff_from_board,
                    'window_size': len(failure_buffer),
                })
            except ValueError:
                # Consensus FEN is illegal (shouldn't happen but guard anyway)
                pass
            failure_buffer.clear()

    return {
        'moves': moves,
        'move_tags': move_tags,
        'board': board,
        'errors': errors,
        'adjustments': all_adjustments,
        'resyncs': resyncs,
        'skipped': skipped,
    }
