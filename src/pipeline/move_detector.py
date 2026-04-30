"""
Detect chess moves by diffing consecutive FEN positions and validating
against legal moves using python-chess.

Production decoder : detect_moves_sequence_with_prior (Bayesian, 87.9% on 40-game eval)
Naive baseline     : detect_moves_sequence_with_feedback (60.7%)
Experimental only  : TemporalBoardTracker (30.2% -- do NOT use in production)
"""

import math
import chess
import numpy as np
from typing import Optional

from src.pipeline.fen_generator import predictions_to_fen, fen_position_only

# Canonical FEN class ordering matching ChessPieceClassifier.fen_class_order
FEN_CLASSES = ['empty', 'P', 'N', 'B', 'R', 'Q', 'K',
               'p', 'n', 'b', 'r', 'q', 'k']
_FEN_CLASS_TO_IDX = {c: i for i, c in enumerate(FEN_CLASSES)}


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


# ---------------------------------------------------------------------------
# Bayesian move-history prior
# ---------------------------------------------------------------------------
def detect_move_with_prior(
    prev_fen_pos: str,
    full_probs: dict,
    board: chess.Board,
    prior_weight: float = 1.0,
) -> tuple:
    """Score every legal move by its Bayesian posterior under the full
    classifier softmax, and return the argmax.

    Given the classifier output p(piece | square, image) — a 13-way softmax
    per square — and a uniform prior over legal moves, the log-posterior
    of a candidate move m is:

        log P(m | image, history) = sum_{sq} log p(piece_m(sq) | square, image)

    where piece_m(sq) is the piece that would occupy `sq` after applying m.
    This is exactly Naive-Bayes with the 64 squares as conditionally-
    independent features under each move hypothesis.

    The `prior_weight` term scales the log-likelihood of squares that
    *change* relative to squares that stay the same — a higher weight
    emphasises the few cells that actually carry move information.

    Advantage over the existing feedback loop: we never truncate at "top-1
    per square", so a move is chosen even when the raw per-square argmax
    doesn't lead to any legal position. This directly addresses the 77 / 100
    failure rate seen on ChessReD games where 1–3 squares are mis-classified.

    Parameters
    ----------
    prev_fen_pos : str    — FEN position before the move
    full_probs   : dict   — {square: np.ndarray(13)} softmax over FEN_CLASSES
    board        : chess.Board — game state at prev_fen_pos
    prior_weight : float  — weight on changed-square likelihood (default 1.0)

    Returns
    -------
    (san_move, tag, score)
      san_move : SAN string of the best move, or None if no legal moves
      tag      : 'sure' if the argmax-per-square FEN is already legal,
                 'prior' if the Bayesian prior was decisive,
                 'failed' if the board has no legal moves
      score    : log-posterior of the chosen move (higher = more confident)
    """
    if not list(board.legal_moves):
        return None, 'failed', -math.inf

    # Fast path: does the raw argmax-per-square FEN already match a legal move?
    argmax_preds = {}
    for sq, probs in full_probs.items():
        idx = int(np.argmax(probs))
        argmax_preds[sq] = (FEN_CLASSES[idx], float(probs[idx]))
    curr_pos = fen_position_only(predictions_to_fen(argmax_preds))
    san_fast = detect_move(prev_fen_pos, curr_pos, board)
    if san_fast:
        return san_fast, 'sure', 0.0

    # Score every legal move under the full softmax
    prev_map = fen_to_piece_map(prev_fen_pos)
    best_move, best_score = None, -math.inf

    # Pre-compute log-probs once (add epsilon for numerical stability)
    log_probs = {sq: np.log(np.maximum(full_probs[sq], 1e-8)) for sq in _ALL_SQUARES}

    for move in board.legal_moves:
        board.push(move)
        resulting_pos = board.fen().split(' ')[0]
        board.pop()
        result_map = fen_to_piece_map(resulting_pos)

        score = 0.0
        for sq in _ALL_SQUARES:
            resulting_piece = result_map.get(sq, 'empty')
            prev_piece = prev_map.get(sq, 'empty')
            class_idx = _FEN_CLASS_TO_IDX[resulting_piece]
            sq_log_p = log_probs[sq][class_idx]
            if resulting_piece != prev_piece:
                score += prior_weight * sq_log_p
            else:
                score += sq_log_p

        if score > best_score:
            best_score = score
            best_move = move

    if best_move is None:
        return None, 'failed', -math.inf
    return board.san(best_move), 'prior', best_score


def detect_moves_sequence_with_prior(
    frames_full_probs: list,
    prior_weight: float = 1.0,
) -> dict:
    """Run the Bayesian-prior detector over a sequence of per-frame softmaxes.

    Much simpler than `detect_moves_sequence_with_feedback` because the
    Bayesian prior already handles classifier noise gracefully — we never
    need a consensus re-sync when every move is a direct argmax over the
    legal-move space.

    Parameters
    ----------
    frames_full_probs : list of dict {square: np.ndarray(13)}
        Per-frame softmax outputs from predict_board_full_probs().

    Returns
    -------
    dict with 'moves', 'move_tags', 'board', 'errors', 'skipped'
    """
    if len(frames_full_probs) < 2:
        return {'moves': [], 'move_tags': [], 'board': chess.Board(),
                'errors': [], 'skipped': 0}

    board = chess.Board()
    moves, move_tags, errors = [], [], []
    skipped = 0

    prev_pos = fen_position_only(predictions_to_fen({
        sq: (FEN_CLASSES[int(np.argmax(p))], float(p[int(np.argmax(p))]))
        for sq, p in frames_full_probs[0].items()
    }))

    for i in range(1, len(frames_full_probs)):
        probs = frames_full_probs[i]
        # Skip if argmax FEN equals prev FEN (no move detected)
        argmax_fen = fen_position_only(predictions_to_fen({
            sq: (FEN_CLASSES[int(np.argmax(p))], float(p[int(np.argmax(p))]))
            for sq, p in probs.items()
        }))
        if argmax_fen == prev_pos:
            skipped += 1
            continue

        san, tag, score = detect_move_with_prior(prev_pos, probs, board,
                                                 prior_weight=prior_weight)
        if san is None:
            errors.append({'index': i, 'reason': 'no_legal_moves'})
            continue

        move = board.parse_san(san)
        board.push(move)
        moves.append(san)
        move_tags.append(tag)
        prev_pos = board.fen().split(' ')[0]

    return {'moves': moves, 'move_tags': move_tags, 'board': board,
            'errors': errors, 'skipped': skipped}


def _argmax_fen_from_probs(full_probs: dict) -> str:
    """Return the position-only FEN for a frame's 64-square softmax output."""
    return fen_position_only(predictions_to_fen({
        sq: (FEN_CLASSES[int(np.argmax(p))], float(p[int(np.argmax(p))]))
        for sq, p in full_probs.items()
    }))


def project_legal_state_sequence(
    frames_full_probs: list,
    prior_weight: float = 1.0,
) -> dict:
    """Project frame softmaxes onto a legal board-state sequence.

    This is the production-friendly form of the legal-state projection that
    scored much higher exact-board accuracy in autoresearch. It preserves one
    FEN per frame, but replaces noisy per-frame argmax boards with the board
    state reached by the best legal Bayesian move whenever a change is detected.

    Returns a dict compatible with `detect_moves_sequence_with_prior`, plus:
      - fen_sequence: projected legal FEN position for each frame
      - raw_fen_sequence: raw argmax FEN position for each frame
      - move_frame_indices: frame indices where a move was accepted
    """
    if not frames_full_probs:
        return {
            'moves': [], 'move_tags': [], 'board': chess.Board(),
            'errors': [], 'skipped': 0,
            'fen_sequence': [], 'raw_fen_sequence': [],
            'move_frame_indices': [],
        }

    board = chess.Board()
    moves, move_tags, errors = [], [], []
    skipped = 0
    move_frame_indices = []

    raw_fens = [_argmax_fen_from_probs(probs) for probs in frames_full_probs]
    prev_pos = raw_fens[0]
    projected_fens = [prev_pos]

    for i, (probs, argmax_fen) in enumerate(zip(frames_full_probs[1:], raw_fens[1:]), start=1):
        if argmax_fen == prev_pos:
            skipped += 1
            projected_fens.append(prev_pos)
            continue

        san, tag, _ = detect_move_with_prior(prev_pos, probs, board,
                                             prior_weight=prior_weight)
        if san is None:
            errors.append({'index': i, 'reason': 'no_legal_moves'})
            projected_fens.append(prev_pos)
            continue

        try:
            move = board.parse_san(san)
        except ValueError:
            errors.append({'index': i, 'reason': 'invalid_san', 'san': san})
            projected_fens.append(prev_pos)
            continue

        board.push(move)
        moves.append(san)
        move_tags.append(tag)
        move_frame_indices.append(i)
        prev_pos = board.fen().split(' ')[0]
        projected_fens.append(prev_pos)

    return {
        'moves': moves,
        'move_tags': move_tags,
        'board': board,
        'errors': errors,
        'skipped': skipped,
        'fen_sequence': projected_fens,
        'raw_fen_sequence': raw_fens,
        'move_frame_indices': move_frame_indices,
    }


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


# ---------------------------------------------------------------------------
# Temporal board tracker
# ── Experimental / Evaluation Only ─────────────────────────────────────────
# TemporalBoardTracker scores 30.2% on the 40-game eval (vs 87.9% Bayesian).
# Root cause: exact argmax-FEN confirmation is too strict (~50% of frames have
# >=2 wrong squares), so the confirmed-state condition almost never triggers.
# Keep for eval scripts (scripts/eval_temporal_tracker.py) — do NOT import in
# production code (server.py, process_game.py).
# ---------------------------------------------------------------------------

class TemporalBoardTracker:
    """Maintains running board state and applies temporal heuristics to resolve
    uncertain classifier predictions frame by frame.

    Three complementary strategies are combined:

    1. **Change-mask gating** — squares that agree between the confirmed board
       FEN and the current classifier argmax are treated as stable. Their softmax
       is overridden with a near-certain prior (STABLE_BOOST) so the Bayesian
       detector ignores them and focuses only on the 2-4 diff squares.

    2. **Piece-inventory constraint** — after each confirmed move the tracker
       knows exactly how many of each piece type remain. If the classifier
       over-predicts a piece type (e.g. reports 3 white rooks when only 2 exist),
       the lowest-confidence excess predictions are squashed toward 'empty'.

    3. **Bayesian move detection** — the adjusted softmax is passed to
       `detect_move_with_prior`, which scores every legal move by the product of
       per-square likelihoods and picks the highest-scoring one.

    Usage::

        tracker = TemporalBoardTracker()
        for frame_probs in sequence_of_softmax_dicts:
            san, tag = tracker.push(frame_probs)
            if san:
                print(san, tag)
        print(tracker.board.fen())
    """

    # Confidence used to represent "we're certain this square is stable"
    STABLE_BOOST = 0.97
    CONFIRMATION_TOLERANCE = 2

    def __init__(self, prior_weight: float = 1.5):
        """
        Parameters
        ----------
        prior_weight : float
            Multiplier on the log-likelihood of changed squares vs stable ones
            inside `detect_move_with_prior`. Higher → more aggressive about
            trusting the diff mask over the background. Default 2.0 works well
            when change-mask gating is active.
        """
        self.board = chess.Board()
        self.prior_weight = prior_weight
        self._confirmed_pos = self.board.fen().split(' ')[0]
        self._seen_frame = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, full_probs: dict) -> tuple:
        """Process one frame's softmax output and attempt to detect a move.

        Parameters
        ----------
        full_probs : dict {square: np.ndarray(shape=(13,))}
            Raw softmax from `ChessPieceClassifier.predict_board_full_probs()`.

        Returns
        -------
        (san_move, tag) where:
          san_move : str | None  — SAN of the detected move, or None if no change
          tag      : str         — 'sure' | 'prior' | 'no_change' | 'failed'
        """
        # Fast path: argmax matches confirmed board → no move
        argmax_pos = self._argmax_pos(full_probs)
        if not self._seen_frame:
            self._seen_frame = True
            if argmax_pos == self._confirmed_pos:
                return None, 'no_change'
            if detect_move(self._confirmed_pos, argmax_pos, self.board) is None:
                self._confirmed_pos = argmax_pos
                return None, 'no_change'

        if argmax_pos == self._confirmed_pos:
            return None, 'no_change'

        adjusted = full_probs
        san, tag, _ = detect_move_with_prior(
            self._confirmed_pos, adjusted, self.board,
            prior_weight=self.prior_weight,
        )

        if san:
            try:
                move = self.board.parse_san(san)
            except ValueError:
                return None, 'failed'

            self.board.push(move)
            candidate_pos = self.board.fen().split(' ')[0]
            self.board.pop()

            self.board.push(move)
            self._confirmed_pos = self.board.fen().split(' ')[0]

        return san, tag

    def reset(self):
        """Reset to starting position."""
        self.board = chess.Board()
        self._confirmed_pos = self.board.fen().split(' ')[0]
        self._seen_frame = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _argmax_pos(self, full_probs: dict) -> str:
        preds = {
            sq: (FEN_CLASSES[int(np.argmax(p))], float(p[int(np.argmax(p))]))
            for sq, p in full_probs.items()
        }
        return fen_position_only(predictions_to_fen(preds))

    def _apply_temporal_heuristics(self, full_probs: dict) -> dict:
        """Return a modified softmax dict with change-mask gating and
        inventory constraints applied."""
        confirmed_map = fen_to_piece_map(self._confirmed_pos)
        argmax_map = fen_to_piece_map(self._argmax_pos(full_probs))

        # --- 1. Identify stable vs changed squares ---
        changed = set()
        for sq in _ALL_SQUARES:
            if confirmed_map.get(sq, 'empty') != argmax_map.get(sq, 'empty'):
                changed.add(sq)

        # --- 2. Build adjusted softmax ---
        adjusted = {}
        for sq in _ALL_SQUARES:
            if sq not in changed:
                # Stable square: inject a strong prior matching confirmed board
                vec = np.full(13, (1.0 - self.STABLE_BOOST) / 12, dtype=np.float32)
                confirmed_piece = confirmed_map.get(sq, 'empty')
                vec[_FEN_CLASS_TO_IDX[confirmed_piece]] = self.STABLE_BOOST
                adjusted[sq] = vec
            else:
                adjusted[sq] = full_probs[sq].copy()

        # --- 3. Inventory constraint on changed squares only ---
        # Count how many of each piece the classifier currently predicts
        # across changed squares, versus what the confirmed board holds.
        # If a piece is over-predicted, squash excess to 'empty'.
        self._squash_excess_pieces(adjusted, confirmed_map, changed)

        return adjusted

    def _squash_excess_pieces(self, adjusted: dict, confirmed_map: dict,
                              changed: set):
        """Squash excess piece predictions on changed squares to enforce inventory.

        For each piece type, count how many are on the confirmed board. If the
        adjusted argmax on changed squares would push the total above that count
        (e.g. predicts a third rook when only two exist), the lowest-confidence
        rook prediction among the changed squares is replaced with 'empty'.
        """
        # Inventory: how many of each piece are confirmed
        inventory: dict = {}
        for piece in confirmed_map.values():
            inventory[piece] = inventory.get(piece, 0) + 1

        # Current argmax on ALL squares given adjusted probs
        # Stable squares already match confirmed, so only changed need checking
        for piece_type in list(FEN_CLASSES):
            if piece_type == 'empty':
                continue
            confirmed_count = inventory.get(piece_type, 0)

            # How many confirmed (stable) squares already have this piece?
            stable_count = sum(
                1 for sq in _ALL_SQUARES
                if sq not in changed and confirmed_map.get(sq, 'empty') == piece_type
            )

            # How many changed squares does the classifier predict this piece on?
            changed_preds = [
                (sq, float(adjusted[sq][_FEN_CLASS_TO_IDX[piece_type]]))
                for sq in changed
                if FEN_CLASSES[int(np.argmax(adjusted[sq]))] == piece_type
            ]

            budget = confirmed_count - stable_count  # how many can exist on changed squares
            excess = len(changed_preds) - max(budget, 0)
            if excess <= 0:
                continue

            # Squash the lowest-confidence excess predictions → empty
            changed_preds.sort(key=lambda x: x[1])  # ascending confidence
            empty_idx = _FEN_CLASS_TO_IDX['empty']
            piece_idx = _FEN_CLASS_TO_IDX[piece_type]
            for sq, _ in changed_preds[:excess]:
                vec = adjusted[sq].copy()
                # Swap: give the empty class what the piece class had, zero the piece
                vec[empty_idx] += vec[piece_idx]
                vec[piece_idx] = 0.0
                # Renormalize
                total = vec.sum()
                if total > 0:
                    vec /= total
                adjusted[sq] = vec
