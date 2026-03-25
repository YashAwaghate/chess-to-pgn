"""
Detect chess moves by diffing consecutive FEN positions and validating
against legal moves using python-chess.
"""

import chess
from typing import Optional


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


def detect_move(prev_fen_pos: str, curr_fen_pos: str, board: chess.Board) -> Optional[str]:
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
    # (allows for 1-2 classifier errors per board)
    if best_move and best_score >= 62:  # at least 62/64 squares match
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


def detect_moves_sequence(fen_positions: list) -> dict:
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

        san = detect_move(prev_pos, curr_pos, board)

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
