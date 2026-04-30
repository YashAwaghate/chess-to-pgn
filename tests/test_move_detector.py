"""
Tests for src/pipeline/move_detector.py

Coverage targets:
  - fen_to_piece_map: parsing FEN position strings
  - diff_positions: clearing / placing / changing squares
  - detect_move: exact match, fuzzy match, no-match, null-move guard
  - _position_similarity: counting matching squares
  - compute_consensus_predictions: majority vote, tie-breaking by confidence
  - detect_moves_sequence: full sequence with errors and skips
"""

import chess
import numpy as np
import pytest

from src.pipeline.move_detector import (
    FEN_CLASSES,
    TemporalBoardTracker,
    fen_to_piece_map,
    diff_positions,
    detect_move,
    _position_similarity,
    compute_consensus_predictions,
    detect_moves_sequence,
    detect_move_with_feedback,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STARTING_POS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
AFTER_E4_POS = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"


def _pred(piece: str, conf: float = 0.98):
    return (piece, conf)


def _board_predictions(board: chess.Board) -> dict:
    """Build a perfect predictions dict from the current board state."""
    preds = {}
    files = "abcdefgh"
    for rank in range(1, 9):
        for f in files:
            sq_name = f"{f}{rank}"
            sq = chess.parse_square(sq_name)
            piece = board.piece_at(sq)
            symbol = piece.symbol() if piece else "empty"
            preds[sq_name] = _pred(symbol)
    return preds


def _probs(piece: str, conf: float = 0.98) -> np.ndarray:
    vec = np.full(len(FEN_CLASSES), (1.0 - conf) / (len(FEN_CLASSES) - 1),
                  dtype=np.float32)
    vec[FEN_CLASSES.index(piece)] = conf
    return vec


def _board_full_probs(board: chess.Board) -> dict:
    """Build a near-certain full-probability dict from the current board."""
    probs = {}
    for rank in range(1, 9):
        for f in "abcdefgh":
            sq_name = f"{f}{rank}"
            piece = board.piece_at(chess.parse_square(sq_name))
            probs[sq_name] = _probs(piece.symbol() if piece else "empty")
    return probs


# ---------------------------------------------------------------------------
# fen_to_piece_map
# ---------------------------------------------------------------------------

class TestFenToPieceMap:

    def test_starting_position_has_32_pieces(self):
        pieces = fen_to_piece_map(STARTING_POS)
        assert len(pieces) == 32

    def test_e1_is_white_king(self):
        pieces = fen_to_piece_map(STARTING_POS)
        assert pieces["e1"] == "K"

    def test_e8_is_black_king(self):
        pieces = fen_to_piece_map(STARTING_POS)
        assert pieces["e8"] == "k"

    def test_empty_squares_omitted(self):
        pieces = fen_to_piece_map(STARTING_POS)
        assert "e4" not in pieces

    def test_empty_board(self):
        assert fen_to_piece_map("8/8/8/8/8/8/8/8") == {}

    def test_after_e4_pawn_on_e4(self):
        pieces = fen_to_piece_map(AFTER_E4_POS)
        assert pieces["e4"] == "P"
        assert "e2" not in pieces   # pawn left e2


# ---------------------------------------------------------------------------
# diff_positions
# ---------------------------------------------------------------------------

class TestDiffPositions:

    def test_e4_move_detected(self):
        diff = diff_positions(STARTING_POS, AFTER_E4_POS)
        cleared_squares = [sq for sq, _ in diff["cleared"]]
        placed_squares  = [sq for sq, _ in diff["placed"]]
        assert "e2" in cleared_squares
        assert "e4" in placed_squares

    def test_identical_positions_no_diff(self):
        diff = diff_positions(STARTING_POS, STARTING_POS)
        assert diff == {"cleared": [], "placed": [], "changed": []}

    def test_changed_square_detected(self):
        # Simulate a promotion: pawn on a8 replaced by queen
        pos_a = "P7/8/8/8/8/8/8/8"
        pos_b = "Q7/8/8/8/8/8/8/8"
        diff = diff_positions(pos_a, pos_b)
        assert len(diff["changed"]) == 1
        sq, old, new = diff["changed"][0]
        assert sq == "a8" and old == "P" and new == "Q"


# ---------------------------------------------------------------------------
# _position_similarity
# ---------------------------------------------------------------------------

class TestPositionSimilarity:

    def test_identical_positions_score_64(self):
        assert _position_similarity(STARTING_POS, STARTING_POS) == 64

    def test_completely_different_positions(self):
        all_white = "QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ"
        all_black = "qqqqqqqq/qqqqqqqq/qqqqqqqq/qqqqqqqq/qqqqqqqq/qqqqqqqq/qqqqqqqq/qqqqqqqq"
        assert _position_similarity(all_white, all_black) == 0

    def test_one_square_different(self):
        # Starting pos vs after e4: e2 cleared, e4 placed → 2 squares differ
        score = _position_similarity(STARTING_POS, AFTER_E4_POS)
        assert score == 62


# ---------------------------------------------------------------------------
# detect_move
# ---------------------------------------------------------------------------

class TestDetectMove:

    def test_detects_e4_from_starting(self):
        board = chess.Board()
        san = detect_move(STARTING_POS, AFTER_E4_POS, board)
        assert san == "e4"

    def test_no_change_returns_none(self):
        board = chess.Board()
        san = detect_move(STARTING_POS, STARTING_POS, board)
        assert san is None

    def test_illegal_position_returns_none(self):
        board = chess.Board()
        # Position with both kings missing — no legal move can reach it
        no_kings = "rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1BNR"
        san = detect_move(STARTING_POS, no_kings, board)
        assert san is None

    def test_detects_nf3(self):
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        prev = board.fen().split(" ")[0]
        board.push_san("Nf3")
        curr = board.fen().split(" ")[0]
        board.pop()  # rewind
        san = detect_move(prev, curr, board)
        assert san == "Nf3"

    def test_fuzzy_match_within_threshold(self):
        """Simulate a 2-square classifier error on an otherwise-correct e4 move."""
        board = chess.Board()
        # Corrupt AFTER_E4_POS by flipping one unrelated pawn (a7→ empty, a6→ p)
        corrupted = AFTER_E4_POS.replace("pppppppp", "1ppppppp").replace(
            "8/8/8/", "p7/8/"
        )
        # The score will be ≥ 62 so fuzzy match should still fire
        san = detect_move(STARTING_POS, corrupted, board, fuzzy_threshold=60)
        assert san == "e4"


# ---------------------------------------------------------------------------
# compute_consensus_predictions
# ---------------------------------------------------------------------------

class TestComputeConsensusPredictions:

    def test_unanimous_frames_return_same_piece(self):
        frame = {f"{f}{r}": _pred("P") for r in "12345678" for f in "abcdefgh"}
        frames = [frame] * 5
        consensus = compute_consensus_predictions(frames)
        assert consensus["e4"][0] == "P"

    def test_majority_wins(self):
        sq = "e4"
        # 3 frames say "P", 2 say "empty"
        frames = [
            {sq: _pred("P")},
            {sq: _pred("P")},
            {sq: _pred("P")},
            {sq: _pred("empty")},
            {sq: _pred("empty")},
        ]
        consensus = compute_consensus_predictions(frames)
        assert consensus[sq][0] == "P"

    def test_tie_broken_by_confidence(self):
        sq = "d5"
        # 2 "P" frames (low conf) vs 2 "empty" frames (high conf)
        frames = [
            {sq: ("P",     0.60)},
            {sq: ("P",     0.60)},
            {sq: ("empty", 0.95)},
            {sq: ("empty", 0.95)},
        ]
        consensus = compute_consensus_predictions(frames)
        # Tie on vote count → higher total conf wins → "empty"
        assert consensus[sq][0] == "empty"

    def test_empty_frames_returns_empty_dict(self):
        assert compute_consensus_predictions([]) == {}

    def test_vote_fraction_in_range(self):
        frames = [{"e4": _pred("P")}] * 3 + [{"e4": _pred("empty")}] * 1
        consensus = compute_consensus_predictions(frames)
        frac = consensus["e4"][1]
        assert 0.0 <= frac <= 1.0


# ---------------------------------------------------------------------------
# detect_moves_sequence
# ---------------------------------------------------------------------------

class TestDetectMovesSequence:

    def _play_sequence(self, sans: list) -> list:
        """Return list of FEN position strings by replaying a move sequence."""
        board = chess.Board()
        positions = [board.fen().split(" ")[0]]
        for san in sans:
            board.push_san(san)
            positions.append(board.fen().split(" ")[0])
        return positions

    def test_too_few_positions_returns_empty(self):
        result = detect_moves_sequence(["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"])
        assert result["moves"] == []
        assert result["errors"] == []

    def test_scholars_mate(self):
        sans = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"]
        positions = self._play_sequence(sans)
        result = detect_moves_sequence(positions)
        # python-chess appends '#' for checkmate; normalise before comparing
        normalised = [m.rstrip("+#") for m in result["moves"]]
        expected   = [m.rstrip("+#") for m in sans]
        assert normalised == expected
        assert result["errors"] == []

    def test_duplicate_positions_counted_as_skipped(self):
        positions = self._play_sequence(["e4"])
        # Duplicate the last position
        positions.append(positions[-1])
        result = detect_moves_sequence(positions)
        assert result["skipped"] == 1
        assert "e4" in result["moves"]

    def test_bad_position_logged_as_error(self):
        positions = self._play_sequence(["e4"])
        positions.append("8/8/8/8/8/8/8/8")   # garbage — not reachable
        result = detect_moves_sequence(positions)
        assert len(result["errors"]) >= 1
        assert result["errors"][0]["reason"] == "no_legal_move_match"

    def test_board_state_matches_moves(self):
        sans = ["d4", "d5", "c4"]
        positions = self._play_sequence(sans)
        result = detect_moves_sequence(positions)
        assert result["board"].fullmove_number == 2   # after 3 half-moves


# ---------------------------------------------------------------------------
# detect_move_with_feedback
# ---------------------------------------------------------------------------

class TestDetectMoveWithFeedback:

    def test_exact_match_tagged_sure(self):
        board = chess.Board()
        preds = _board_predictions(board)
        # Apply e4 to get the "current" position
        board_copy = chess.Board()
        board_copy.push_san("e4")
        preds_after = _board_predictions(board_copy)

        san, tag, adj = detect_move_with_feedback(
            STARTING_POS, preds_after, chess.Board()
        )
        assert san == "e4"
        assert tag == "sure"
        assert adj == []

    def test_failed_returns_none_and_failed_tag(self):
        board = chess.Board()
        # Predictions that describe an unreachable position (all squares empty)
        preds = {f"{f}{r}": _pred("empty") for r in "12345678" for f in "abcdefgh"}
        san, tag, _ = detect_move_with_feedback(STARTING_POS, preds, board)
        assert san is None
        assert tag == "failed"


# ---------------------------------------------------------------------------
# TemporalBoardTracker
# ---------------------------------------------------------------------------

class TestTemporalBoardTracker:

    def test_confirms_legal_candidate_with_two_wrong_argmax_squares(self):
        tracker = TemporalBoardTracker()
        board_after = chess.Board()
        board_after.push_san("e4")
        probs = _board_full_probs(board_after)
        probs["a7"] = _probs("empty")
        probs["h1"] = _probs("empty")

        san, tag = tracker.push(probs)

        assert san == "e4"
        assert tag in {"sure", "prior"}
        assert tracker.board.fen().split(" ")[0] == AFTER_E4_POS

    def test_initializes_from_first_unreachable_frame_without_advancing_board(self):
        tracker = TemporalBoardTracker()
        board_after = chess.Board()
        board_after.push_san("e4")
        probs = _board_full_probs(board_after)
        probs["a7"] = _probs("empty")
        probs["h1"] = _probs("empty")
        probs["b8"] = _probs("empty")

        san, tag = tracker.push(probs)

        assert san is None
        assert tag == "no_change"
        assert tracker.board.fen().split(" ")[0] == STARTING_POS
        assert tracker._confirmed_pos != STARTING_POS
