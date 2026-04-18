"""
Tests for src/pipeline/fen_generator.py

Coverage targets:
  - predictions_to_fen: starting position, empty board, partial board, turn alternation
  - fen_position_only: strips metadata fields correctly
"""

import pytest
from src.pipeline.fen_generator import predictions_to_fen, fen_position_only


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _piece(cls: str, conf: float = 0.99):
    """Shorthand: build a single prediction tuple."""
    return (cls, conf)


def _starting_predictions() -> dict:
    """Return the classifier predictions for the standard chess starting position."""
    preds = {}

    # Rank 8 — black back rank
    for f, piece in zip("abcdefgh", "rnbqkbnr"):
        preds[f"{f}8"] = _piece(piece)

    # Rank 7 — black pawns
    for f in "abcdefgh":
        preds[f"{f}7"] = _piece("p")

    # Ranks 6–3 — empty
    for rank in "6543":
        for f in "abcdefgh":
            preds[f"{f}{rank}"] = _piece("empty")

    # Rank 2 — white pawns
    for f in "abcdefgh":
        preds[f"{f}2"] = _piece("P")

    # Rank 1 — white back rank
    for f, piece in zip("abcdefgh", "RNBQKBNR"):
        preds[f"{f}1"] = _piece(piece)

    return preds


# ---------------------------------------------------------------------------
# predictions_to_fen
# ---------------------------------------------------------------------------

class TestPredictionsToFen:

    def test_starting_position_fen_position(self):
        preds = _starting_predictions()
        fen = predictions_to_fen(preds, move_number=1)
        position = fen.split(" ")[0]
        assert position == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

    def test_white_to_move_on_odd_move_number(self):
        preds = _starting_predictions()
        fen = predictions_to_fen(preds, move_number=1)
        assert fen.split(" ")[1] == "w"

    def test_black_to_move_on_even_move_number(self):
        preds = _starting_predictions()
        fen = predictions_to_fen(preds, move_number=2)
        assert fen.split(" ")[1] == "b"

    def test_empty_board(self):
        preds = {f"{f}{r}": _piece("empty") for r in "12345678" for f in "abcdefgh"}
        fen = predictions_to_fen(preds)
        position = fen.split(" ")[0]
        assert position == "8/8/8/8/8/8/8/8"

    def test_single_piece_on_e4(self):
        """One white pawn on e4, everything else empty."""
        preds = {f"{f}{r}": _piece("empty") for r in "12345678" for f in "abcdefgh"}
        preds["e4"] = _piece("P")
        fen = predictions_to_fen(preds)
        position = fen.split(" ")[0]
        # rank 4 should be '4P3'
        rank4 = position.split("/")[4]  # FEN rank index: 8,7,6,5,4 → idx 4
        assert rank4 == "4P3"

    def test_missing_squares_treated_as_empty(self):
        """predictions dict with no keys → fully empty board."""
        fen = predictions_to_fen({})
        position = fen.split(" ")[0]
        assert position == "8/8/8/8/8/8/8/8"

    def test_move_number_encoded_in_fen(self):
        preds = _starting_predictions()
        fen = predictions_to_fen(preds, move_number=5)
        fields = fen.split(" ")
        assert fields[-1] == "5"        # fullmove number
        assert fields[1] == "w"         # move 5 is white's turn

    def test_all_queens_board(self):
        """Edge case: every square holds a white queen."""
        preds = {f"{f}{r}": _piece("Q") for r in "12345678" for f in "abcdefgh"}
        fen = predictions_to_fen(preds)
        position = fen.split(" ")[0]
        assert position == "QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ"

    def test_confidence_ignored_in_fen_building(self):
        """Confidence values must not affect the position string."""
        preds_high = {f"{f}{r}": ("P", 0.99) for r in "12345678" for f in "abcdefgh"}
        preds_low  = {f"{f}{r}": ("P", 0.01) for r in "12345678" for f in "abcdefgh"}
        assert predictions_to_fen(preds_high).split(" ")[0] == predictions_to_fen(preds_low).split(" ")[0]

    def test_consecutive_empty_runs_compressed(self):
        """Rank with alternating piece-empty should not merge runs across pieces."""
        preds = {f"{f}{r}": _piece("empty") for r in "12345678" for f in "abcdefgh"}
        # Place pieces on a1 and h1
        preds["a1"] = _piece("R")
        preds["h1"] = _piece("R")
        fen = predictions_to_fen(preds)
        rank1 = fen.split(" ")[0].split("/")[7]
        assert rank1 == "R6R"


# ---------------------------------------------------------------------------
# fen_position_only
# ---------------------------------------------------------------------------

class TestFenPositionOnly:

    def test_extracts_position_from_full_fen(self):
        full = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert fen_position_only(full) == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

    def test_position_only_string_is_idempotent(self):
        pos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        assert fen_position_only(pos) == pos

    def test_empty_board_fen(self):
        full = "8/8/8/8/8/8/8/8 w - - 0 1"
        assert fen_position_only(full) == "8/8/8/8/8/8/8/8"
