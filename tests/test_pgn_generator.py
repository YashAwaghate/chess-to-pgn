"""
Tests for src/pipeline/pgn_generator.py

Coverage targets:
  - generate_pgn: headers, movetext, result appending, move tags, wrapping
  - _format_date: ISO input, empty input, already-formatted input
  - _wrap_movetext: short lines untouched, long lines wrapped
  - save_pgn: file written correctly (tmp_path fixture)
"""

import datetime
import os
import pytest

from src.pipeline.pgn_generator import generate_pgn, save_pgn, _format_date, _wrap_movetext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_INFO = {"white": "Alice", "black": "Bob"}

FULL_INFO = {
    "event": "Office Championship",
    "site": "Break Room",
    "date": "2026-04-18",
    "round": "3",
    "white": "Alice",
    "black": "Bob",
    "time_control": "600+5",
    "notes": "Annotated by Claude",
}

SCHOLARS_MATE = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7"]


# ---------------------------------------------------------------------------
# _format_date
# ---------------------------------------------------------------------------

class TestFormatDate:

    def test_iso_date_converted(self):
        assert _format_date("2026-04-18") == "2026.04.18"

    def test_empty_string_gives_today(self):
        result = _format_date("")
        today = datetime.date.today().strftime("%Y.%m.%d")
        assert result == today

    def test_already_dotted_passthrough(self):
        # Non-ISO format with dashes → replace dashes with dots
        assert _format_date("2026-01-01") == "2026.01.01"

    def test_invalid_date_replaces_dashes(self):
        assert _format_date("not-a-date") == "not.a.date"


# ---------------------------------------------------------------------------
# _wrap_movetext
# ---------------------------------------------------------------------------

class TestWrapMovetext:

    def test_short_text_not_wrapped(self):
        text = "1. e4 e5 *"
        assert _wrap_movetext(text, width=80) == text

    def test_long_text_wrapped(self):
        # Build a string > 80 chars
        moves = " ".join([f"{i}. e4 e5" for i in range(1, 15)])
        wrapped = _wrap_movetext(moves, width=80)
        for line in wrapped.split("\n"):
            assert len(line) <= 80

    def test_empty_text(self):
        assert _wrap_movetext("") == ""


# ---------------------------------------------------------------------------
# generate_pgn — headers
# ---------------------------------------------------------------------------

class TestGeneratePgnHeaders:

    def test_required_headers_present(self):
        pgn = generate_pgn(SCHOLARS_MATE, FULL_INFO, result="1-0")
        for tag in ("[Event", "[Site", "[Date", "[Round", "[White", "[Black", "[Result"):
            assert tag in pgn

    def test_white_and_black_names(self):
        pgn = generate_pgn([], MINIMAL_INFO)
        assert '"Alice"' in pgn
        assert '"Bob"' in pgn

    def test_time_control_included_when_present(self):
        pgn = generate_pgn([], FULL_INFO)
        assert "[TimeControl" in pgn
        assert "600+5" in pgn

    def test_time_control_omitted_when_absent(self):
        pgn = generate_pgn([], MINIMAL_INFO)
        assert "[TimeControl" not in pgn

    def test_annotator_tag_from_notes(self):
        pgn = generate_pgn([], FULL_INFO)
        assert "[Annotator" in pgn

    def test_result_header(self):
        pgn = generate_pgn([], MINIMAL_INFO, result="0-1")
        assert '[Result "0-1"]' in pgn

    def test_default_result_is_asterisk(self):
        pgn = generate_pgn([], MINIMAL_INFO)
        assert '[Result "*"]' in pgn

    def test_missing_event_uses_question_mark(self):
        pgn = generate_pgn([], MINIMAL_INFO)
        assert '[Event "?"]' in pgn


# ---------------------------------------------------------------------------
# generate_pgn — movetext
# ---------------------------------------------------------------------------

class TestGeneratePgnMovetext:

    def test_move_numbers_correct(self):
        pgn = generate_pgn(["e4", "e5", "Nf3"], MINIMAL_INFO)
        assert "1. e4" in pgn
        assert "2. Nf3" in pgn

    def test_result_appended_to_movetext(self):
        pgn = generate_pgn(SCHOLARS_MATE, MINIMAL_INFO, result="1-0")
        assert pgn.rstrip().endswith("1-0")

    def test_empty_moves_result_only(self):
        pgn = generate_pgn([], MINIMAL_INFO, result="*")
        assert "*" in pgn

    def test_unsure_move_gets_question_mark_comment(self):
        moves = ["e4", "e5"]
        tags  = ["sure", "unsure"]
        pgn = generate_pgn(moves, MINIMAL_INFO, move_tags=tags)
        assert "{ ? }" in pgn

    def test_consensus_move_gets_double_question_mark_comment(self):
        moves = ["e4", "e5"]
        tags  = ["sure", "consensus_sure"]
        pgn = generate_pgn(moves, MINIMAL_INFO, move_tags=tags)
        assert "{ ?? }" in pgn

    def test_sure_moves_have_no_comment(self):
        moves = ["e4", "e5"]
        tags  = ["sure", "sure"]
        pgn = generate_pgn(moves, MINIMAL_INFO, move_tags=tags)
        assert "{ " not in pgn

    def test_no_move_tags_no_comments(self):
        pgn = generate_pgn(SCHOLARS_MATE, MINIMAL_INFO)
        assert "{ " not in pgn

    def test_scholars_mate_full_pgn(self):
        pgn = generate_pgn(SCHOLARS_MATE, FULL_INFO, result="1-0")
        assert "Qxf7" in pgn
        assert "1-0" in pgn

    def test_pgn_has_blank_line_between_headers_and_moves(self):
        pgn = generate_pgn(["e4"], MINIMAL_INFO)
        lines = pgn.split("\n")
        # Find last header line
        last_header_idx = max(i for i, l in enumerate(lines) if l.startswith("["))
        assert lines[last_header_idx + 1] == ""   # blank separator

    def test_long_game_movetext_wrapped(self):
        # 40 moves — movetext would exceed 80 chars without wrapping
        moves = ["e4", "e5"] * 20
        pgn = generate_pgn(moves, MINIMAL_INFO)
        movetext_lines = [l for l in pgn.split("\n") if l and not l.startswith("[")]
        for line in movetext_lines:
            assert len(line) <= 80


# ---------------------------------------------------------------------------
# save_pgn
# ---------------------------------------------------------------------------

class TestSavePgn:

    def test_file_created(self, tmp_path):
        out = tmp_path / "game.pgn"
        pgn = generate_pgn(SCHOLARS_MATE, FULL_INFO, result="1-0")
        save_pgn(pgn, str(out))
        assert out.exists()

    def test_file_contents_match(self, tmp_path):
        out = tmp_path / "game.pgn"
        pgn = generate_pgn(["e4", "e5"], MINIMAL_INFO)
        save_pgn(pgn, str(out))
        assert out.read_text() == pgn

    def test_overwrites_existing_file(self, tmp_path):
        out = tmp_path / "game.pgn"
        out.write_text("old content")
        pgn = generate_pgn([], MINIMAL_INFO)
        save_pgn(pgn, str(out))
        assert out.read_text() == pgn
