"""
Convert 64-square classifier predictions into a FEN position string.
"""


def predictions_to_fen(predictions: dict, move_number: int = 1) -> str:
    """Convert per-square predictions to a full FEN string.

    Parameters
    ----------
    predictions : dict {square_name: (piece_class, confidence)}
                  square_name like 'a8', 'b8', ..., 'h1'
                  piece_class is one of: 'empty','P','N','B','R','Q','K','p','n','b','r','q','k'
    move_number : int — used to determine whose turn it is (odd = white, even = black)

    Returns
    -------
    str — full FEN string, e.g. 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    """
    files = "abcdefgh"
    ranks_descending = "87654321"  # FEN goes rank 8 first

    fen_rows = []
    for rank in ranks_descending:
        empty_count = 0
        row_str = ""
        for f in files:
            square = f"{f}{rank}"
            piece, _ = predictions.get(square, ('empty', 0.0))

            if piece == 'empty':
                empty_count += 1
            else:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                row_str += piece

        if empty_count > 0:
            row_str += str(empty_count)
        fen_rows.append(row_str)

    position = "/".join(fen_rows)

    # Turn: white moves on odd move numbers
    turn = 'w' if move_number % 2 == 1 else 'b'

    # We can't determine castling rights or en passant from a single image,
    # so use placeholders. The move detector will track these via python-chess.
    return f"{position} {turn} - - 0 {move_number}"


def fen_position_only(fen: str) -> str:
    """Extract just the position part of a FEN string (before the first space)."""
    return fen.split(' ')[0]
