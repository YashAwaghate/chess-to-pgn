"""
Generate PGN (Portable Game Notation) from a list of SAN moves and game metadata.
"""

import datetime


def generate_pgn(moves: list, game_info: dict, result: str = '*') -> str:
    """Build a PGN string from moves and game metadata.

    Parameters
    ----------
    moves     : list of str — SAN notation moves ['e4', 'e5', 'Nf3', ...]
    game_info : dict — metadata from game_info.json (white, black, event, site, date, etc.)
    result    : str — game result: '1-0', '0-1', '1/2-1/2', or '*'

    Returns
    -------
    str — complete PGN string
    """
    # Headers
    headers = []
    headers.append(('Event', game_info.get('event', '?')))
    headers.append(('Site', game_info.get('site', '?')))
    headers.append(('Date', _format_date(game_info.get('date', ''))))
    headers.append(('Round', game_info.get('round', '-')))
    headers.append(('White', game_info.get('white', '?')))
    headers.append(('Black', game_info.get('black', '?')))
    headers.append(('Result', result))

    tc = game_info.get('time_control', '')
    if tc:
        headers.append(('TimeControl', tc))

    notes = game_info.get('notes', '')
    if notes:
        headers.append(('Annotator', notes))

    # Build header block
    pgn_lines = []
    for tag, value in headers:
        pgn_lines.append(f'[{tag} "{value}"]')
    pgn_lines.append('')  # blank line between headers and movetext

    # Movetext
    movetext_parts = []
    for i, san in enumerate(moves):
        move_num = i // 2 + 1
        if i % 2 == 0:
            movetext_parts.append(f'{move_num}. {san}')
        else:
            movetext_parts.append(san)

    movetext = ' '.join(movetext_parts)
    if movetext:
        movetext += ' ' + result
    else:
        movetext = result

    # Wrap movetext at ~80 chars
    pgn_lines.append(_wrap_movetext(movetext))
    pgn_lines.append('')  # trailing newline

    return '\n'.join(pgn_lines)


def _format_date(date_str: str) -> str:
    """Convert date to PGN format YYYY.MM.DD."""
    if not date_str:
        today = datetime.date.today()
        return today.strftime('%Y.%m.%d')
    # Handle ISO format (2026-03-25)
    try:
        d = datetime.date.fromisoformat(date_str)
        return d.strftime('%Y.%m.%d')
    except (ValueError, TypeError):
        return date_str.replace('-', '.')


def _wrap_movetext(text: str, width: int = 80) -> str:
    """Wrap movetext to approximately `width` characters per line."""
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if current_line and len(current_line) + 1 + len(word) > width:
            lines.append(current_line)
            current_line = word
        else:
            current_line = f"{current_line} {word}" if current_line else word

    if current_line:
        lines.append(current_line)

    return '\n'.join(lines)


def save_pgn(pgn_str: str, output_path: str):
    """Write PGN string to a file."""
    with open(output_path, 'w') as f:
        f.write(pgn_str)
