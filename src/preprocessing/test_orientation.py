import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from batch_process_images import process_game

if __name__ == '__main__':
    game_folder = r"f:\Chess-pgn\chess-to-pgn\data\raw\dataset\ChessRed_images\0"
    output_dir = r"f:\Chess-pgn\chess-to-pgn\data\temp_test_output"
    
    print(f"Testing process_game on {game_folder}")
    result = process_game((game_folder, output_dir))
    print(f"Result: {result}")
