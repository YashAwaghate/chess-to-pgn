import os
import glob
import sys
import shutil
import multiprocessing
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Ensure the process_board module can be imported
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from process_board import preprocess_image, determine_orientation

def process_game(args):
    """Processes an entire game folder, finding the '00' image first to determine rotation."""
    game_folder_path, output_dir = args
    game_folder_name = os.path.basename(game_folder_path)
    
    # 1. Find all images in the game
    game_images = glob.glob(os.path.join(game_folder_path, "*.jpg"))
    if not game_images:
        return game_folder_name, 0, 0
        
    # 2. Identify the '00' image
    # Filenames are like G000_IMG000.jpg, G000_IMG001.jpg
    # Some games might have it named slightly differently, let's find the one ending in 000.jpg or similar
    # Sort them to find the first one
    game_images.sort()
    
    # Heuristic: the first image is the initial setup
    setup_image_path = game_images[0]
    
    # Try to find exactly IMG000 or IMG00
    for img in game_images:
        if 'IMG000' in img or 'IMG00.' in img:
            setup_image_path = img
            break

    # 3. Process the '00' image specifically to find orientation
    target_dir = os.path.join(output_dir, game_folder_name)
    os.makedirs(target_dir, exist_ok=True)
    rotation_file = os.path.join(target_dir, "rotation.json")
    
    rotation_angle = 0 # Default
    
    if os.path.exists(rotation_file):
        with open(rotation_file, 'r') as f:
            rotation_angle = json.load(f).get("rotation", 0)
    else:
        # Determine it!
        import cv2
        # Run it through the first steps of preprocess_board manually, or modify it to return warped without saving
        # A simpler way is to just call preprocess_image with no output, then run determine_orientation on the result
        with open(os.devnull, 'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            try:
                # Get the warped board without saving patches to disk yet
                warped_board = preprocess_image(setup_image_path, output_path=None)
                if warped_board is not None:
                    rotation_angle = determine_orientation(warped_board)
            except Exception as e:
                rotation_angle = 0
            finally:
                sys.stdout = old_stdout
                
        # Save it for future reference (or resuming)
        with open(rotation_file, 'w') as f:
            json.dump({"rotation": rotation_angle, "setup_image": os.path.basename(setup_image_path)}, f)
            
    # 4. Process ALL images in this game using the determined rotation_angle
    success_count = 0
    for img_path in game_images:
        filename = os.path.basename(img_path)
        base = os.path.splitext(filename)[0]
        img_num = base.split('IMG')[-1] if 'IMG' in base else base
        
        img_target_dir = os.path.join(target_dir, img_num)
        os.makedirs(img_target_dir, exist_ok=True)
        
        dummy_out = os.path.join(img_target_dir, "board.jpg")
        
        with open(os.devnull, 'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            try:
                preprocess_image(img_path, output_path=dummy_out, rotation=rotation_angle)
                
                patches_dir = os.path.join(img_target_dir, "board_patches")
                if os.path.exists(patches_dir):
                    for patch_file in os.listdir(patches_dir):
                        src_path = os.path.join(patches_dir, patch_file)
                        dest_path = os.path.join(img_target_dir, patch_file)
                        if os.path.exists(dest_path):
                            os.remove(dest_path)
                        shutil.move(src_path, dest_path)
                    os.rmdir(patches_dir)
                    
                success_count += 1
            except Exception as e:
                pass
            finally:
                sys.stdout = old_stdout
                
    return game_folder_name, success_count, len(game_images)

def main():
    dataset_dir = r"f:\Chess-pgn\chess-to-pgn\data\raw\dataset\ChessRed_images"
    output_dir = r"f:\Chess-pgn\chess-to-pgn\data\processed_image_patches"
    
    # Using glob to discover all game folders
    game_folders = glob.glob(os.path.join(dataset_dir, "*"))
    game_folders = [f for f in game_folders if os.path.isdir(f)]
    print(f"Found {len(game_folders)} game folders to process.")
    
    if len(game_folders) == 0:
        print("No folders found! Check the dataset path.")
        return
        
    workers = min(16, os.cpu_count() or 4)
    print(f"Using {workers} CPU cores for multiprocessing by game...")
    
    args_list = [(folder, output_dir) for folder in game_folders]
    
    total_images = 0
    total_success = 0
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_game, args): args for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing games"):
            game_name, success, total = future.result()
            total_success += success
            total_images += total
                
    print(f"\nFinished processing! Successfully created patches for {total_success}/{total_images} images.")
    print(f"Results are saved in {output_dir}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
