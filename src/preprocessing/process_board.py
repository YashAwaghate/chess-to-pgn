import cv2
import numpy as np
import os
import argparse
import shutil

# CONFIGURATION
SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE  # 400
CANVAS_HEIGHT = BOARD_SIZE    # 400 — no extra height, board fills the whole image
BOARD_START_Y = 0             # Board grid starts at Y=0

def auto_canny(image, sigma=0.33):
    """Calculates adaptive Canny thresholds based on the image median."""
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def preprocess_image(image_path, output_path=None, show=False, rotation=0):
    """
    Reads an image, detects the chess board, warps it to a clean 400x400 view,
    and segments into 64 square patches.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # 1. Preprocessing with CLAHE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)
    edges = auto_canny(blurred)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # 2. ROI Masking
    contours, _ = cv2.findContours(dilated_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    board_approx = None
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 5000:
            board_approx = approx
            break
            
    if board_approx is None:
        masked_edges = dilated_edges
    else:
        mask = np.zeros_like(dilated_edges)
        cv2.drawContours(mask, [board_approx], -1, 255, -1)
        mask = cv2.erode(mask, kernel, iterations=5)
        masked_edges = cv2.bitwise_and(dilated_edges, dilated_edges, mask=mask)

    # 3. Robust Hough Grid Detection
    lines = cv2.HoughLines(masked_edges, 1, np.pi / 180, 60)
    grid_corners = None
    used_method = "Contour (Fallback)"
    
    if lines is not None:
        horizontals, verticals = cluster_hough_lines(lines)
        h_seq = find_and_complete_grid(horizontals, img.shape[0])
        v_seq = find_and_complete_grid(verticals, img.shape[1])
        
        if len(h_seq) == 9 and len(v_seq) == 9:
            p_tl = intersection(h_seq[0], v_seq[0])
            p_tr = intersection(h_seq[0], v_seq[-1])
            p_br = intersection(h_seq[-1], v_seq[-1])
            p_bl = intersection(h_seq[-1], v_seq[0])
            grid_corners = np.array([p_tl, p_tr, p_br, p_bl], dtype="float32")
            used_method = "Extrapolated Grid"

    if grid_corners is None:
        if board_approx is not None:
             grid_corners = board_approx.reshape(4, 2).astype("float32")
        else:
             return None

    # 4. Rectification — inset margin to remove board notation/border
    margin = 30.0
    dst_points = np.array([
        [-margin, -margin],
        [400 + margin, -margin],
        [400 + margin, 400 + margin],
        [-margin, 400 + margin]
    ], dtype="float32")
    
    rect_points = order_points(grid_corners)
    M = cv2.getPerspectiveTransform(rect_points, dst_points)
    warped = cv2.warpPerspective(img, M, (400, CANVAS_HEIGHT))
    
    if output_path:
        cv2.imwrite(output_path, warped)
        segment_board(warped, output_path, rotation)
        print(f"Using method: {used_method}. Saved to {output_path}")
        
    return warped

def find_and_complete_grid(lines, max_dim):
    """Finds equidistant lines and extrapolates to reach a full 9-line grid."""
    if len(lines) < 2: return lines
    lines = sorted(lines, key=lambda x: x[0])
    rhos = [l[0] for l in lines]
    all_diffs = np.diff(rhos)
    valid_diffs = [d for d in all_diffs if d > 10]
    if not valid_diffs: return lines
    median_spacing = np.median(valid_diffs)
    
    sequences = []
    current_seq = [lines[0]]
    for i in range(1, len(lines)):
        dist = lines[i][0] - current_seq[-1][0]
        if abs(dist - median_spacing) < median_spacing * 0.35:
            current_seq.append(lines[i])
        elif dist > median_spacing * 1.5:
             if len(current_seq) >= 3: sequences.append(current_seq)
             current_seq = [lines[i]]
    if len(current_seq) >= 3: sequences.append(current_seq)
    if not sequences: return lines
    best_seq = max(sequences, key=len)
    
    if 6 <= len(best_seq) < 9:
        grid_spacing = np.mean(np.diff([l[0] for l in best_seq]))
        while len(best_seq) < 9:
            new_rho = best_seq[-1][0] + grid_spacing
            if new_rho < max_dim * 1.5: best_seq.append([new_rho, best_seq[-1][1]])
            else: break
        while len(best_seq) < 9:
            best_seq.insert(0, [best_seq[0][0] - grid_spacing, best_seq[0][1]])
    return best_seq

def cluster_hough_lines(lines):
    horizontals, verticals = [], []
    for line in lines:
        rho, theta = line[0]
        if theta < 0: rho, theta = -rho, theta + np.pi
        if theta < np.deg2rad(15) or theta > np.deg2rad(165): verticals.append([rho, theta])
        elif abs(theta - np.pi/2) < np.deg2rad(15): horizontals.append([rho, theta])
            
    def merge_nearby(lines_list):
        if not lines_list: return []
        lines_list.sort(key=lambda x: x[0])
        merged, curr_cluster = [], [lines_list[0]]
        for i in range(1, len(lines_list)):
            if lines_list[i][0] - curr_cluster[-1][0] < 20: curr_cluster.append(lines_list[i])
            else:
                merged.append((np.median([c[0] for c in curr_cluster]), np.median([c[1] for c in curr_cluster])))
                curr_cluster = [lines_list[i]]
        merged.append((np.median([c[0] for c in curr_cluster]), np.median([c[1] for c in curr_cluster])))
        return merged
    return merge_nearby(horizontals), merge_nearby(verticals)

def segment_board(board_image, output_path, rotation=0):
    """Slices the 400x400 board into 64 clean 50x50 patches."""
    base_dir = os.path.splitext(output_path)[0] + "_patches"
    if os.path.exists(base_dir): shutil.rmtree(base_dir) 
    os.makedirs(base_dir, exist_ok=True)
    
    internal_crop = 5 
    
    # Base logical files and ranks
    files = "abcdefgh"
    ranks = "87654321"
    
    # Adjust logical mapping based on rotation (0, 90, 180, 270 degrees clockwise)
    # The physical patches extracted from top-left (i=0, j=0) to bottom-right (i=7, j=7)
    # represent different logical squares depending on how the camera viewed the board.
    # 0 deg (Standard): Top row is rank 8, Bottom is rank 1. Left is file 'a', Right is file 'h'.
    
    for i in range(8):
        for j in range(8):
            # Calculate logical rank and file based on orientation
            # Standard (0): i -> rank (0=8), j -> file (0=a)
            # 180 rotated: board is viewed from Black's side. Top row is rank 1. Left is file 'h'.
            # 90 rotated: board viewed from side.
            if rotation == 0:
                logical_file_idx, logical_rank_idx = j, i
            elif rotation == 180:
                logical_file_idx, logical_rank_idx = 7 - j, 7 - i
            elif rotation == 90:
                # View from left side (e.g. White is on the right)
                logical_file_idx, logical_rank_idx = i, 7 - j
            elif rotation == 270:
                # View from right side (e.g. White is on the left)
                logical_file_idx, logical_rank_idx = 7 - i, j
            else:
                logical_file_idx, logical_rank_idx = j, i
                
            square_name = f"{files[logical_file_idx]}{ranks[logical_rank_idx]}"

            y_base = i * SQUARE_SIZE
            y_end = (i + 1) * SQUARE_SIZE
            x_start = j * SQUARE_SIZE
            x_end = (j + 1) * SQUARE_SIZE

            patch = board_image[y_base:y_end, x_start:x_end]

            # Trim edges to remove grid lines
            h, w = patch.shape[:2]
            cropped = patch[internal_crop:h-internal_crop, internal_crop:w-internal_crop]
            
            # Normalize and resize back to 50x50 (the piece will be vertically scaled)
            normalized = cv2.normalize(cropped, None, 0, 255, cv2.NORM_MINMAX)
            final_patch = cv2.resize(normalized, (50, 50))
            
            cv2.imwrite(os.path.join(base_dir, f"{square_name}.jpg"), final_patch)
    print(f"Segmented 64 full-height patches saved to {base_dir}")

def determine_orientation(warped_board):
    """
    Analyzes the '00' setup image to determine rotation.
    1. Piece density: White pieces in bottom ranks -> 0 or 90. Black pieces -> 180 or 270.
    2. Square Parity: Bottom right square (h1) must be a physical light square.
    """
    if warped_board is None:
        return 0
        
    # Analyze bottom two ranks (indices i=6,7, physical bottom)
    # Get raw physical patches for these ranks to check piece density
    bottom_ranks_intensity = 0
    top_ranks_intensity = 0
    
    internal_crop = 5
    for i in range(8):
        for j in range(8):
            y_base = i * SQUARE_SIZE
            y_end = (i + 1) * SQUARE_SIZE
            x_start = j * SQUARE_SIZE
            x_end = (j + 1) * SQUARE_SIZE
            patch = warped_board[y_base:y_end, x_start:x_end]
            hc, wc = patch.shape[:2]
            cropped = patch[internal_crop:hc-internal_crop, internal_crop:wc-internal_crop]
            
            gray_patch = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            # Estimate piece brightness (assuming pieces occupy center of patch)
            # We can just take average pixel value. White pieces -> higher mean.
            intensity = np.mean(gray_patch)
            
            if i >= 6: # Bottom 2 ranks
                bottom_ranks_intensity += intensity
            elif i <= 1: # Top 2 ranks
                top_ranks_intensity += intensity

    # If bottom ranks are brighter than top, White is at the bottom.
    is_white_at_bottom = bottom_ranks_intensity > top_ranks_intensity
    
    # 2. Square Parity Verification. Physical bottom right is patch (i=7, j=7)
    y_base = 7 * SQUARE_SIZE
    y_end = 8 * SQUARE_SIZE
    x_start = 7 * SQUARE_SIZE
    x_end = 8 * SQUARE_SIZE
    h1_patch = warped_board[y_base:y_end, x_start:x_end]
    gray_h1 = cv2.cvtColor(h1_patch, cv2.COLOR_BGR2GRAY)
    
    # Extract just the corners of the square to avoid piece interference
    h, w = gray_h1.shape
    c_size = 5
    corners = np.concatenate([
        gray_h1[0:c_size, 0:c_size].flatten(),
        gray_h1[0:c_size, w-c_size:w].flatten(),
        gray_h1[h-c_size:h, 0:c_size].flatten(),
        gray_h1[h-c_size:h, w-c_size:w].flatten()
    ])
    
    h1_brightness = np.median(corners)
    # Estimate light vs dark square by thresholding (naive)
    # A better way is comparing h1 (7,7) to g1 (7,6)
    
    y_base_g1 = 7 * SQUARE_SIZE
    y_end_g1 = 8 * SQUARE_SIZE
    x_start_g1 = 6 * SQUARE_SIZE
    x_end_g1 = 7 * SQUARE_SIZE
    g1_patch = warped_board[y_base_g1:y_end_g1, x_start_g1:x_end_g1]
    gray_g1 = cv2.cvtColor(g1_patch, cv2.COLOR_BGR2GRAY)
    corners_g1 = np.concatenate([
        gray_g1[0:c_size, 0:c_size].flatten(),
        gray_g1[0:c_size, w-c_size:w].flatten(),
        gray_g1[h-c_size:h, 0:c_size].flatten(),
        gray_g1[h-c_size:h, w-c_size:w].flatten()
    ])
    g1_brightness = np.median(corners_g1)
    
    # h1 is typically a light square.
    is_h1_light = h1_brightness > g1_brightness
    
    # Combine logic:
    if is_white_at_bottom:
        if is_h1_light: return 0
        else: return 90 # Flipped sideways
    else:
        if is_h1_light: return 180
        else: return 270

def intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([rho1, rho2])
    try: return np.linalg.solve(A, b)
    except: return [0,0]

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def crop_squares_from_grid(board_img, grid, rotation=0):
    """Crop 64 squares using corrected grid lines (9 x_lines + 9 y_lines).

    Parameters
    ----------
    board_img : numpy array — 400×400 warped+rotated board image (BGR).
    grid      : dict with 'x_lines' (9 ints) and 'y_lines' (9 ints).
    rotation  : int — 0, 90, 180, or 270. Already applied to the image,
                used only for mapping physical (row, col) to logical square names.

    Returns
    -------
    dict  {square_name: numpy_patch}  e.g. {'a8': array, 'b8': array, ...}
          Each patch is resized to 50×50.
    """
    x_lines = grid['x_lines']
    y_lines = grid['y_lines']
    patches = {}
    files = "abcdefgh"
    ranks = "87654321"

    for i in range(8):
        for j in range(8):
            x1, x2 = x_lines[j], x_lines[j + 1]
            y1, y2 = y_lines[i], y_lines[i + 1]
            patch = board_img[y1:y2, x1:x2]

            if rotation == 0:
                fi, ri = j, i
            elif rotation == 180:
                fi, ri = 7 - j, 7 - i
            elif rotation == 90:
                fi, ri = i, 7 - j
            elif rotation == 270:
                fi, ri = 7 - i, j
            else:
                fi, ri = j, i

            square_name = f"{files[fi]}{ranks[ri]}"
            patches[square_name] = cv2.resize(patch, (50, 50))

    return patches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    preprocess_image(args.image, args.output)