import cv2
import numpy as np
import os
import argparse
import shutil

# CONFIGURATION
SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE  # 400
CANVAS_HEIGHT = 500           # Extra height to capture piece height
BOARD_START_Y = 100           # The board grid starts at Y=100 in the warped image
PIECE_HEAD_BUFFER = 60        # How many pixels above the square to look for the head

def auto_canny(image, sigma=0.33):
    """Calculates adaptive Canny thresholds based on the image median."""
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def preprocess_image(image_path, output_path=None, show=False):
    """
    Reads an image, detects the chess board, warps it to a 400x500 view 
    (shifted down to capture piece heads), and segments patches.
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

    # 4. Rectification with Vertical Offset
    # Shift board down to Y=100 and apply Inset margin to remove notation
    margin = 30.0 
    dst_points = np.array([
        [-margin, BOARD_START_Y - margin],
        [400 + margin, BOARD_START_Y - margin],
        [400 + margin, 500 + margin],
        [-margin, 500 + margin]
    ], dtype="float32")
    
    rect_points = order_points(grid_corners)
    M = cv2.getPerspectiveTransform(rect_points, dst_points)
    warped = cv2.warpPerspective(img, M, (400, CANVAS_HEIGHT))
    
    if output_path:
        cv2.imwrite(output_path, warped)
        segment_board(warped, output_path)
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

def segment_board(board_image, output_path):
    """Slices the grid into 64 patches, capturing piece height by looking upward."""
    base_dir = os.path.splitext(output_path)[0] + "_patches"
    if os.path.exists(base_dir): shutil.rmtree(base_dir) 
    os.makedirs(base_dir, exist_ok=True)
    
    internal_crop = 5 
    files, ranks = "abcdefgh", "87654321"
    
    for i in range(8):
        for j in range(8):
            # The board grid starts at BOARD_START_Y (100)
            y_base = BOARD_START_Y + (i * SQUARE_SIZE)
            y_end = BOARD_START_Y + ((i + 1) * SQUARE_SIZE)
            x_start = j * SQUARE_SIZE
            x_end = (j + 1) * SQUARE_SIZE
            
            # Capture piece heads by extending the crop upward into the sky area
            y_extended = max(0, y_base - PIECE_HEAD_BUFFER)
            
            patch = board_image[y_extended:y_end, x_start:x_end]
            
            # Safety crop for sides/bottom to remove grid lines
            h, w = patch.shape[:2]
            cropped = patch[0:h-internal_crop, internal_crop:w-internal_crop]
            
            # Normalize and resize back to 50x50 (the piece will be vertically scaled)
            normalized = cv2.normalize(cropped, None, 0, 255, cv2.NORM_MINMAX)
            final_patch = cv2.resize(normalized, (50, 50))
            
            cv2.imwrite(os.path.join(base_dir, f"{files[j]}{ranks[i]}.jpg"), final_patch)
    print(f"Segmented 64 full-height patches saved to {base_dir}")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    preprocess_image(args.image, args.output)