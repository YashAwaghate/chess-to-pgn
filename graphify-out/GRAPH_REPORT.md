# Graph Report - chess-to-pgn  (2026-04-30)

## Corpus Check
- 45 files · ~68,004 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 884 nodes · 1903 edges · 64 communities detected
- Extraction: 46% EXTRACTED · 54% INFERRED · 0% AMBIGUOUS · INFERRED: 1027 edges (avg confidence: 0.56)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]

## God Nodes (most connected - your core abstractions)
1. `ChessPieceClassifier` - 294 edges
2. `HandDetector` - 268 edges
3. `CornerDetector` - 247 edges
4. `generate_pgn()` - 28 edges
5. `TemporalBoardTracker` - 27 edges
6. `ChessPieceCNN` - 25 edges
7. `process_game_session()` - 25 edges
8. `PretrainedBoardClassifier` - 24 edges
9. `predictions_to_fen()` - 20 edges
10. `_game_info()` - 19 edges

## Surprising Connections (you probably didn't know these)
- `Run the trained corner detector on a set of images.  Two input modes:   1. Chess` --uses--> `CornerDetector`  [INFERRED]
  .graphify-slim-20260429_153932\scripts\auto_annotate_corners.py → .graphify-slim-20260429_153932\src\models\corner_detector.py
- `Per-frame breakdown for game 62 — find which frames are bad.` --uses--> `ChessPieceClassifier`  [INFERRED]
  .graphify-slim-20260429_153932\scripts\debug_g62_per_frame.py → .graphify-slim-20260429_153932\src\models\inference.py
- `Count matching squares for an 'empty-aware' comparison.` --uses--> `ChessPieceClassifier`  [INFERRED]
  .graphify-slim-20260429_153932\scripts\eval_masouris_metrics.py → .graphify-slim-20260429_153932\src\models\inference.py
- `Derive ground-truth SAN move list by replaying the FEN sequence.` --uses--> `ChessPieceClassifier`  [INFERRED]
  .graphify-slim-20260429_153932\scripts\eval_masouris_metrics.py → .graphify-slim-20260429_153932\src\models\inference.py
- `Group by game and run both pipelines (feedback vs Bayesian prior).` --uses--> `ChessPieceClassifier`  [INFERRED]
  .graphify-slim-20260429_153932\scripts\eval_masouris_metrics.py → .graphify-slim-20260429_153932\src\models\inference.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.03
Nodes (142): Return enriched analysis for a processed session (cached after /api/generate_pgn, Save manual labels for a specific image., Save manual labels for a specific image., Return structured benchmark results from the April 2026 ChessReD evaluation., Get existing labels for a specific image., Return structured benchmark results from the April 2026 ChessReD evaluation., Compute per-square accuracy for a session that has ground-truth labels., Auto-label all frames using the classifier's per-square softmax.      Frames w (+134 more)

### Community 1 - "Community 1"
Cohesion: 0.04
Nodes (60): argmax_boards_accuracy(), build_gt_index(), count_correct(), crop_squares(), evaluate_game(), gt_moves_from_fens(), gt_to_fen(), main() (+52 more)

### Community 2 - "Community 2"
Cohesion: 0.04
Nodes (55): collect_logits(), expected_calibration_error(), fit_temperature(), main(), ECE: weighted avg of |confidence - accuracy| across confidence bins., ChessPieceCNN, CVCHESS-inspired residual CNN for chess piece classification.  13 classes: emp, Two-conv residual block with optional 1×1 skip projection and max-pool. (+47 more)

### Community 3 - "Community 3"
Cohesion: 0.03
Nodes (73): List available sessions with image counts for labeling., List warped images and existing labels for a session., List warped images and existing labels for a session., Get classifier predictions for a specific image (auto-fill for labeling)., Get classifier predictions for a specific image (auto-fill for labeling)., Get existing labels for a specific image., Compute per-square accuracy for a session that has ground-truth labels., Auto-label all frames using the classifier's per-square softmax.      Frames w (+65 more)

### Community 4 - "Community 4"
Cohesion: 0.05
Nodes (66): BaseModel, apply_rotation(), auto_calibrate_endpoint(), AutoCornersRequest, AutoLabelRequest, bgr_to_pil(), calibrate(), calibrate_from_points() (+58 more)

### Community 5 - "Community 5"
Cohesion: 0.06
Nodes (43): load_session_from_local(), load_session_from_s3(), main(), process_game_session(), End-to-end pipeline: process a captured game session into PGN.    S3 session f, Full pipeline: session images → PGN.      Parameters     ----------     game, Load a game session from a local directory.      Returns dict with 'game_info', Load a game session from S3.      Returns dict with 'game_info', 'images' (lis (+35 more)

### Community 6 - "Community 6"
Cohesion: 0.08
Nodes (15): _format_date(), generate_pgn(), Generate PGN (Portable Game Notation) from a list of SAN moves and game metadata, Build a PGN string from moves and game metadata.      Parameters     --------, Write PGN string to a file., Convert date to PGN format YYYY.MM.DD., Wrap movetext to approximately `width` characters per line., save_pgn() (+7 more)

### Community 7 - "Community 7"
Cohesion: 0.08
Nodes (35): build_gt_index(), compare_board(), crop_squares(), evaluate(), evaluate_move_detection(), _gt_moves_from_fens(), main(), Derive ground-truth SAN move list by replaying the FEN sequence. (+27 more)

### Community 8 - "Community 8"
Cohesion: 0.1
Nodes (30): clientToCanvas(), defaultCropBox(), drawCropBox(), drawGridLines(), drawOrientationGrid(), edgeMidpoints(), fetchState(), getCanvasScale() (+22 more)

### Community 9 - "Community 9"
Cohesion: 0.11
Nodes (16): _append_result(), _default_command(), _ensure_results_file(), _git_short_hash(), main(), _read_score(), _safe_tag(), main() (+8 more)

### Community 10 - "Community 10"
Cohesion: 0.1
Nodes (19): main(), Run the trained corner detector on a set of images.  Two input modes:   1. Chess, make_target_heatmap(), predict_corners(), Chess board corner detector.  Heatmap-based regression of 4 board corners (a8, h, Run inference on a single BGR image and return corners in image coords.      Arg, Sub-pixel-accurate argmax over the last 2 dims.      Args:         heatmaps: (B,, Build Gaussian target heatmaps from normalized (x, y) corners.      Args: (+11 more)

### Community 11 - "Community 11"
Cohesion: 0.13
Nodes (14): predictions_to_fen(), Convert 64-square classifier predictions into a FEN position string., Convert per-square predictions to a full FEN string.      Parameters     ----, _piece(), Tests for src/pipeline/fen_generator.py  Coverage targets:   - predictions_to_fe, Edge case: every square holds a white queen., Confidence values must not affect the position string., Rank with alternating piece-empty should not merge runs across pieces. (+6 more)

### Community 12 - "Community 12"
Cohesion: 0.15
Nodes (14): build_dataset(), _build_gt_index(), _count_hard_squares(), evaluate(), finetune(), _gt_to_labels(), HardPositionDataset, Full-board ChessReD images filtered to mid/end-game positions.      Each item: ( (+6 more)

### Community 13 - "Community 13"
Cohesion: 0.16
Nodes (15): process_game(), Processes an entire game folder, finding the '00' image first to determine rotat, auto_canny(), cluster_hough_lines(), determine_orientation(), find_and_complete_grid(), intersection(), order_points() (+7 more)

### Community 14 - "Community 14"
Cohesion: 0.33
Nodes (4): BaseHTTPRequestHandler, current_task(), Handler, save()

### Community 15 - "Community 15"
Cohesion: 0.31
Nodes (8): bbox_centroid_to_grid(), crop_patch(), prepare_dataset(), Prepare ChessReD dataset for patch-based classification (CVCHESS approach).  App, Crop a SQUARE_SIZE×SQUARE_SIZE patch at grid position (row, col)., Perspective-warp image to BOARD_SIZE×BOARD_SIZE using annotated corners.      co, Map a bbox [x, y, w, h] in original coords to a warped grid cell (row, col)., warp_board()

### Community 16 - "Community 16"
Cohesion: 0.33
Nodes (2): make_summary_table(), set_cell_bg()

### Community 17 - "Community 17"
Cohesion: 0.43
Nodes (6): crop_patch(), prepare_dataset(), Patch preparation V2 — labels each grid cell by ChessReD's canonical `chessboard, Map chessboard_position ('a8'..'h1') to warped grid cell (row, col).      Assume, square_to_grid(), warp_board()

### Community 18 - "Community 18"
Cohesion: 0.4
Nodes (4): HandDetectionResult, _point_in_polygon(), Ray-casting point-in-polygon test; treats boundary as inside enough for hand gat, Detect hands in a BGR frame.          Parameters         ----------

### Community 20 - "Community 20"
Cohesion: 0.67
Nodes (3): main(), Renumber session warped images so they form a contiguous sequence.  If a session, renumber()

### Community 21 - "Community 21"
Cohesion: 0.67
Nodes (1): Per-frame breakdown for game 62 — find which frames are bad.

### Community 22 - "Community 22"
Cohesion: 0.67
Nodes (1): Download the .task model file if not already present.

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (1): Debug: visualize warped board for game 62 frame 0 with manual corners.

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (2): gallery_get_session(), Return game_info + warped/raw image lists for a session.

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (2): get_eval_summary(), Return structured benchmark results from the April 2026 ChessReD evaluation.

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (2): labeling_list_sessions(), List available sessions with image counts for labeling.

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (2): labeling_save_labels(), Save manual labels for a specific image.

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (2): gallery_get_image(), Run the trained ML corner detector on the provided camera frame.      Returns

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (2): labeling_get_labels(), Get existing labels for a specific image.

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (2): labeling_list_images(), List warped images and existing labels for a session.

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (2): gallery_get_image_typed(), Stream a single image from warped/ or raw/ subfolder.

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (2): get_session_analysis(), Return enriched analysis for a processed session (cached after /api/generate_pgn

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (2): gallery_list_sessions(), Stream a single image — backward compat for old flat sessions or warped fallback

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Process one frame's softmax output and attempt to detect a move.          Para

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Reset to starting position.

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Return a modified softmax dict with change-mask gating and         inventory co

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Squash excess piece predictions on changed squares to enforce inventory.

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Parameters         ----------         prior_weight : float             Multip

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Process one frame's softmax output and attempt to detect a move.          Para

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Reset to starting position.

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Return a modified softmax dict with change-mask gating and         inventory co

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Squash excess piece predictions on changed squares to enforce inventory.

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Build a perfect predictions dict from the current board state.

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Simulate a 2-square classifier error on an otherwise-correct e4 move.

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Return list of FEN position strings by replaying a move sequence.

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Wraps the MediaPipe Hand Landmarker (Tasks API, mediapipe>=0.10) to detect

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Download the .task model file if not already present.

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Detect hands in a BGR frame.          Parameters         ----------

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Convert FEN position string to {square_name: piece_char} dict.      Empty squa

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Find squares that changed between two FEN positions.      Returns dict with:

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Find the legal move that transforms the board from prev to curr position.

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Count number of matching squares between two FEN positions (out of 64).

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Try to detect a legal move, with feedback-driven correction on failure.      I

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Score every legal move by its Bayesian posterior under the full     classifier

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): Run the Bayesian-prior detector over a sequence of per-frame softmaxes.      M

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): Process a sequence of FEN positions and detect all moves.      Parameters

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): Majority-vote across K frames' predictions for each of 64 squares.      For ea

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): Process a sequence of per-frame predictions and detect all moves,     with feed

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (1): Maintains running board state and applies temporal heuristics to resolve     un

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): Parameters         ----------         prior_weight : float             Multip

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): Process one frame's softmax output and attempt to detect a move.          Para

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (1): Reset to starting position.

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (1): Return a modified softmax dict with change-mask gating and         inventory co

### Community 68 - "Community 68"
Cohesion: 1.0
Nodes (1): Squash excess piece predictions on changed squares to enforce inventory.

## Knowledge Gaps
- **128 isolated node(s):** `check_coverage_gate.py — CI gate: fail if any src/ module is below MIN_COVERAGE`, `Debug: visualize warped board for game 62 frame 0 with manual corners.`, `Count squares that differ from the starting position (proxy for mid-game).`, `Return {image_id: {square: piece_char}} from annotations.`, `Convert GT map to a list of 64 category IDs in _FEN_SQUARES order.      Returns` (+123 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 16`** (7 nodes): `generate_report_doc.py`, `add_body()`, `add_bullet()`, `add_chart()`, `add_heading()`, `make_summary_table()`, `set_cell_bg()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (3 nodes): `crop()`, `Per-frame breakdown for game 62 — find which frames are bad.`, `debug_g62_per_frame.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (3 nodes): `._ensure_model()`, `.__init__()`, `Download the .task model file if not already present.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (2 nodes): `Debug: visualize warped board for game 62 frame 0 with manual corners.`, `debug_warp_g62.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (2 nodes): `gallery_get_session()`, `Return game_info + warped/raw image lists for a session.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (2 nodes): `get_eval_summary()`, `Return structured benchmark results from the April 2026 ChessReD evaluation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (2 nodes): `labeling_list_sessions()`, `List available sessions with image counts for labeling.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (2 nodes): `labeling_save_labels()`, `Save manual labels for a specific image.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (2 nodes): `gallery_get_image()`, `Run the trained ML corner detector on the provided camera frame.      Returns`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (2 nodes): `labeling_get_labels()`, `Get existing labels for a specific image.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (2 nodes): `labeling_list_images()`, `List warped images and existing labels for a session.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (2 nodes): `gallery_get_image_typed()`, `Stream a single image from warped/ or raw/ subfolder.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (2 nodes): `get_session_analysis()`, `Return enriched analysis for a processed session (cached after /api/generate_pgn`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (2 nodes): `gallery_list_sessions()`, `Stream a single image — backward compat for old flat sessions or warped fallback`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Process one frame's softmax output and attempt to detect a move.          Para`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Reset to starting position.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Return a modified softmax dict with change-mask gating and         inventory co`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Squash excess piece predictions on changed squares to enforce inventory.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Parameters         ----------         prior_weight : float             Multip`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Process one frame's softmax output and attempt to detect a move.          Para`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Reset to starting position.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Return a modified softmax dict with change-mask gating and         inventory co`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Squash excess piece predictions on changed squares to enforce inventory.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Build a perfect predictions dict from the current board state.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Simulate a 2-square classifier error on an otherwise-correct e4 move.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Return list of FEN position strings by replaying a move sequence.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Wraps the MediaPipe Hand Landmarker (Tasks API, mediapipe>=0.10) to detect`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Download the .task model file if not already present.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Detect hands in a BGR frame.          Parameters         ----------`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Convert FEN position string to {square_name: piece_char} dict.      Empty squa`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Find squares that changed between two FEN positions.      Returns dict with:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Find the legal move that transforms the board from prev to curr position.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Count number of matching squares between two FEN positions (out of 64).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Try to detect a legal move, with feedback-driven correction on failure.      I`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Score every legal move by its Bayesian posterior under the full     classifier`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `Run the Bayesian-prior detector over a sequence of per-frame softmaxes.      M`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `Process a sequence of FEN positions and detect all moves.      Parameters`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `Majority-vote across K frames' predictions for each of 64 squares.      For ea`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `Process a sequence of per-frame predictions and detect all moves,     with feed`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `Maintains running board state and applies temporal heuristics to resolve     un`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `Parameters         ----------         prior_weight : float             Multip`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `Process one frame's softmax output and attempt to detect a move.          Para`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `Reset to starting position.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `Return a modified softmax dict with change-mask gating and         inventory co`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (1 nodes): `Squash excess piece predictions on changed squares to enforce inventory.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ChessPieceClassifier` connect `Community 0` to `Community 32`, `Community 33`, `Community 2`, `Community 1`, `Community 4`, `Community 5`, `Community 3`, `Community 7`, `Community 9`, `Community 21`, `Community 24`, `Community 25`, `Community 26`, `Community 27`, `Community 28`, `Community 29`, `Community 30`, `Community 31`?**
  _High betweenness centrality (0.471) - this node is a cross-community bridge._
- **Why does `process_game_session()` connect `Community 5` to `Community 0`, `Community 1`, `Community 2`, `Community 6`, `Community 7`, `Community 11`?**
  _High betweenness centrality (0.202) - this node is a cross-community bridge._
- **Why does `HandDetector` connect `Community 0` to `Community 32`, `Community 33`, `Community 3`, `Community 4`, `Community 5`, `Community 7`, `Community 9`, `Community 18`, `Community 22`, `Community 24`, `Community 25`, `Community 26`, `Community 27`, `Community 28`, `Community 29`, `Community 30`, `Community 31`?**
  _High betweenness centrality (0.097) - this node is a cross-community bridge._
- **Are the 288 inferred relationships involving `ChessPieceClassifier` (e.g. with `CaptureState` and `SessionState`) actually correct?**
  _`ChessPieceClassifier` has 288 INFERRED edges - model-reasoned connections that need verification._
- **Are the 262 inferred relationships involving `HandDetector` (e.g. with `CaptureState` and `SessionState`) actually correct?**
  _`HandDetector` has 262 INFERRED edges - model-reasoned connections that need verification._
- **Are the 243 inferred relationships involving `CornerDetector` (e.g. with `CaptureState` and `SessionState`) actually correct?**
  _`CornerDetector` has 243 INFERRED edges - model-reasoned connections that need verification._
- **Are the 24 inferred relationships involving `generate_pgn()` (e.g. with `run_demo()` and `make_charts()`) actually correct?**
  _`generate_pgn()` has 24 INFERRED edges - model-reasoned connections that need verification._