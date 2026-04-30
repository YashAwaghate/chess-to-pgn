"""
Inference wrapper for the chess piece classifier.

Loads a trained model and provides prediction methods for individual
square patches and full board images.

Two classifiers are available:
  - ChessPieceClassifier: patch-based, uses our custom ResidualCNN
  - PretrainedBoardClassifier: full-board ResNeXt-101 from the ChessReD pretrained checkpoint
    (832 outputs = 64 squares × 13 classes, trained for 145 epochs)

PretrainedBoardClassifier is adapted from:
  Masouris, A. & van Gemert, J. (2024). End-to-End Chess Recognition.
  VISIGRAPP 2024. https://arxiv.org/abs/2310.04086
  GitHub: https://github.com/tmasouris/end-to-end-chess-recognition
  (train.py — ChessResNeXt class; dataset.py — normalization stats and label ordering)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as tv_models

from src.models.classifier import ChessPieceCNN, CLASS_NAMES


# ChessReD category_id → FEN piece character
# Source: annotations.json categories field
_CHESSRED_ID_TO_PIECE = {
    0: 'P',   # white-pawn
    1: 'R',   # white-rook
    2: 'N',   # white-knight
    3: 'B',   # white-bishop
    4: 'Q',   # white-queen
    5: 'K',   # white-king
    6: 'p',   # black-pawn
    7: 'r',   # black-rook
    8: 'n',   # black-knight
    9: 'b',   # black-bishop
    10: 'q',  # black-queen
    11: 'k',  # black-king
    12: 'empty',
}

# 64 squares in FEN order: rows="87654321", cols="abcdefgh"
# array_pos = 8*rows.index(rank) + cols.index(file)  (from dataset.py)
_FEN_SQUARES = [f"{f}{r}" for r in "87654321" for f in "abcdefgh"]


class _ChessRedResNeXt(nn.Module):
    """ResNeXt-101 32×8d backbone + linear head matching the pretrained checkpoint.

    Mirrors ChessResNeXt from the original train.py exactly:
      layers = list(backbone.children())[:-1]   # all except fc
      self.feature_extractor = nn.Sequential(*layers)
      self.classifier = nn.Linear(2048, 64*13)

    State dict keys (params only — relu/maxpool/avgpool have none):
      feature_extractor.{0,1,4,5,6,7}.*  — conv1, bn1, layer1-4
      classifier.{weight,bias}
    """

    def __init__(self):
        super().__init__()
        base = tv_models.resnext101_32x8d(weights=None)
        layers = list(base.children())[:-1]  # all except fc; includes avgpool at index 8
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(base.fc.in_features, 64 * 13)

    def forward(self, x):
        x = self.feature_extractor(x).flatten(1)
        return self.classifier(x)


class PretrainedBoardClassifier:
    """Chess board classifier using the ChessReD pretrained ResNeXt-101 32×8d.

    Takes a raw camera image (not warped) and returns per-square predictions
    with the same interface as ChessPieceClassifier.predict_board().

    The model outputs 832 logits (64 squares × 13 classes) from a single
    forward pass, giving it global context unavailable to patch-based models.
    Trained for 145 epochs on 10,800 real chess game photos.

    Input expectations (matching original training in dataset.py / train.py):
      - Raw camera image (BGR), any size — resized to 1024×1024 internally
      - Normalized with ChessReD dataset stats (not ImageNet)
    """

    def __init__(self, checkpoint_path='src/models/pretrained/checkpoint.ckpt', device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt['state_dict']

        self.model = _ChessRedResNeXt().to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Transform matching dataset.py exactly:
        #   read_image() → .float() [0-255 tensor] → Resize → ToPILImage → ToTensor [0-1] → Normalize
        self.transform = transforms.Compose([
            transforms.Resize(1024, antialias=None),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.47225544, 0.51124555, 0.55296206],
                                 std=[0.27787283, 0.27054584, 0.27802786]),
        ])

    def predict_full_board(self, raw_img: np.ndarray) -> dict:
        """Predict all 64 squares from a raw camera image.

        Parameters
        ----------
        raw_img : np.ndarray
            BGR image from the camera (any size). Should be the full frame
            showing the chessboard — the model handles end-to-end recognition
            without requiring perspective correction.

        Returns
        -------
        dict {square_name: (piece_class, confidence)}
        """
        probs_fen = self.predict_full_board_probs(raw_img)
        results = {}
        fen_order = ['empty', 'P', 'N', 'B', 'R', 'Q', 'K',
                     'p', 'n', 'b', 'r', 'q', 'k']
        for square, probs in probs_fen.items():
            idx = int(np.argmax(probs))
            results[square] = (fen_order[idx], float(probs[idx]))
        return results

    def predict_full_board_probs(self, raw_img: np.ndarray) -> dict:
        """Return full 13-class softmax per square.

        Output class order matches ChessPieceClassifier.fen_class_order:
        ['empty','P','N','B','R','Q','K','p','n','b','r','q','k']
        — i.e. empty first, then white pieces, then black.
        """
        import cv2
        rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float()
        tensor = self.transform(t).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor).view(1, 64, 13)
            probs = torch.softmax(logits, dim=2).squeeze(0).cpu().numpy()  # (64, 13)

        # The checkpoint's internal ordering is ChessReD's _CHESSRED_ID_TO_PIECE:
        #   0..5 = P,R,N,B,Q,K (white); 6..11 = p,r,n,b,q,k (black); 12 = empty
        # Remap to fen_class_order.
        fen_order = ['empty', 'P', 'N', 'B', 'R', 'Q', 'K',
                     'p', 'n', 'b', 'r', 'q', 'k']
        chessred_to_fen = {0:'P',1:'R',2:'N',3:'B',4:'Q',5:'K',
                           6:'p',7:'r',8:'n',9:'b',10:'q',11:'k',12:'empty'}
        remap = [fen_order.index(chessred_to_fen[i]) for i in range(13)]

        results = {}
        for i, square in enumerate(_FEN_SQUARES):
            row = np.zeros(13, dtype=np.float32)
            for src, dst in enumerate(remap):
                row[dst] = probs[i, src]
            results[square] = row
        return results

    def predict_board(self, patches: dict) -> dict:
        """Reconstruct a board image from patches and run full-board inference.

        Accepts the same interface as ChessPieceClassifier.predict_board()
        so it can be used as a drop-in replacement. Patches are stitched back
        into a single board image before inference.

        Parameters
        ----------
        patches : dict {square_name: numpy_patch (BGR, any size)}

        Returns
        -------
        dict {square_name: (piece_class, confidence)}
        """
        import cv2
        sq_size = 128
        board_img = np.zeros((8 * sq_size, 8 * sq_size, 3), dtype=np.uint8)
        for row_i, rank in enumerate("87654321"):
            for col_i, file in enumerate("abcdefgh"):
                square = f"{file}{rank}"
                if square in patches:
                    patch = cv2.resize(patches[square], (sq_size, sq_size))
                    r0, r1 = row_i * sq_size, (row_i + 1) * sq_size
                    c0, c1 = col_i * sq_size, (col_i + 1) * sq_size
                    board_img[r0:r1, c0:c1] = patch
        return self.predict_full_board(board_img)


# Folder name → FEN character mapping.
# On Windows, ImageFolder folder names use trailing underscores for black pieces
# (e.g. 'b_' instead of 'b') to avoid case-insensitive filesystem collisions.
# This table normalises both conventions so predictions are always valid FEN chars.
_FOLDER_TO_FEN = {
    'P': 'P', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K',
    'p': 'p', 'n': 'n', 'b': 'b', 'r': 'r', 'q': 'q', 'k': 'k',
    'p_': 'p', 'n_': 'n', 'b_': 'b', 'r_': 'r', 'q_': 'q', 'k_': 'k',
    'empty': 'empty',
}


class ChessPieceClassifier:
    """Wraps the trained ChessPieceCNN for inference.

    Supports three inference modes:
      - plain:  one forward pass per square, top-1 prediction (baseline)
      - TTA:    averages softmax over `tta_views` augmented crops per square
      - temperature-scaled: divides logits by `temperature` before softmax
                            to produce calibrated confidences

    Also exposes `predict_board_full_probs()` which returns the full
    13-class softmax vector per square, required by the move-history
    Bayesian prior in move_detector.detect_move_with_prior().
    """

    def __init__(self, model_path='models/chess_piece_classifier_v2.pth', device=None,
                 temperature: float = 1.0, tta_views: int = 1):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.temperature = float(temperature)
        self.tta_views = int(tta_views)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        # Reconstruct class name ordering from checkpoint
        class_to_idx = checkpoint.get('class_to_idx', None)
        num_classes = checkpoint.get('num_classes', 13)

        if class_to_idx:
            # ImageFolder sorts alphabetically; build idx→folder_name map
            # then translate folder names to FEN characters via _FOLDER_TO_FEN
            self.idx_to_class = {
                v: _FOLDER_TO_FEN.get(k, k)
                for k, v in class_to_idx.items()
            }
        else:
            # Fallback: assume CLASS_NAMES ordering
            self.idx_to_class = {i: name for i, name in enumerate(CLASS_NAMES)}

        # Canonical FEN piece ordering used by the Bayesian prior
        self.fen_class_order = ['empty', 'P', 'N', 'B', 'R', 'Q', 'K',
                                'p', 'n', 'b', 'r', 'q', 'k']
        # Map model's output index → position in fen_class_order
        self._idx_to_fen_pos = [
            self.fen_class_order.index(self.idx_to_class[i])
            for i in range(num_classes)
        ]

        self.model = ChessPieceCNN(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # ------------------------------------------------------------------
    # TTA helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _tta_variants(rgb_50: np.ndarray, n_views: int) -> list:
        """Produce n_views mild augmentations of a 50×50 RGB crop.

        Chess pieces are orientation-sensitive (a knight at 180° ≠ knight),
        so we only use photometric / translation jitter — NOT rotations.
        """
        import cv2
        out = [rgb_50]
        if n_views <= 1:
            return out
        h, w = rgb_50.shape[:2]
        # shift +/- 2 pixels
        shifts = [(2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (-2, -2)]
        for dx, dy in shifts[: n_views - 1]:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(rgb_50, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            out.append(shifted)
        # brightness nudges
        if n_views > len(out):
            for scale in (0.92, 1.08):
                if n_views <= len(out):
                    break
                out.append(np.clip(rgb_50.astype(np.float32) * scale, 0, 255).astype(np.uint8))
        return out[:n_views]

    def predict_square(self, patch: np.ndarray) -> tuple:
        """Predict piece class for a single 50×50 BGR patch.

        Returns (class_name, confidence).
        """
        import cv2
        # BGR → RGB, resize to 50×50
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (50, 50))

        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(1)

        class_name = self.idx_to_class[idx.item()]
        return class_name, conf.item()

    def predict_board(self, patches: dict) -> dict:
        """Batch-predict all 64 squares.

        Respects self.tta_views (averages softmax over augmentations) and
        self.temperature (divides logits before softmax).

        Parameters
        ----------
        patches : dict {square_name: numpy_patch (BGR)}

        Returns
        -------
        dict {square_name: (class_name, confidence)}
        """
        probs_per_sq = self.predict_board_full_probs(patches)
        results = {}
        for sq, probs in probs_per_sq.items():
            idx = int(np.argmax(probs))
            results[sq] = (self.fen_class_order[idx], float(probs[idx]))
        return results

    def predict_board_full_probs(self, patches: dict) -> dict:
        """Return the full 13-class softmax vector for every square.

        Needed by the Bayesian move-history prior, which multiplies
        classifier likelihoods by a legal-move prior before arg-maxing.

        Output class order is always self.fen_class_order, i.e.
        ['empty','P','N','B','R','Q','K','p','n','b','r','q','k'],
        regardless of the checkpoint's internal folder ordering.

        Returns
        -------
        dict {square_name: np.ndarray(shape=(13,), dtype=float32)}
        """
        import cv2

        square_names = sorted(patches.keys())
        # Build a tensor of (64 × tta_views, 3, 50, 50)
        views_per_sq = max(1, self.tta_views)
        all_tensors = []
        for sq in square_names:
            patch = patches[sq]
            rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (50, 50))
            for variant in self._tta_variants(rgb, views_per_sq):
                all_tensors.append(self.transform(variant))

        batch_tensor = torch.stack(all_tensors).to(self.device)
        with torch.no_grad():
            logits = self.model(batch_tensor) / self.temperature
            probs_model = torch.softmax(logits, dim=1).cpu().numpy()

        # Average softmax across views, then re-order columns to fen_class_order
        num_sq = len(square_names)
        probs_model = probs_model.reshape(num_sq, views_per_sq, -1).mean(axis=1)

        probs_fen = np.zeros((num_sq, 13), dtype=np.float32)
        for src_idx, dst_idx in enumerate(self._idx_to_fen_pos):
            probs_fen[:, dst_idx] = probs_model[:, src_idx]

        return {sq: probs_fen[i] for i, sq in enumerate(square_names)}
