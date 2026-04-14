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
        import cv2
        rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        # Match dataset.py: make a float [0-255] CHW tensor before applying transform
        t = torch.from_numpy(rgb).permute(2, 0, 1).float()  # [3, H, W], values 0-255
        tensor = self.transform(t).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)            # (1, 832)
            logits = logits.view(1, 64, 13)        # (1, 64, 13)
            probs = torch.softmax(logits, dim=2)   # (1, 64, 13)
            confs, idxs = probs.max(dim=2)         # (1, 64) each

        results = {}
        for i, square in enumerate(_FEN_SQUARES):
            class_id = idxs[0, i].item()
            conf = confs[0, i].item()
            results[square] = (_CHESSRED_ID_TO_PIECE[class_id], conf)
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
    """Wraps the trained ChessPieceCNN for inference."""

    def __init__(self, model_path='models/chess_piece_classifier.pth', device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

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

        self.model = ChessPieceCNN(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

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

        Parameters
        ----------
        patches : dict {square_name: numpy_patch (BGR)}

        Returns
        -------
        dict {square_name: (class_name, confidence)}
        """
        import cv2

        square_names = sorted(patches.keys())
        batch = []
        for sq in square_names:
            patch = patches[sq]
            rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (50, 50))
            batch.append(self.transform(rgb))

        batch_tensor = torch.stack(batch).to(self.device)

        with torch.no_grad():
            logits = self.model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
            confs, idxs = probs.max(1)

        results = {}
        for i, sq in enumerate(square_names):
            class_name = self.idx_to_class[idxs[i].item()]
            results[sq] = (class_name, confs[i].item())

        return results
