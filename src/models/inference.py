"""
Inference wrapper for the chess piece classifier.

Loads a trained model and provides prediction methods for individual
square patches and full board images.
"""

import os
import numpy as np
import torch
from torchvision import transforms

from src.models.classifier import ChessPieceCNN, CLASS_NAMES


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
            # ImageFolder sorts alphabetically; build idx→class_name map
            self.idx_to_class = {v: k for k, v in class_to_idx.items()}
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
