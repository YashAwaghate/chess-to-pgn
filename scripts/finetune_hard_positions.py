#!/usr/bin/env python3
"""
Fine-tune the pretrained ChessReD ResNeXt-101 on hard (mid-game) positions.

"Hard" positions are images where the board has diverged significantly from
the starting position (many pieces moved/captured). The pretrained model
struggles most on these frames, showing 85-89% square accuracy vs 98-100%
on opening positions.

Strategy:
  Phase 1 — freeze backbone, train only the linear classifier head
  Phase 2 — unfreeze layer4 (last ResNeXt block) for deeper fine-tuning

The fine-tuned checkpoint is saved in the same format as the original, so
PretrainedBoardClassifier loads it without any code changes.

Usage:
    python scripts/finetune_hard_positions.py
    python scripts/finetune_hard_positions.py --hard_threshold 15 --epochs 15

Then use with the demo:
    python scripts/demo_chessred.py --checkpoint src/models/pretrained/checkpoint_finetuned.ckpt
"""

import argparse
import json
import os
import sys

# Force unbuffered output so progress is visible in background runs
sys.stdout.reconfigure(line_buffering=True)

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.inference import _ChessRedResNeXt

# Square order matching the model's output (64 squares × 13 classes)
_FEN_SQUARES = [f"{f}{r}" for r in "87654321" for f in "abcdefgh"]

# ChessReD category_id → FEN piece char (12 = empty)
_CAT_TO_PIECE = {
    0: 'P', 1: 'R', 2: 'N', 3: 'B', 4: 'Q', 5: 'K',
    6: 'p', 7: 'r', 8: 'n', 9: 'b', 10: 'q', 11: 'k',
}

# Starting position piece map: {square: piece_char}
_START_MAP = {
    'a1': 'R', 'b1': 'N', 'c1': 'B', 'd1': 'Q', 'e1': 'K', 'f1': 'B', 'g1': 'N', 'h1': 'R',
    'a2': 'P', 'b2': 'P', 'c2': 'P', 'd2': 'P', 'e2': 'P', 'f2': 'P', 'g2': 'P', 'h2': 'P',
    'a7': 'p', 'b7': 'p', 'c7': 'p', 'd7': 'p', 'e7': 'p', 'f7': 'p', 'g7': 'p', 'h7': 'p',
    'a8': 'r', 'b8': 'n', 'c8': 'b', 'd8': 'q', 'e8': 'k', 'f8': 'b', 'g8': 'n', 'h8': 'r',
}

# Same normalization as PretrainedBoardClassifier (Resize already done in __getitem__)
_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.47225544, 0.51124555, 0.55296206],
                         std=[0.27787283, 0.27054584, 0.27802786]),
])


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _count_hard_squares(gt_square_map: dict) -> int:
    """Count squares that differ from the starting position (proxy for mid-game)."""
    changed = 0
    for sq in _FEN_SQUARES:
        gt = gt_square_map.get(sq)       # None = empty in GT
        start = _START_MAP.get(sq)       # None = empty at start
        if gt != start:
            changed += 1
    return changed


def _build_gt_index(annotations: dict) -> dict:
    """Return {image_id: {square: piece_char}} from annotations."""
    index = {}
    for ann in annotations['annotations']['pieces']:
        img_id = ann['image_id']
        piece = _CAT_TO_PIECE.get(ann['category_id'])
        if piece is None:
            continue
        index.setdefault(img_id, {})[ann['chessboard_position']] = piece
    return index


def _gt_to_labels(gt_square_map: dict) -> list:
    """Convert GT map to a list of 64 category IDs in _FEN_SQUARES order.

    Returns a list of ints 0-12 (12 = empty).
    """
    piece_to_cat = {v: k for k, v in _CAT_TO_PIECE.items()}
    labels = []
    for sq in _FEN_SQUARES:
        piece = gt_square_map.get(sq)
        labels.append(piece_to_cat[piece] if piece else 12)
    return labels


class HardPositionDataset(Dataset):
    """Full-board ChessReD images filtered to mid/end-game positions.

    Each item: (tensor [3,1024,1024], labels [64] int64).
    """

    def __init__(self, samples: list):
        """
        Parameters
        ----------
        samples : list of {'img_path': str, 'gt_labels': list[int]}
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # Load at 1/4 resolution (768×768) directly from JPEG — avoids
        # allocating the full 3072×3072 ~27MB buffer in system RAM
        bgr = cv2.imread(s['img_path'], cv2.IMREAD_REDUCED_COLOR_4)
        if bgr is None:
            tensor = torch.zeros(3, 1024, 1024)
        else:
            bgr = cv2.resize(bgr, (1024, 1024), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float()
            tensor = _TRANSFORM(t)

        labels = torch.tensor(s['gt_labels'], dtype=torch.long)
        return tensor, labels


def build_dataset(
    annotations_path: str,
    images_base: str,
    split: str,
    hard_threshold: int,
    val_fraction: float,
) -> tuple:
    """Build train/val splits of hard-position samples.

    Returns (train_samples, val_samples) — lists of dicts for HardPositionDataset.
    """
    with open(annotations_path) as f:
        annotations = json.load(f)

    split_ids = set(annotations['splits'][split]['image_ids'])
    id_to_img = {img['id']: img for img in annotations['images']}
    gt_index = _build_gt_index(annotations)

    hard_samples = []
    easy_count = 0

    for img_id in split_ids:
        gt_sq = gt_index.get(img_id, {})
        difficulty = _count_hard_squares(gt_sq)

        if difficulty < hard_threshold:
            easy_count += 1
            continue

        img_meta = id_to_img[img_id]
        # Resolve path: annotations store 'images/0/G000_IMG000.jpg'
        img_path = os.path.join(
            images_base,
            str(img_meta['game_id']),
            img_meta['file_name'],
        )
        if not os.path.exists(img_path):
            continue

        gt_labels = _gt_to_labels(gt_sq)
        hard_samples.append({'img_path': img_path, 'gt_labels': gt_labels})

    # Shuffle deterministically
    import random
    rng = random.Random(42)
    rng.shuffle(hard_samples)

    n_val = max(1, int(len(hard_samples) * val_fraction))
    val_samples = hard_samples[:n_val]
    train_samples = hard_samples[n_val:]

    return train_samples, val_samples, easy_count


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple:
    """Return (avg_loss, per_square_accuracy)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_squares = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images).view(-1, 64, 13)          # (B, 64, 13)
            loss = criterion(logits.view(-1, 13), labels.view(-1))
            total_loss += loss.item() * images.size(0)

            preds = logits.argmax(dim=2)                     # (B, 64)
            total_correct += (preds == labels).sum().item()
            total_squares += labels.numel()

    return total_loss / len(loader.dataset), total_correct / total_squares


def finetune(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build dataset
    print(f"\nBuilding dataset (split={args.split}, hard_threshold={args.hard_threshold}) ...")
    train_samples, val_samples, easy_count = build_dataset(
        annotations_path='data/annotations.json',
        images_base='data/ChessRed_images',
        split=args.split,
        hard_threshold=args.hard_threshold,
        val_fraction=args.val_fraction,
    )
    print(f"  Hard positions (>={args.hard_threshold} sq changed): {len(train_samples) + len(val_samples)}")
    print(f"  Easy positions (skipped):                          {easy_count}")
    print(f"  Train: {len(train_samples)}  |  Val: {len(val_samples)}")

    if len(train_samples) == 0:
        print(f"\nNo hard positions found with threshold={args.hard_threshold}. "
              f"Try lowering --hard_threshold.")
        return

    train_loader = DataLoader(
        HardPositionDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        HardPositionDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == 'cuda'),
    )

    # Load pretrained model
    print(f"\nLoading checkpoint from {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = _ChessRedResNeXt().to(device)
    model.load_state_dict(ckpt['state_dict'])

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):

        # Phase 1: freeze backbone, only train classifier head
        # Phase 2: unfreeze layer4 (feature_extractor[7]) after unfreeze_after epochs
        if epoch == 1:
            for param in model.feature_extractor.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
            print(f"\nPhase 1: classifier head only")
        elif epoch == args.unfreeze_after + 1:
            # Unfreeze layer4 (index 7 in feature_extractor)
            for param in model.feature_extractor[7].parameters():
                param.requires_grad = True
            print(f"\nPhase 2: classifier + layer4 (epoch {epoch})")

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=args.lr)

        # Train epoch
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_squares = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(images).view(-1, 64, 13)          # (B, 64, 13)
            loss = criterion(logits.view(-1, 13), labels.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=2)
            train_correct += (preds == labels).sum().item()
            train_squares += labels.numel()

            if (batch_idx + 1) % 10 == 0:
                running_acc = train_correct / train_squares
                print(f"  Epoch {epoch:2d} [{batch_idx+1:3d}/{len(train_loader)}]  "
                      f"loss={train_loss/train_squares:.4f}  acc={running_acc:.4f}", end='\r')

        train_acc = train_correct / train_squares
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"  Epoch {epoch:2d}  "
              f"train_loss={train_loss/train_squares:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}", end='')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'state_dict': model.state_dict()}, args.output)
            print(f"  [saved]")
        else:
            print()

    print(f"\nFine-tuning complete.")
    print(f"  Best val accuracy: {best_val_acc:.4f}")
    print(f"  Checkpoint saved:  {args.output}")
    print(f"\nRun demo with fine-tuned model:")
    print(f"  python scripts/demo_chessred.py --checkpoint {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune ChessReD model on hard positions')
    parser.add_argument('--checkpoint', default='src/models/pretrained/checkpoint.ckpt',
                        help='Input checkpoint path (default: src/models/pretrained/checkpoint.ckpt)')
    parser.add_argument('--output', default='src/models/pretrained/checkpoint_finetuned.ckpt',
                        help='Output checkpoint path (default: src/models/pretrained/checkpoint_finetuned.ckpt)')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'],
                        help='Dataset split to fine-tune on (default: val)')
    parser.add_argument('--hard_threshold', type=int, default=20,
                        help='Min changed squares to count as hard position (default: 20)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Total fine-tuning epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size — keep low for 1024x1024 images (default: 4)')
    parser.add_argument('--unfreeze_after', type=int, default=5,
                        help='Epoch after which layer4 is unfrozen (default: 5)')
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='Fraction of hard images held out for validation (default: 0.1)')
    parser.add_argument('--workers', type=int, default=0,
                        help='DataLoader worker processes (default: 0, single-process)')
    args = parser.parse_args()
    finetune(args)
