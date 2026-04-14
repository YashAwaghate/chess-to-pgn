"""
Training script for the chess piece classifier.

Usage:
  python -m src.models.train --data_dir data/chessred_patches --epochs 30 --batch_size 64
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.models.classifier import ChessPieceCNN, CLASS_NAMES


def get_transforms(train=True):
    """Return image transforms for training or validation."""
    if train:
        return transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc='  Train', leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc='  Val', leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Datasets
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')

    if not os.path.isdir(train_dir):
        print(f"ERROR: Training directory not found: {train_dir}")
        print("Run prepare_chessred.py first to create the dataset.")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(train=True))
    val_dataset = datasets.ImageFolder(val_dir, transform=get_transforms(train=False))

    # Verify class ordering matches our CLASS_NAMES
    folder_classes = train_dataset.classes
    print(f"Found classes: {folder_classes}")
    print(f"Expected:      {CLASS_NAMES}")

    # Build class name mapping (ImageFolder sorts alphabetically)
    # We need to ensure predictions map correctly
    class_to_idx = train_dataset.class_to_idx
    print(f"Class to index mapping: {class_to_idx}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    num_classes = len(folder_classes)
    model = ChessPieceCNN(num_classes=num_classes).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}  (lr={scheduler.get_last_lr()[0]:.6f})")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train — loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"  Val   — loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, 'chess_piece_classifier.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_to_idx': class_to_idx,
                'num_classes': num_classes,
            }, save_path)
            print(f"  ** New best model saved (val_acc={val_acc:.4f}) -> {save_path}")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train chess piece classifier')
    parser.add_argument('--data_dir', default='data/chessred_patches', help='Dataset directory')
    parser.add_argument('--output_dir', default='models', help='Where to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    main(args)
