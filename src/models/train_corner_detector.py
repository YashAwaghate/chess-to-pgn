"""Train the corner detector on ChessReD's 2078 frames with corner annotations.

Heavy augmentation (perspective + lighting + scale) to generalize to phone-camera
session captures.

Usage:
    python -m src.models.train_corner_detector --epochs 30 --batch_size 16
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.corner_detector import (
    CornerDetector, soft_argmax_2d, make_target_heatmap,
    INPUT_SIZE, HEATMAP_SIZE, CORNER_ORDER
)


def heatmap_cross_entropy(pred_logits, target_heatmaps):
    """Spatial CE over each corner heatmap.

    Unlike pixel MSE, every corner channel is treated as one probability
    distribution, so the background cannot dominate the objective.
    """
    bsz, channels, _, _ = pred_logits.shape
    target = target_heatmaps.reshape(bsz, channels, -1)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    log_probs = F.log_softmax(pred_logits.reshape(bsz, channels, -1), dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


def save_checkpoint(path, model, epoch, val_loss, val_px, args, best_val):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
        'val_px_err': val_px,
        'best_val_loss': best_val,
        'args': vars(args),
    }, path)


def make_logger(log_path):
    log_file = Path(log_path) if log_path else None
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(message):
        print(message, flush=True)
        if log_file:
            with log_file.open('a', encoding='utf-8') as f:
                f.write(message + '\n')

    return log


def save_validation_overlays(model, val_loader, out_dir, epoch, device, args, count=4):
    if not out_dir or count <= 0:
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    model.eval()
    saved = 0
    with torch.no_grad():
        for img, gt_norm in val_loader:
            batch = img.to(device, non_blocking=True)
            pred_hm = model(batch)
            pred = soft_argmax_2d(pred_hm, temperature=args.softmax_temperature).cpu().numpy()
            gt = gt_norm.numpy()
            img_np = img.permute(0, 2, 3, 1).numpy()

            for i in range(img_np.shape[0]):
                rgb = np.clip((img_np[i] * std + mean) * 255.0, 0, 255).astype(np.uint8)
                canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                for corner_idx, label in enumerate(CORNER_ORDER):
                    gx, gy = (gt[i, corner_idx] * INPUT_SIZE).astype(int)
                    px, py = (pred[i, corner_idx] * INPUT_SIZE).astype(int)
                    cv2.circle(canvas, (gx, gy), 6, (0, 220, 0), -1)
                    cv2.circle(canvas, (px, py), 6, (0, 0, 255), 2)
                    cv2.putText(canvas, label, (px + 8, py - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imwrite(str(out_path / f'epoch_{epoch:03d}_sample_{saved:02d}.jpg'), canvas)
                saved += 1
                if saved >= count:
                    return


class ChessReDCornerDataset(Dataset):
    def __init__(self, ann_path, images_base, image_ids, augment=False):
        with open(ann_path) as f:
            self.ann = json.load(f)
        self.id_to_img    = {im['id']: im for im in self.ann['images']}
        self.corner_index = {c['image_id']: c['corners']
                              for c in self.ann['annotations']['corners']}
        # Filter to ids that have corners
        self.image_ids = [iid for iid in image_ids if iid in self.corner_index]
        self.images_base = images_base
        self.augment = augment

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.image_ids)

    def _augment(self, img, corners_pix):
        """Apply random perspective + lighting jitter; return updated img + corners."""
        h, w = img.shape[:2]
        # Random perspective shift on each corner of the IMAGE (not the board) by
        # up to ±5% in each direction — simulates camera-angle changes.
        if random.random() < 0.7:
            jitter = 0.05
            src_pts = np.float32([[0,0],[w,0],[w,h],[0,h]])
            dst_pts = src_pts + np.random.uniform(-jitter, jitter, src_pts.shape) * np.array([w, h])
            M = cv2.getPerspectiveTransform(src_pts, dst_pts.astype(np.float32))
            img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            # Update corner positions
            pts = np.array([[c[0], c[1], 1.0] for c in corners_pix]).T   # 3×4
            warped_pts = M @ pts
            warped_pts /= warped_pts[2:3]
            corners_pix = [[float(warped_pts[0, i]), float(warped_pts[1, i])]
                           for i in range(4)]

        # Lighting / colour jitter
        if random.random() < 0.6:
            img = img.astype(np.float32)
            # brightness
            img += np.random.uniform(-25, 25)
            # contrast
            img = (img - 128) * np.random.uniform(0.85, 1.15) + 128
            # per-channel tint
            for c in range(3):
                img[..., c] *= np.random.uniform(0.9, 1.1)
            img = np.clip(img, 0, 255).astype(np.uint8)

        return img, corners_pix

    def __getitem__(self, idx):
        iid = self.image_ids[idx]
        meta = self.id_to_img[iid]
        path = os.path.join(self.images_base, str(meta['game_id']), meta['file_name'])
        bgr = cv2.imread(path)
        if bgr is None:
            raise RuntimeError(f"Could not read {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        c = self.corner_index[iid]
        # ChessReD order: top_left, top_right, bottom_right, bottom_left
        corners_pix = [c[k] for k in CORNER_ORDER]

        # Resize FIRST (before augmentation) to keep memory usage low
        sx = INPUT_SIZE / w
        sy = INPUT_SIZE / h
        rsz = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
        corners_resized = [[c[0]*sx, c[1]*sy] for c in corners_pix]

        if self.augment:
            rsz, corners_resized = self._augment(rsz, corners_resized)

        # Normalize to [0, 1]
        corners_norm = np.array(corners_resized, dtype=np.float32) / INPUT_SIZE
        corners_norm = np.clip(corners_norm, 0.0, 1.0)

        # Normalize image
        img = rsz.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img, torch.from_numpy(corners_norm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', default='data/annotations.json')
    parser.add_argument('--images_base', default='data/ChessRed_images')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out', default='models/corner_detector.pth')
    parser.add_argument('--last_out', default='models/corner_detector_last.pth')
    parser.add_argument('--log', default='poster_assets/corner_detector_train_v2.log')
    parser.add_argument('--viz_dir', default='poster_assets/corner_detector_val_overlays')
    parser.add_argument('--viz_every', type=int, default=1)
    parser.add_argument('--viz_count', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--split', choices=['game', 'image'], default='game',
                        help='Use game-level validation split to avoid adjacent-frame leakage.')
    parser.add_argument('--max_train', type=int, default=0,
                        help='Limit train images for smoke tests; 0 uses all.')
    parser.add_argument('--max_val', type=int, default=0,
                        help='Limit val images for smoke tests; 0 uses all.')
    parser.add_argument('--heatmap_sigma', type=float, default=2.0)
    parser.add_argument('--softmax_temperature', type=float, default=20.0)
    parser.add_argument('--coord_loss_weight', type=float, default=5.0)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        Path(args.log).write_text('', encoding='utf-8')
    log = make_logger(args.log)

    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device == 'auto':
        device = 'cpu'
    log(f"Device: {device}")

    with open(args.annotations) as f:
        ann = json.load(f)

    corner_ids = sorted({c['image_id'] for c in ann['annotations']['corners']})
    if args.split == 'game':
        id_to_img = {im['id']: im for im in ann['images']}
        corner_games = sorted({id_to_img[iid]['game_id'] for iid in corner_ids})
        random.shuffle(corner_games)
        n_val_games = max(1, len(corner_games) // 10)
        val_games = set(corner_games[:n_val_games])
        val_ids = [iid for iid in corner_ids if id_to_img[iid]['game_id'] in val_games]
        train_ids = [iid for iid in corner_ids if id_to_img[iid]['game_id'] not in val_games]
        log(f"Split: game-level  Train games: {len(corner_games) - n_val_games}  "
            f"Val games: {n_val_games}")
    else:
        random.shuffle(corner_ids)
        n_val = max(50, len(corner_ids) // 10)
        val_ids = corner_ids[:n_val]
        train_ids = corner_ids[n_val:]
        log("Split: image-level")
    if args.max_train:
        train_ids = train_ids[:args.max_train]
    if args.max_val:
        val_ids = val_ids[:args.max_val]
    log(f"Train: {len(train_ids)}  Val: {len(val_ids)}")

    train_ds = ChessReDCornerDataset(args.annotations, args.images_base, train_ids, augment=True)
    val_ds   = ChessReDCornerDataset(args.annotations, args.images_base, val_ids,   augment=False)

    pin_memory = device == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin_memory)

    model = CornerDetector(pretrained=args.pretrained).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_val = float('inf')
    for ep in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0; n_train = 0
        for img, gt_norm in train_loader:
            img = img.to(device, non_blocking=True)
            gt_norm = gt_norm.to(device, non_blocking=True)
            target_hm = make_target_heatmap(
                gt_norm, HEATMAP_SIZE, sigma=args.heatmap_sigma, normalize=True
            )
            pred_hm = model(img)

            loss_hm = heatmap_cross_entropy(pred_hm, target_hm)
            pred_coords = soft_argmax_2d(pred_hm, temperature=args.softmax_temperature)
            loss_xy = F.l1_loss(pred_coords, gt_norm)
            loss = loss_hm + args.coord_loss_weight * loss_xy

            optim.zero_grad(); loss.backward(); optim.step()
            train_loss += loss.item() * img.size(0)
            n_train += img.size(0)
        train_loss /= n_train

        # ---- Val ----
        model.eval()
        val_loss = 0.0; val_px = 0.0; n_val_b = 0
        with torch.no_grad():
            for img, gt_norm in val_loader:
                img = img.to(device); gt_norm = gt_norm.to(device)
                target_hm = make_target_heatmap(
                    gt_norm, HEATMAP_SIZE, sigma=args.heatmap_sigma, normalize=True
                )
                pred_hm = model(img)
                pred_coords = soft_argmax_2d(pred_hm, temperature=args.softmax_temperature)
                loss_hm = heatmap_cross_entropy(pred_hm, target_hm)
                loss_xy = F.l1_loss(pred_coords, gt_norm)
                val_loss += (loss_hm + args.coord_loss_weight * loss_xy).item() * img.size(0)
                # Pixel error in 512 space
                px_err = ((pred_coords - gt_norm) * INPUT_SIZE).abs().mean().item()
                val_px += px_err * img.size(0)
                n_val_b += img.size(0)
        val_loss /= n_val_b; val_px /= n_val_b

        scheduler.step()
        log(f"Epoch {ep:3d}/{args.epochs}  lr={optim.param_groups[0]['lr']:.2e}  "
            f"train={train_loss:.5f}  val={val_loss:.5f}  val_px_err={val_px:.2f}")

        save_checkpoint(args.last_out, model, ep, val_loss, val_px, args, best_val)
        if args.viz_every and ep % args.viz_every == 0:
            save_validation_overlays(
                model, val_loader, args.viz_dir, ep, device, args, count=args.viz_count
            )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(args.out, model, ep, val_loss, val_px, args, best_val)
            log(f"  ** saved (val={val_loss:.5f}, px_err={val_px:.2f}) -> {args.out}")

    log(f"\nBest val loss: {best_val:.5f}")


if __name__ == '__main__':
    main()
