# steps/model_train_dice.py
"""
Optimized training based on high-performance TensorFlow implementation.
Key changes:
1. Dice Loss instead of BCE
2. Simple normalization (/255.0)
3. No dropout
4. Longer training
"""
import os
import json
import argparse
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet import UNet


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""
    def __init__(self, smooth: float = 1e-15):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Flatten
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()

        dice = (2. * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice loss (even better than Dice alone)
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-15):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, preds, targets):
        return (self.bce_weight * self.bce(preds, targets) +
                self.dice_weight * self.dice(torch.sigmoid(preds), targets))


def train_fn_optimized(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                       device: str = "cuda", epochs: int = 100, lr: float = 1e-4,
                       out_dir: str = "./artifacts", patience: int = 20,
                       loss_type: str = "dice") -> Dict[str, Any]:
    """
    Optimized training function mirroring the TensorFlow implementation:
    - Dice Loss
    - Simple normalization (handled in dataset)
    - Longer training
    - Larger patience
    """
    device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.to(device)

    # Use Dice Loss or Combined Loss
    if loss_type == "dice":
        criterion = DiceLoss(smooth=1e-15)
        print("Using Dice Loss")
    elif loss_type == "combined":
        criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
        print("Using Combined BCE + Dice Loss")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Using BCE Loss (original)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler (more aggressive than before)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Train epoch {ep+1}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            out = model(xb)

            if loss_type in ["dice", "combined"]:
                # Dice/Combined loss expects sigmoid probabilities
                loss = criterion(torch.sigmoid(out), yb)
            else:
                # BCE expects logits
                loss = criterion(out, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.detach()) * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                out = model(xb)

                if loss_type in ["dice", "combined"]:
                    loss = criterion(torch.sigmoid(out), yb)
                else:
                    loss = criterion(out, yb)

                val_running += float(loss.detach()) * xb.size(0)

        val_loss = val_running / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {ep+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            os.makedirs(out_dir, exist_ok=True)
            ckpt_path = os.path.join(out_dir, "baseline_trained.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"✓ Validation loss improved to {val_loss:.4f}. Saving model...")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {ep+1}")
                break

    # Save training history
    with open(os.path.join(out_dir, "train_history.json"), "w") as f:
        json.dump(history, f)

    return {"ckpt": ckpt_path, "history": history}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--train_ckpt_out", type=str, default="./artifacts")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--loss_type", type=str, default="dice", choices=["bce", "dice", "combined"])
    args = p.parse_args()

    from steps.ingest_data import run_ingest
    train_loader, val_loader, _ = run_ingest(args.data_root, batch_size=args.batch_size, out_dir=args.train_ckpt_out)

    model = UNet(n_channels=3, n_classes=1, base_filters=64, dropout_prob=0.0)  # No dropout!
    result = train_fn_optimized(model, train_loader, val_loader, device=args.device,
                                epochs=args.epochs, lr=args.lr, out_dir=args.train_ckpt_out,
                                patience=args.patience, loss_type=args.loss_type)
    print("Saved checkpoint to:", result["ckpt"])
