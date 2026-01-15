# steps/model_train.py
import os
import json
import argparse
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet import UNet
from models.evaluation import SegmentationEvaluator


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation - CRITICAL for high IoU!
    Matches the TensorFlow implementation that achieves IoU 0.90
    """
    def __init__(self, smooth: float = 1e-15):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Flatten spatial dimensions
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()

        dice = (2. * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )

        return 1 - dice

def train_fn(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: str = "cuda",
             epochs: int = 10, lr: float = 1e-4, out_dir: str = "./artifacts",
             patience: int = 20, weight_decay: float = 0.0, use_dice_loss: bool = False) -> Dict[str, Any]:
    """
    Optimized training function matching high-performance TensorFlow implementation:
    - Dice Loss option (CRITICAL for IoU 0.90!)
    - No weight decay
    - Longer patience
    - Learning rate scheduling
    - Gradient clipping
    - Early stopping
    """
    device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.to(device)

    # Use Dice Loss if requested (CRITICAL!)
    if use_dice_loss:
        criterion = DiceLoss(smooth=1e-15)
        print("✓ Using DICE LOSS (matches IoU 0.90 implementation)")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Using BCE Loss")

    # CRITICAL FIX: Use higher learning rate for larger models (base_filters=96)
    # Larger models need higher LR to learn effectively
    effective_lr = lr
    total_params = sum(p.numel() for p in model.parameters())
    if total_params > 10e6:  # If model has >10M parameters
        effective_lr = lr * 1.5  # Increase LR by 50% for larger models
        print(f"Model has {total_params/1e6:.1f}M parameters. Using adjusted LR: {effective_lr:.6f}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr, weight_decay=weight_decay)

    # Learning rate scheduler - FIXED: Less aggressive to prevent premature LR reduction
    # Increased patience so LR doesn't drop too quickly when validation loss plateaus
    # Note: verbose parameter not available in all PyTorch versions, removed for compatibility
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5
    )

    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}

    # Early stopping variables
    best_val_loss = float('inf')
    best_val_dice = 0.0
    epochs_no_improve = 0
    evaluator = SegmentationEvaluator(threshold=0.5)

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Train epoch {ep+1}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            out = model(xb)

            # Apply loss
            if use_dice_loss:
                # Dice Loss expects probabilities, not logits
                loss = criterion(torch.sigmoid(out), yb)
            else:
                # BCE expects logits
                loss = criterion(out, yb)

            loss.backward()

            # Gradient clipping for stability - FIXED: Less aggressive clipping
            # Higher max_norm allows gradients to flow better, preventing learning stagnation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            running_loss += float(loss.detach()) * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                out = model(xb)

                if use_dice_loss:
                    vloss = criterion(torch.sigmoid(out), yb)
                else:
                    vloss = criterion(out, yb)

                val_running += float(vloss.detach()) * xb.size(0)

        val_loss = val_running / len(val_loader.dataset)
        
        # Calculate validation metrics (Dice, IoU) for monitoring
        val_metrics = evaluator.calculate_score(model, val_loader, device=device)
        val_dice = val_metrics["dice"]
        val_iou = val_metrics["iou"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        
        # Get current learning rate for monitoring
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {ep+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_dice={val_dice:.4f} val_iou={val_iou:.4f} lr={current_lr:.6f}")

        # Step the learning rate scheduler
        scheduler.step(val_loss)
        
        # FIXED: Warn if learning rate gets too low (might be causing stagnation)
        if current_lr < 1e-5:
            print(f"⚠️  WARNING: Learning rate is very low ({current_lr:.6f}). Model might have stopped learning.")

        # Early stopping logic - use Dice score for segmentation (better metric than loss)
        if use_dice_loss:
            # For Dice loss, higher Dice is better
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save best model
                os.makedirs(out_dir, exist_ok=True)
                ckpt_path = os.path.join(out_dir, "baseline_trained.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"✓ Validation Dice improved to {val_dice:.4f} (loss={val_loss:.4f}). Saving model...")
            else:
                epochs_no_improve += 1
                print(f"No Dice improvement for {epochs_no_improve} epoch(s) (best={best_val_dice:.4f})")
        else:
            # For BCE loss, lower loss is better
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_dice = val_dice
                epochs_no_improve = 0
                # Save best model
                os.makedirs(out_dir, exist_ok=True)
                ckpt_path = os.path.join(out_dir, "baseline_trained.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"✓ Validation loss improved to {val_loss:.4f} (dice={val_dice:.4f}). Saving model...")
            else:
                epochs_no_improve += 1
                print(f"No loss improvement for {epochs_no_improve} epoch(s) (best={best_val_loss:.4f})")
        
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {ep+1} (patience={patience})")
            break

    # Save training history
    with open(os.path.join(out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return {
        "ckpt": ckpt_path, 
        "history": history,
        "best_val_dice": best_val_dice,
        "best_val_iou": history["val_iou"][-1] if history["val_iou"] else 0.0
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_ckpt_out", type=str, default="./artifacts")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    p.add_argument("--weight_decay", type=float, default=0.0, help="L2 regularization")
    p.add_argument("--use_dice_loss", action="store_true", help="Use Dice Loss (recommended)")
    args = p.parse_args()

    # quick ingest (small code reuse)
    from steps.ingest_data import run_ingest
    train_loader, val_loader, _ = run_ingest(args.data_root, batch_size=args.batch_size, out_dir=args.train_ckpt_out)

    model = UNet(n_channels=3, n_classes=1, base_filters=64, dropout_prob=0.0)  # No dropout!
    result = train_fn(model, train_loader, val_loader, device=args.device, epochs=args.epochs,
                     lr=args.lr, out_dir=args.train_ckpt_out, patience=args.patience,
                     weight_decay=args.weight_decay, use_dice_loss=args.use_dice_loss)
    print("Saved checkpoint to:", result["ckpt"])
