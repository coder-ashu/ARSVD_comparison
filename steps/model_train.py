# steps/model_train.py
import os
import json
import argparse
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

from models.unet import UNet
from models.losses import CombinedBCEDiceLoss, DiceLoss, DiceFocalLoss


def finetune_fn(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                device: str = "cuda", epochs: int = 3, lr: float = 1e-5,
                out_dir: str = "./artifacts", model_name: str = "finetuned") -> Dict[str, Any]:
    """
    Fine-tune a pre-trained (or compressed) model with a very low learning rate.

    Args:
        model: Pre-trained model to fine-tune
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs: Number of fine-tuning epochs (typically 2-5)
        lr: Learning rate for fine-tuning (typically 1e-5, much lower than initial training)
        out_dir: Directory to save checkpoints
        model_name: Name for saving checkpoint

    Returns:
        Dict with checkpoint path and training history
    """
    device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.to(device)

    # Use much lower learning rate for fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # More aggressive LR reduction for fine-tuning
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)

    # BCE loss for binary segmentation
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": []}

    best_val_loss = float('inf')
    os.makedirs(out_dir, exist_ok=True)
    best_ckpt_path = os.path.join(out_dir, f"{model_name}_best.pth")

    print(f"\n{'='*60}")
    print(f"Fine-tuning model: {model_name}")
    print(f"Learning rate: {lr}, Epochs: {epochs}")
    print(f"{'='*60}\n")

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Fine-tune epoch {ep+1}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()

            # Gradient clipping for stability during fine-tuning
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
                vloss = criterion(out, yb)
                val_running += float(vloss) * xb.size(0)

        val_loss = val_running / len(val_loader.dataset)

        # Step scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        if new_lr != old_lr:
            print(f"Fine-tune Epoch {ep+1}: reducing learning rate to {new_lr:.6f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Fine-tune Epoch {ep+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.4f} -> {val_loss:.4f}). Saving fine-tuned model...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt_path)

    # Save history
    history_path = os.path.join(out_dir, f"{model_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)

    print(f"\nFine-tuning complete! Best val_loss: {best_val_loss:.4f}")
    print(f"Saved checkpoint to: {best_ckpt_path}\n")

    return {"ckpt": best_ckpt_path, "history": history}

def train_fn(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: str = "cuda",
             epochs: int = 10, lr: float = 1e-4, out_dir: str = "./artifacts",
             loss_type: str = "combined", base_filters: int = 64,
             use_cosine_lr: bool = False) -> Dict[str, Any]:
    """
    Training function with support for multiple loss functions and architectures.

    Args:
        model: U-Net model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs: Number of training epochs
        lr: Learning rate
        out_dir: Directory to save checkpoints
        loss_type: Type of loss - 'bce', 'dice', 'combined', 'dice_focal' (default: 'combined')
        base_filters: Number of base filters in U-Net (64 or 128)
        use_cosine_lr: Use CosineAnnealingLR instead of ReduceLROnPlateau

    Returns:
        Dict with checkpoint path and training history
    """
    device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.to(device)

    # L2 Regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Learning rate scheduler
    if use_cosine_lr:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    else:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # Select loss function
    if loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
        print(f"Using BCE Loss")
    elif loss_type == "dice":
        criterion = DiceLoss(smooth=1.0)
        print(f"Using Dice Loss")
    elif loss_type == "combined":
        criterion = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5, smooth=1.0)
        print(f"Using Combined BCE+Dice Loss (0.5 * BCE + 0.5 * Dice)")
    elif loss_type == "dice_focal":
        criterion = DiceFocalLoss(dice_weight=0.5, focal_weight=0.5)
        print(f"Using Dice+Focal Loss")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'bce', 'dice', 'combined', or 'dice_focal'")

    history = {"train_loss": [], "val_loss": []}
    
    best_val_loss = float('inf')
    os.makedirs(out_dir, exist_ok=True)
    best_ckpt_path = os.path.join(out_dir, "best_model.pth")

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Train epoch {ep+1}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss) * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        
        # validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                out = model(xb)
                vloss = criterion(out, yb)
                val_running += float(vloss) * xb.size(0)
        val_loss = val_running / len(val_loader.dataset)

        # Step the scheduler based on validation loss or epoch
        old_lr = optimizer.param_groups[0]['lr']
        if use_cosine_lr:
            scheduler.step()  # CosineAnnealingLR is epoch-based
        else:
            scheduler.step(val_loss)  # ReduceLROnPlateau is metric-based
        new_lr = optimizer.param_groups[0]['lr']
        
        # Manual verbose printing since the argument was removed
        if new_lr != old_lr:
            print(f"Epoch {ep+1}: reducing learning rate to {new_lr:.6f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        print(f"Epoch {ep+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.4f} -> {val_loss:.4f}). Saving model...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt_path)

    with open(os.path.join(out_dir, "train_history.json"), "w") as f:
        json.dump(history, f)
        
    return {"ckpt": best_ckpt_path, "history": history}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_ckpt_out", type=str, default="./artifacts")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--data_root", type=str, required=True)
    args = p.parse_args()

    from steps.ingest_data import run_ingest
    train_loader, val_loader, _ = run_ingest(args.data_root, batch_size=args.batch_size, out_dir=args.train_ckpt_out)

    model = UNet(n_channels=3, n_classes=1, base_filters=64)
    result = train_fn(model, train_loader, val_loader, device=args.device, epochs=args.epochs, lr=args.lr, out_dir=args.train_ckpt_out)
    print("Saved checkpoint to:", result["ckpt"])