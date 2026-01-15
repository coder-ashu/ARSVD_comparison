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

def train_fn(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: str = "cuda",
             epochs: int = 10, lr: float = 1e-4, out_dir: str = "./artifacts",
             patience: int = 10, weight_decay: float = 5e-6) -> Dict[str, Any]:
    """
    Training function with:
    - Weight decay (L2 regularization)
    - Learning rate scheduling
    - Gradient clipping
    - Early stopping
    """
    device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.to(device)

    # Fix #2: Add weight decay for L2 regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Fix #7: Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": []}

    # Fix #4: Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Train epoch {ep+1}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()

            # Fix #8: Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += float(loss.detach()) * xb.size(0)  # Use .detach() to avoid warning

        train_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                out = model(xb)
                vloss = criterion(out, yb)
                val_running += float(vloss.detach()) * xb.size(0)  # Use .detach() to avoid warning
        val_loss = val_running / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {ep+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        # Fix #7: Step the learning rate scheduler
        scheduler.step(val_loss)

        # Fix #4: Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best model
            os.makedirs(out_dir, exist_ok=True)
            ckpt_path = os.path.join(out_dir, "baseline_trained.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Validation loss improved ({best_val_loss:.4f} -> {val_loss:.4f}). Saving model...")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {ep+1} (patience={patience})")
                break

    # Save training history
    with open(os.path.join(out_dir, "train_history.json"), "w") as f:
        json.dump(history, f)

    return {"ckpt": ckpt_path, "history": history}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_ckpt_out", type=str, default="./artifacts")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    p.add_argument("--weight_decay", type=float, default=1e-5, help="L2 regularization")
    args = p.parse_args()

    # quick ingest (small code reuse)
    from steps.ingest_data import run_ingest
    train_loader, val_loader, _ = run_ingest(args.data_root, batch_size=args.batch_size, out_dir=args.train_ckpt_out)

    model = UNet(n_channels=3, n_classes=1, base_filters=64, dropout_prob=0.05)
    result = train_fn(model, train_loader, val_loader, device=args.device, epochs=args.epochs,
                     lr=args.lr, out_dir=args.train_ckpt_out, patience=args.patience,
                     weight_decay=args.weight_decay)
    print("Saved checkpoint to:", result["ckpt"])
