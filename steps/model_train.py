# steps/model_train.py
import os
import json
import argparse
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from models.unet import UNet

def train_fn(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: str = "cuda",
             epochs: int = 10, lr: float = 1e-4, out_dir: str = "./artifacts") -> Dict[str, Any]:
    device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.to(device)
    
    # L2 Regularization added as requested previously
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # FIX: Removed 'verbose=True' which causes the crash in newer PyTorch versions
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # Using standard BCE for now (or DiceBCELoss if you updated it)
    criterion = nn.BCEWithLogitsLoss()

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

        # Step the scheduler based on validation loss
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
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