# steps/model_train_regularized.py
import os
import json
import argparse
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --------------------------------------------------------------
# 1. Improved U-Net with Dropout + more BatchNorm
# --------------------------------------------------------------
from models.unet import UNet  # assuming your original UNet

class UNetWithDropout(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, base_filters=64, dropout=0.3):
        super().__init__()
        self.unet = UNet(
            n_channels=n_channels,
            n_classes=n_classes,
            base_filters=base_filters,
            # most modern U-Net implementations allow passing dropout
            # if your original UNet doesn't support it, use the version below
        )
        # If your original UNet does NOT have dropout argument, 
        # replace it with the small modified version I give at the bottom.
        
        self.dropout = dropout

    def forward(self, x):
        return self.unet(x)  # dropout is inside if you use the modified UNet


# --------------------------------------------------------------
# 2. Strong data augmentation (this alone often cuts overfitting in half)
# --------------------------------------------------------------
def get_training_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),                    # swap x/y
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
        A.GridDistortion(p=0.5),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),  # Cutout
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        # Keep image/ mask synchronized
    ], is_check_shapes=False)

# You need to modify your dataset to accept transform (see note below)

# --------------------------------------------------------------
# 3. Updated training function with all regularizations
# --------------------------------------------------------------
def train_fn_regularized(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cuda",
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,       # L2 regularization
    out_dir: str = "./artifacts",
    patience: int = 15                # for early stopping + LR scheduler
) -> Dict[str, Any]:

    device = torch.device("cpu" if not torch.cuda.is_available() else device)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, verbose=True)

    history = {"train_loss": [], "val_loss": [], "best_val": 1e9}
    best_epoch = -1
    epochs_no_improve = 0

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Train {ep+1}/{epochs}"):
            xb = xb.to(device)
            yb = yb.to(device).float().unsqueeze(1)   # ensure (B,1,H,W)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ---------- validation ----------
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).float().unsqueeze(1)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_running += loss.item() * xb.size(0)
        val_loss = val_running / len(val_loader.dataset)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {ep+1:3d} | train {train_loss:.5f} | val {val_loss:.5f} | lr {optimizer.param_groups[0]['lr']:.1e}")

        # ---------- early stopping & checkpoint ----------
        if val_loss < history["best_val"] - 1e-4:
            history["best_val"] = val_loss
            best_epoch = ep
            epochs_no_improve = 0
            ckpt_path = os.path.join(out_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {ep+1}")
            break

    # save final model too
    final_path = os.path.join(out_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f)

    print(f"Best model (val_loss={history['best_val']:.5f}) saved to {ckpt_path}")
    return {"best_ckpt": ckpt_path, "final_ckpt": final_path, "history": history}


# --------------------------------------------------------------
# If your original UNet does NOT support dropout, use this tiny patch:
# --------------------------------------------------------------
class UNetWithDropoutFixed(UNet):
    def __init__(self, n_channels=3, n_classes=1, base_filters=64, dropout=0.3):
        super().__init__(n_channels=n_channels, n_classes=n_classes, base_filters=base_filters)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        # original UNet forward has skip connections list `skips`
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.dropout(x2)          # <--- add dropout in deeper layers
        x3 = self.down2(x2)
        x3 = self.dropout(x3)
        x4 = self.down3(x3)
        x4 = self.dropout(x4)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./artifacts_reg")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # You need your dataset to accept albumentations transform now
    # Example modification in ingest_data.py:
    #   train_dataset = YourDataset(..., transform=get_training_augmentations())
    #   val_dataset   = YourDataset(..., transform=A.Compose([ToTensorV2()]))

    from steps.ingest_data import run_ingest
    train_loader, val_loader, _ = run_ingest(
        args.data_root,
        batch_size=args.batch_size,
        out_dir=args.out_dir,
        augment=True        # <-- make sure you add this flag and apply augmentations
    )

    model = UNetWithDropoutFixed(
        n_channels=3, n_classes=1, base_filters=64, dropout=args.dropout
    )

    train_fn_regularized(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cuda",
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        out_dir=args.out_dir,
        patience=15
    )