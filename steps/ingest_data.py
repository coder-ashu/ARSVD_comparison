# steps/ingest_data.py
import os
import json
import argparse
from typing import Tuple

import torch
from data.dataset import create_dataloaders

def run_ingest(data_root: str, batch_size: int = 8, image_size=(256,256), multi_class=False, num_workers=4, out_dir="./artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        image_size=image_size,
        multi_class=multi_class,
        num_workers=num_workers
    )

    # Save a small manifest about the dataset shapes and counts
    manifest = {
        "data_root": data_root,
        "batch_size": batch_size,
        "image_size": image_size,
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "test_batches": len(test_loader)
    }
    with open(os.path.join(out_dir, "dataset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--image_size", nargs=2, type=int, default=(256,256))
    p.add_argument("--multi_class", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_dir", type=str, default="./artifacts")
    args = p.parse_args()
    run_ingest(args.data_root, args.batch_size, tuple(args.image_size), args.multi_class, args.num_workers, args.out_dir)
    print("Ingest complete. Manifest saved to", os.path.join(args.out_dir, "dataset_manifest.json"))
