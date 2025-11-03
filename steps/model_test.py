# steps/model_test.py
import argparse
import os
import json

import torch
from models.unet import UNet
from models.evaluation import SegmentationEvaluator

def test_checkpoint(ckpt_path: str, data_root: str, batch_size: int = 8, device: str = "cuda", out_dir: str = "./artifacts"):
    from steps.ingest_data import run_ingest
    _, _, test_loader = run_ingest(data_root, batch_size=batch_size, out_dir=out_dir)

    model = UNet(n_channels=3, n_classes=1)
    device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    evaluator = SegmentationEvaluator()
    metrics = evaluator.calculate_score(model, test_loader, device=device)

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, os.path.basename(ckpt_path) + ".test_metrics.json")
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default="./artifacts")
    args = p.parse_args()
    res = test_checkpoint(args.ckpt, args.data_root, args.batch_size, args.device, args.out_dir)
    print("Test metrics:", res)
