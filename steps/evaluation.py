# steps/evaluation.py
import os
import json
import argparse
from typing import Dict, Any

import torch
from models.evaluation import SegmentationEvaluator

def evaluate_model_ckpt(ckpt_path: str, model: torch.nn.Module, dataloader, device: str = "cuda", out_dir: str = "./artifacts"):
    device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    evaluator = SegmentationEvaluator(threshold=0.5)
    metrics = evaluator.calculate_score(model, dataloader, device=device)
    os.makedirs(out_dir, exist_ok=True)
    metric_file = os.path.join(out_dir, os.path.basename(ckpt_path) + "_metrics.json")
    with open(metric_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to", metric_file)
    return metrics

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_dir", type=str, default="./artifacts")
    args = p.parse_args()

    # reuse loader
    from steps.ingest_data import run_ingest
    _, _, test_loader = run_ingest(args.data_root, batch_size=args.batch_size, out_dir=args.out_dir)

    # import model skeleton
    from models.unet import UNet
    model = UNet(n_channels=3, n_classes=1)
    metrics = evaluate_model_ckpt(args.ckpt, model, test_loader, device=args.device, out_dir=args.out_dir)
    print("Metrics:", metrics)
