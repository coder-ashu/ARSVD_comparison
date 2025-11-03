"""
run_pipeline.py

Top-level script to run the ingestion -> training -> evaluation pipeline,
extended to:
  - Train baseline U-Net
  - Compress baseline with SVD (one or multiple ranks)
  - Compress baseline with ARSVD (one or multiple taus)
  - Evaluate baseline and each compressed variant on the test set
  - Save a JSON summary with parameter counts, size (MB), compression % and metrics
"""

import argparse
import json
import logging
import os
from typing import Tuple, List

import torch

from pipelines.train_pipeline import train_pipeline

# Steps (adjust these names if your step modules differ)
from steps.ingest_data import run_ingest
from steps.model_train import train_fn
from steps.evaluation import evaluate_model_ckpt  # kept for compatibility if needed

# Models + compression + evaluator
from models.unet import UNet
from models.compression import compress_model_svd, compress_model_arsvd
from models.evaluation import SegmentationEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_list_of_ints(s: str) -> List[int]:
    if s is None or str(s).strip() == "":
        return []
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _parse_list_of_floats(s: str) -> List[float]:
    if s is None or str(s).strip() == "":
        return []
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def make_adapters(data_root: str,
                  batch_size: int,
                  image_size: Tuple[int, int],
                  device: str,
                  epochs: int,
                  lr: float,
                  out_dir: str,
                  multi_class: bool = False,
                  svd_ranks: List[int] = None,
                  arsvd_taus: List[float] = None):
    """
    Build pipeline-adapter callables that match the expected chaining behavior.

    svd_ranks: list of ints, e.g. [16,32] (if empty, default [32] will be used)
    arsvd_taus: list of floats, e.g. [0.85,0.9] (if empty, default [0.9] will be used)
    """

    if svd_ranks is None or len(svd_ranks) == 0:
        svd_ranks = [32]
    if arsvd_taus is None or len(arsvd_taus) == 0:
        arsvd_taus = [0.9]

    def ingest_step():
        """
        Ingest step: no input. Returns (train_loader, val_loader, test_loader).
        """
        return run_ingest(data_root=data_root, batch_size=batch_size, image_size=image_size,
                          multi_class=multi_class, num_workers=4, out_dir=out_dir)

    def train_step(prev):
        """
        Train step: expects (train_loader, val_loader, test_loader).
        Creates a new UNet, trains it using train_fn, and returns a dict with 'ckpt' and optional 'model'.
        """
        # prev could be either a tuple (train_loader, val_loader, test_loader)
        try:
            train_loader, val_loader, test_loader = prev
        except Exception:
            # if prev is a dict or custom object, try common keys
            if isinstance(prev, dict) and "train_loader" in prev:
                train_loader = prev["train_loader"]
                val_loader = prev["val_loader"]
                test_loader = prev.get("test_loader")
            else:
                raise ValueError("train_step received unexpected input from previous step")

        model = UNet(n_channels=3, n_classes=1, base_filters=64)
        result = train_fn(model=model, train_loader=train_loader, val_loader=val_loader,
                          device=device, epochs=epochs, lr=lr, out_dir=out_dir)
        # train_fn is expected to return a dict with at least 'ckpt' (path to saved checkpoint).
        return result

    def eval_step(prev):
        """
        Evaluation and compression step:
        - load baseline checkpoint (from prev)
        - compress baseline with SVD for each rank in svd_ranks
        - compress baseline with ARSVD for each tau in arsvd_taus
        - evaluate baseline + all variants on the test set
        - save summary json to out_dir/experiment_summary.json
        """
        ckpt = None
        model_obj = None
        if isinstance(prev, dict):
            ckpt = prev.get("ckpt")
            model_obj = prev.get("model")
        elif isinstance(prev, str) and os.path.exists(prev):
            ckpt = prev

        if ckpt is None:
            ckpt = os.path.join(out_dir, "baseline_trained.pth")
            if not os.path.exists(ckpt):
                raise FileNotFoundError("No checkpoint found to evaluate.")

        _, _, test_loader = run_ingest(data_root=data_root, batch_size=batch_size, image_size=image_size,
                                       multi_class=multi_class, num_workers=2, out_dir=out_dir)

        baseline = UNet(n_channels=3, n_classes=1)
        baseline.load_state_dict(torch.load(ckpt, map_location="cpu"))

        # Create evaluator
        evaluator = SegmentationEvaluator(threshold=0.5)

        device_to_use = device if (torch.cuda.is_available() and device.startswith("cuda")) else "cpu"
        logger.info(f"Evaluating on device: {device_to_use}")

        # helper to get params and serialized size in MB
        def model_info(m):
            params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            size_mb = sum(p.numel() * p.element_size() for p in m.parameters()) / (1024 ** 2)
            return {"params": params, "size_MB": round(size_mb, 4)}

        # Evaluate baseline
        logger.info("Evaluating baseline model...")
        baseline_metrics = evaluator.calculate_score(baseline, test_loader, device=device_to_use)
        base_info = model_info(baseline)

        # Collect SVD variants
        svd_results = []
        for rank in svd_ranks:
            logger.info(f"Creating SVD compression with rank={rank} ...")
            svd_model = compress_model_svd(baseline, rank=rank, use_randomized=False)
            svd_ckpt = os.path.join(out_dir, f"svd_rank_{rank}.pth")
            torch.save(svd_model.state_dict(), svd_ckpt)

            # evaluate
            logger.info(f"Evaluating SVD rank={rank} ...")
            svd_metrics = evaluator.calculate_score(svd_model, test_loader, device=device_to_use)
            svd_info = model_info(svd_model)
            svd_results.append({
                "rank": rank,
                "ckpt": svd_ckpt,
                "info": svd_info,
                "metrics": svd_metrics,
                "compression_vs_baseline_%": {
                    "params": round(100 * (1 - svd_info["params"] / base_info["params"]), 4),
                    "size_MB": round(100 * (1 - svd_info["size_MB"] / base_info["size_MB"]), 4),
                }
            })

        # Collect ARSVD variants
        arsvd_results = []
        for tau in arsvd_taus:
            logger.info(f"Creating ARSVD compression with tau={tau} ...")
            arsvd_model = compress_model_arsvd(baseline, tau=tau, recon_method="randomized")
            arsvd_ckpt = os.path.join(out_dir, f"arsvd_tau_{tau:.3f}.pth")
            torch.save(arsvd_model.state_dict(), arsvd_ckpt)

            logger.info(f"Evaluating ARSVD tau={tau} ...")
            arsvd_metrics = evaluator.calculate_score(arsvd_model, test_loader, device=device_to_use)
            arsvd_info = model_info(arsvd_model)
            arsvd_results.append({
                "tau": tau,
                "ckpt": arsvd_ckpt,
                "info": arsvd_info,
                "metrics": arsvd_metrics,
                "compression_vs_baseline_%": {
                    "params": round(100 * (1 - arsvd_info["params"] / base_info["params"]), 4),
                    "size_MB": round(100 * (1 - arsvd_info["size_MB"] / base_info["size_MB"]), 4),
                }
            })


        summary = {
            "baseline": {
                "ckpt": ckpt,
                "info": base_info,
                "metrics": baseline_metrics,
            },
            "svd_variants": svd_results,
            "arsvd_variants": arsvd_results
        }


        summary_path = os.path.join(out_dir, "experiment_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Experiment summary saved to {summary_path}")
     
        return summary

    return ingest_step, train_step, eval_step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="Root folder that contains train/valid/test subfolders")
    p.add_argument("--out_dir", default="./artifacts")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--image_size", nargs=2, type=int, default=(256, 256))
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--multi_class", action="store_true")
    p.add_argument("--svd_ranks", type=str, default="32",
                   help="Comma-separated ranks to try for SVD compression, e.g. '16,32,64'")
    p.add_argument("--arsvd_taus", type=str, default="0.9",
                   help="Comma-separated taus to try for ARSVD, e.g. '0.85,0.9,0.95'")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    svd_ranks = _parse_list_of_ints(args.svd_ranks)
    arsvd_taus = _parse_list_of_floats(args.arsvd_taus)

    ingest_step, train_step, eval_step = make_adapters(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=tuple(args.image_size),
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        out_dir=args.out_dir,
        multi_class=args.multi_class,
        svd_ranks=svd_ranks,
        arsvd_taus=arsvd_taus,
    )

    pipeline = train_pipeline(
        ingest_step,
        train_step,
        eval_step,
        name="arsvd_training_pipeline",
    )

    outputs = pipeline.run()

    # Save a small summary mapping for UI convenience 
    summary_for_ui = {}
    for idx, info in outputs.items():
        out_repr = info["output"]
        try:
            if isinstance(out_repr, dict):
                small = {}
                for k, v in out_repr.items():
                    if k in ("baseline", "svd_variants", "arsvd_variants"):
                        small[k] = str(type(v))
                    else:
                        small[k] = str(type(v))
                summary_for_ui[idx] = small
            else:
                summary_for_ui[idx] = str(type(out_repr))
        except Exception:
            summary_for_ui[idx] = str(type(out_repr))

    with open(os.path.join(args.out_dir, "pipeline_outputs_summary.json"), "w") as f:
        json.dump(summary_for_ui, f, indent=2)

    logger.info("Pipeline finished. Summary saved to %s", os.path.join(args.out_dir, "pipeline_outputs_summary.json"))
    print("Pipeline finished. Summary saved to", os.path.join(args.out_dir, "pipeline_outputs_summary.json"))


if __name__ == "__main__":
    main()
