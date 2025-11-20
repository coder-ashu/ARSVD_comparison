#!/usr/bin/env python3
"""
run_pipeline.py

Top-level script to run the ingestion -> training -> evaluation pipeline,
extended to:
  - Train baseline U-Net
  - Compress baseline with SVD (one or multiple ranks) using randomized SVD
  - Compress baseline with ARSVD (one or multiple taus) using randomized reconstruction
  - Factorize compressed models (actual reduced weights) and evaluate them
  - Save a JSON summary with parameter counts, size (MB), compression % and metrics
  - Plot IoU vs SVD_rank and IoU vs ARSVD_tau and save to out_dir
"""
import argparse
import json
import logging
import os
from typing import Tuple, List, Optional, Any, Dict

import torch
import torch.nn as nn

# matplotlib: ensure non-interactive backend for headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pipelines.train_pipeline import train_pipeline

# Steps (adjust these names if your step modules differ)
from steps.ingest_data import run_ingest
from steps.model_train import train_fn
from steps.evaluation import evaluate_model_ckpt  # kept for compatibility if needed

# Models + compression + evaluator
from models.unet import UNet
from models.compression import (
    compress_model_svd,
    compress_model_arsvd,
    model_compressed_storage,
    compress_model_factorized_copy,
    model_size_bytes,
)
from models.evaluation import SegmentationEvaluator

from models.vgg16 import vgg16_cifar

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


def extract_iou(metrics: Any) -> Optional[float]:
    """
    Try to extract a single IoU value (mean IoU) from the evaluator metrics output.
    Handles:
      - scalar under keys like 'iou', 'IoU', 'mean_iou', 'mIoU', 'miou'
      - dict of per-class IoUs -> returns mean
      - list/tuple -> takes mean of numeric elements
      - nested structures: searches recursively for common keys
    Returns None if no IoU-like value is found.
    """
    if metrics is None:
        return None

    # If a number, return it
    if isinstance(metrics, (int, float)):
        return float(metrics)

    # If dict, try common keys first, else collapse per-class values
    if isinstance(metrics, dict):
        # normalise keys
        key_map = {k.lower(): k for k in metrics.keys()}
        # candidate names
        candidates = ['iou', 'mean_iou', 'm_iou', 'miou', 'mean iou', 'mean-iou']
        for c in candidates:
            if c in key_map:
                v = metrics[key_map[c]]
                if isinstance(v, (int, float)):
                    return float(v)
                # if it's a dict or list, handle below
                metrics = v  # fallthrough to next logic

        # If value still a dict: likely per-class IoUs
        # If values are numeric, average them
        numeric_vals = []
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                numeric_vals.append(float(v))
            elif isinstance(v, dict) or isinstance(v, list) or isinstance(v, tuple):
                # try recursively
                nested = extract_iou(v)
                if nested is not None:
                    numeric_vals.append(nested)
        if numeric_vals:
            return float(sum(numeric_vals) / len(numeric_vals))

        # Otherwise, search recursively in nested dicts
        for v in metrics.values():
            nested = extract_iou(v)
            if nested is not None:
                return nested

        return None

    # If list/tuple, average numeric entries or try to extract from first element
    if isinstance(metrics, (list, tuple)):
        numeric_vals = [float(x) for x in metrics if isinstance(x, (int, float))]
        if numeric_vals:
            return float(sum(numeric_vals) / len(numeric_vals))
        # else try recursively
        for item in metrics:
            nested = extract_iou(item)
            if nested is not None:
                return nested
        return None

    # Unknown type - give up
    return None


def make_adapters(data_root: str,
                  batch_size: int,
                  image_size: Tuple[int, int],
                  device: str,
                  epochs: int,
                  lr: float,
                  out_dir: str,
                  dataset_type: str = "coco",
                  num_classes: int = 1,
                  multi_class: bool = False,
                  svd_ranks: List[int] = None,
                  arsvd_taus: List[float] = None,
                  finetune_compressed: bool = False,
                  finetune_epochs: int = 3,
                  finetune_lr: float = 1e-5):

    """
    Build pipeline-adapter callables that match the expected chaining behavior.
    """

    if svd_ranks is None or len(svd_ranks) == 0:
        svd_ranks = [32]
    if arsvd_taus is None or len(arsvd_taus) == 0:
        arsvd_taus = [0.9]

    def ingest_step():
        """
        Ingest step: no input. Returns (train_loader, val_loader, test_loader).
        """
        # note: ensure run_ingest accepts dataset_type; if not, update steps/ingest_data.py accordingly.
        return run_ingest(data_root=data_root, batch_size=batch_size, image_size=image_size,
                          multi_class=multi_class, num_workers=4, out_dir=out_dir,
                          dataset_type=dataset_type)


    def train_step(prev):
        """
        Train step: expects (train_loader, val_loader, test_loader).
        Creates a model (UNet or VGG), trains it using train_fn or a classification training wrapper,
        and returns a dict with 'ckpt' and 'model'.
        """
        try:
            train_loader, val_loader, test_loader = prev
        except Exception:
            if isinstance(prev, dict) and "train_loader" in prev:
                train_loader = prev["train_loader"]
                val_loader = prev["val_loader"]
                test_loader = prev.get("test_loader")
            else:
                raise ValueError("train_step received unexpected input from previous step")

        # choose model based on dataset_type or explicit model_name
        chosen_model = None
        if dataset_type.lower() == "cifar10" or (args.model_name and args.model_name.lower() == "vgg16"):
            # CIFAR-10 classification: use VGG-16 CIFAR variant
            chosen_model = vgg16_cifar(num_classes=num_classes, pretrained=False)
        else:
            # default: U-Net for segmentation
            chosen_model = UNet(n_channels=3, n_classes=num_classes, base_filters=64)

        # training function: you already have train_fn for UNet. For classification,
        # create or reuse a classification training function in steps.model_train.
        # Here we try to call a generic train_fn that accepts dataset_type flag; if not present,
        # the user should implement train_fn_classification in steps.model_train.
        try:
            # prefer a classification-specific training function if dataset_type == cifar10
            if dataset_type.lower() == "cifar10" and hasattr(train_fn, "train_classification"):
                result = train_fn.train_classification(model=chosen_model, train_loader=train_loader,
                                                       val_loader=val_loader, device=device,
                                                       epochs=epochs, lr=lr, out_dir=out_dir,
                                                       num_classes=num_classes)
            elif dataset_type.lower() == "cifar10" and hasattr(train_fn, "__call__"):
                # If train_fn is generic and can handle classification based on model outputs, call it.
                result = train_fn(model=chosen_model, train_loader=train_loader, val_loader=val_loader,
                                  device=device, epochs=epochs, lr=lr, out_dir=out_dir, dataset_type=dataset_type)
            else:
                # segmentation / default flow
                result = train_fn(model=chosen_model, train_loader=train_loader, val_loader=val_loader,
                                  device=device, epochs=epochs, lr=lr, out_dir=out_dir)
        except TypeError:
            # fallback: try calling without dataset_type
            result = train_fn(model=chosen_model, train_loader=train_loader, val_loader=val_loader,
                              device=device, epochs=epochs, lr=lr, out_dir=out_dir)

        return result


def eval_step(prev):
    """
    Evaluation and compression step (supports 'coco' segmentation and 'cifar10' classification).
    - load baseline checkpoint (from prev)
    - compress baseline with SVD for each rank in svd_ranks (randomized)
    - compress baseline with ARSVD for each tau in arsvd_taus (randomized recon)
    - factorize the compressed models (actual reduced params)
    - evaluate baseline + factorized variants on the test set
    - save summary json to out_dir/experiment_summary.json
    - plot IoU/Accuracy vs SVD_rank and IoU/Accuracy vs ARSVD_tau and save PNGs
    - plot per-layer ranks for SVD & ARSVD
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

    # Request the proper test loader for chosen dataset_type
    _, _, test_loader = run_ingest(data_root=data_root, batch_size=batch_size, image_size=image_size,
                                   multi_class=multi_class, num_workers=2, out_dir=out_dir,
                                   dataset_type=dataset_type)

    # Select device
    device_to_use = device if (torch.cuda.is_available() and device.startswith("cuda")) else "cpu"
    logger.info(f"Evaluating on device: {device_to_use}")

    # helper to get params and serialized size in MB
    def model_info(m: nn.Module):
        params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        size_mb = model_size_bytes(m) / (1024 ** 2)
        return {"params": int(params), "size_MB": round(float(size_mb), 6)}

    # classification evaluator (Top-1 accuracy)
    def evaluate_classification(model, loader, device):
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = 100.0 * correct / (total if total > 0 else 1)
        return {"top1_accuracy": acc, "correct": correct, "total": total}

    # segmentation evaluator (existing)
    seg_evaluator = SegmentationEvaluator(threshold=0.5)

    # wrapper to evaluate model according to dataset type (moves model to device and back)
    def evaluate_model_wrapper(m: nn.Module):
        m.to(device_to_use)
        m.eval()
        if dataset_type.lower() == "cifar10":
            metrics = evaluate_classification(m, test_loader, device_to_use)
        else:
            metrics = seg_evaluator.calculate_score(m, test_loader, device=device_to_use)
        m.to("cpu")
        return metrics

    # Metric extractor for plotting (IoU for segmentation, top1_accuracy for classification)
    def get_plot_metric(metrics: Any):
        if metrics is None:
            return None
        if dataset_type.lower() == "cifar10":
            # expect metrics to be {"top1_accuracy": value}
            if isinstance(metrics, dict) and "top1_accuracy" in metrics:
                return float(metrics["top1_accuracy"])
            # fallback: try numeric extraction
            if isinstance(metrics, (int, float)):
                return float(metrics)
            return None
        else:
            # segmentation: use your existing extractor for IoU
            return extract_iou(metrics)

    # ---------------------
    # Evaluate baseline
    # ---------------------
    logger.info("Evaluating baseline model...")
    if dataset_type.lower() == "cifar10":
        baseline = vgg16_cifar(num_classes=num_classes)
    else:
        baseline = UNet(n_channels=3, n_classes=num_classes, base_filters=64)

    baseline.load_state_dict(torch.load(ckpt, map_location="cpu"))
    baseline_metrics = evaluate_model_wrapper(baseline)
    base_info = model_info(baseline)

    summary = {
        "baseline": {
            "ckpt": ckpt,
            "info": base_info,
            "metrics": baseline_metrics,
        },
        "svd_variants": [],
        "arsvd_variants": []
    }

    # ---------------------
    # SVD fixed-rank variants
    # ---------------------
    for rank in svd_ranks:
        logger.info(f"Compressing (SVD randomized) with rank={rank} ...")
        # compress_model_svd may return (model, rank_map, weights_map) — we only need rank_map here
        try:
            _, svd_rank_map, svd_weights_map = compress_model_svd(
                baseline,
                rank=rank,
                use_randomized=True,
                random_state=42,
                svd_kwargs={"n_oversamples": 10, "n_iter": 2},
                return_rank_map=True,
                return_weights=True,
            )
        except TypeError:
            # fallback if function signature differs
            svd_res = compress_model_svd(baseline, rank=rank, use_randomized=True)
            svd_rank_map = svd_res[1] if len(svd_res) > 1 else {}
            svd_weights_map = svd_res[2] if len(svd_res) > 2 else {}

        # theoretical storage for factors
        theo_mb = model_compressed_storage(baseline, svd_rank_map)

        # create real factorized model (actual param reduction)
        svd_fact_model = compress_model_factorized_copy(baseline, rank_map=svd_rank_map, default_rank=rank,
                                                        use_randomized=True, random_state=42,
                                                        n_oversamples=10, n_iter=2)
        svd_fact_model.eval()

        # evaluate factorized model
        svd_metrics = evaluate_model_wrapper(svd_fact_model)
        svd_info = model_info(svd_fact_model)

        # save checkpoint
        svd_ckpt = os.path.join(out_dir, f"svd_factorized_rank_{rank}.pth")
        torch.save(svd_fact_model.state_dict(), svd_ckpt)
        svd_fact_model.to("cpu")

        summary["svd_variants"].append({
            "rank": rank,
            "ckpt": svd_ckpt,
            "info": svd_info,
            "theoretical_factors_MB": round(float(theo_mb), 6),
            "metrics": svd_metrics,
            "compression_vs_baseline_%": {
                "params": round(100 * (1 - svd_info["params"] / base_info["params"]), 4),
                "size_MB": round(100 * (1 - svd_info["size_MB"] / base_info["size_MB"]), 4),
            },
            "per_layer_ranks": svd_rank_map
        })

    # ---------------------
    # ARSVD adaptive-rank variants
    # ---------------------
    for tau in arsvd_taus:
        logger.info(f"Compressing (ARSVD randomized) with tau={tau} ...")
        try:
            _, arsvd_rank_map, arsvd_weights_map = compress_model_arsvd(
                baseline,
                tau=tau,
                recon_method="randomized",
                n_oversamples=10,
                n_iter=2,
                random_state=42,
                return_rank_map=True,
                return_weights=True,
            )
        except TypeError:
            arsvd_res = compress_model_arsvd(baseline, tau=tau, recon_method="randomized")
            arsvd_rank_map = arsvd_res[1] if len(arsvd_res) > 1 else {}
            arsvd_weights_map = arsvd_res[2] if len(arsvd_res) > 2 else {}

        theo_mb = model_compressed_storage(baseline, arsvd_rank_map)

        arsvd_fact_model = compress_model_factorized_copy(baseline, rank_map=arsvd_rank_map,
                                                          use_randomized=True, random_state=42,
                                                          n_oversamples=10, n_iter=2)
        arsvd_fact_model.eval()

        arsvd_metrics = evaluate_model_wrapper(arsvd_fact_model)
        arsvd_info = model_info(arsvd_fact_model)

        arsvd_ckpt = os.path.join(out_dir, f"arsvd_factorized_tau_{tau:.3f}.pth")
        torch.save(arsvd_fact_model.state_dict(), arsvd_ckpt)
        arsvd_fact_model.to("cpu")

        summary["arsvd_variants"].append({
            "tau": tau,
            "ckpt": arsvd_ckpt,
            "info": arsvd_info,
            "theoretical_factors_MB": round(float(theo_mb), 6),
            "metrics": arsvd_metrics,
            "compression_vs_baseline_%": {
                "params": round(100 * (1 - arsvd_info["params"] / base_info["params"]), 4),
                "size_MB": round(100 * (1 - arsvd_info["size_MB"] / base_info["size_MB"]), 4),
            },
            "per_layer_ranks": arsvd_rank_map
        })

    # ---------------------
    # Save summary JSON
    # ---------------------
    summary_path = os.path.join(out_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Experiment summary saved to {summary_path}")

    # ---------------------
    # Plotting section
    # ---------------------
    try:
        # Helper to extract plotting metric (IoU or Accuracy)
        def plot_metric_from_metrics(metrics):
            return get_plot_metric(metrics) if (globals().get('get_plot_metric') is not None) else get_plot_metric(metrics)

        # ---------- SVD plot data (Metric and Size) ----------
        svd_data_x = []
        svd_data_y = []
        svd_data_size = []
        for v in summary.get("svd_variants", []):
            rank = v.get("rank")
            metrics = v.get("metrics")
            info = v.get("info", {})
            metric_val = get_plot_metric(metrics)
            size_mb = info.get("size_MB", 0)
            if metric_val is None:
                logger.warning(f"Could not extract metric for SVD rank={rank}; skipping in plot.")
                continue
            svd_data_x.append(rank)
            svd_data_y.append(metric_val)
            svd_data_size.append(size_mb)

        if svd_data_x and svd_data_y:
            svd_pairs = sorted(zip(svd_data_x, svd_data_y, svd_data_size), key=lambda x: x[0])
            svd_x_sorted, svd_y_sorted, svd_size_sorted = zip(*svd_pairs)

            # Create subplot with Metric and Size
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

            # Metric subplot
            ax1.plot(list(svd_x_sorted), list(svd_y_sorted), marker='o', color='blue', label=('Accuracy' if dataset_type.lower() == 'cifar10' else 'IoU'))
            ax1.set_xlabel("SVD Rank")
            ax1.set_ylabel("Accuracy" if dataset_type.lower() == 'cifar10' else "IoU")
            ax1.set_title(("Accuracy" if dataset_type.lower() == 'cifar10' else "IoU") + " vs SVD Rank")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Size subplot
            ax2.plot(list(svd_x_sorted), list(svd_size_sorted), marker='s', color='red', label='Size (MB)')
            ax2.set_xlabel("SVD Rank")
            ax2.set_ylabel("Model Size (MB)")
            ax2.set_title("Model Size vs SVD Rank")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            svd_plot_path = os.path.join(out_dir, "metric_and_size_vs_svd_rank.png")
            plt.savefig(svd_plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"SVD metric and Size plot saved to {svd_plot_path}")
        else:
            logger.info("No valid SVD metric data found; skipping SVD plot.")

        # ---------- ARSVD plot data (Metric and Size) ----------
        arsvd_data_x = []
        arsvd_data_y = []
        arsvd_data_size = []
        for v in summary.get("arsvd_variants", []):
            tau = v.get("tau")
            metrics = v.get("metrics")
            info = v.get("info", {})
            metric_val = get_plot_metric(metrics)
            size_mb = info.get("size_MB", 0)
            if metric_val is None:
                logger.warning(f"Could not extract metric for ARSVD tau={tau}; skipping in plot.")
                continue
            arsvd_data_x.append(float(tau))
            arsvd_data_y.append(metric_val)
            arsvd_data_size.append(size_mb)

        if arsvd_data_x and arsvd_data_y:
            arsvd_pairs = sorted(zip(arsvd_data_x, arsvd_data_y, arsvd_data_size), key=lambda x: x[0])
            arsvd_x_sorted, arsvd_y_sorted, arsvd_size_sorted = zip(*arsvd_pairs)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

            # Metric subplot
            ax1.plot(list(arsvd_x_sorted), list(arsvd_y_sorted), marker='o', color='blue', label=('Accuracy' if dataset_type.lower() == 'cifar10' else 'IoU'))
            ax1.set_xlabel("ARSVD Tau")
            ax1.set_ylabel("Accuracy" if dataset_type.lower() == 'cifar10' else "IoU")
            ax1.set_title(("Accuracy" if dataset_type.lower() == 'cifar10' else "IoU") + " vs ARSVD Tau")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Size subplot
            ax2.plot(list(arsvd_x_sorted), list(arsvd_size_sorted), marker='s', color='red', label='Size (MB)')
            ax2.set_xlabel("ARSVD Tau")
            ax2.set_ylabel("Model Size (MB)")
            ax2.set_title("Model Size vs ARSVD Tau")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            arsvd_plot_path = os.path.join(out_dir, "metric_and_size_vs_arsvd_tau.png")
            plt.savefig(arsvd_plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"ARSVD metric and Size plot saved to {arsvd_plot_path}")
        else:
            logger.info("No valid ARSVD metric data found; skipping ARSVD plot.")

        # ---------- Per-layer ranks plots (SVD) ----------
        svd_per_layer_data = {}
        for v in summary.get("svd_variants", []):
            rank = v.get("rank")
            per_layer_ranks = v.get("per_layer_ranks", {})
            if per_layer_ranks:
                svd_per_layer_data[rank] = per_layer_ranks

        if svd_per_layer_data:
            all_layers = set()
            for ranks in svd_per_layer_data.values():
                all_layers.update(ranks.keys())
            all_layers = sorted(all_layers)

            fig, ax = plt.subplots(figsize=(15, 8))
            x_pos = range(len(all_layers))

            rank_values = sorted(svd_per_layer_data.keys())
            colors = ['blue', 'red', 'green', 'orange', 'purple']

            for i, rank in enumerate(rank_values):
                ranks_dict = svd_per_layer_data[rank]
                y_values = [ranks_dict.get(layer, 0) for layer in all_layers]
                ax.plot(x_pos, y_values, marker='o', color=colors[i % len(colors)],
                        label=f'Rank {rank}', linewidth=2, markersize=6)

            ax.set_xlabel('Layer Names')
            ax.set_ylabel('Ranks')
            ax.set_title('Per-Layer Ranks for Different SVD Rank Values')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([layer.replace('double_conv', 'dc').replace('maxpool_conv', 'mpc')
                               for layer in all_layers], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            svd_layer_plot_path = os.path.join(out_dir, "per_layer_ranks_svd.png")
            plt.savefig(svd_layer_plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"SVD per-layer ranks plot saved to {svd_layer_plot_path}")
        else:
            logger.info("No SVD per-layer rank data found; skipping SVD layer plot.")

        # ---------- Per-layer ranks plots (ARSVD) ----------
        arsvd_per_layer_data = {}
        for v in summary.get("arsvd_variants", []):
            tau = v.get("tau")
            per_layer_ranks = v.get("per_layer_ranks", {})
            if per_layer_ranks:
                arsvd_per_layer_data[tau] = per_layer_ranks

        if arsvd_per_layer_data:
            all_layers = set()
            for ranks in arsvd_per_layer_data.values():
                all_layers.update(ranks.keys())
            all_layers = sorted(all_layers)

            fig, ax = plt.subplots(figsize=(15, 8))
            x_pos = range(len(all_layers))

            tau_values = sorted(arsvd_per_layer_data.keys())
            colors = ['blue', 'red', 'green', 'orange', 'purple']

            for i, tau in enumerate(tau_values):
                ranks_dict = arsvd_per_layer_data[tau]
                y_values = [ranks_dict.get(layer, 0) for layer in all_layers]
                ax.plot(x_pos, y_values, marker='s', color=colors[i % len(colors)],
                        label=f'Tau {tau:.3f}', linewidth=2, markersize=6)

            ax.set_xlabel('Layer Names')
            ax.set_ylabel('Ranks')
            ax.set_title('Per-Layer Ranks for Different ARSVD Tau Values')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([layer.replace('double_conv', 'dc').replace('maxpool_conv', 'mpc')
                               for layer in all_layers], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            arsvd_layer_plot_path = os.path.join(out_dir, "per_layer_ranks_arsvd.png")
            plt.savefig(arsvd_layer_plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"ARSVD per-layer ranks plot saved to {arsvd_layer_plot_path}")
        else:
            logger.info("No ARSVD per-layer rank data found; skipping ARSVD layer plot.")

    except Exception as e:
        logger.exception("Failed to generate plots: %s", e)

    # return the summary
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
    p.add_argument("--finetune_compressed", action="store_true",
                   help="If set, will call train_fn to fine-tune each compressed model (train_fn must accept the same signature).")
    p.add_argument("--finetune_epochs", type=int, default=3)
    p.add_argument("--finetune_lr", type=float, default=1e-5)
    p.add_argument("--dataset_type", default="coco",
                   choices=["coco", "cifar10"],
                   help="Dataset type: 'coco' for segmentation (U-Net) or 'cifar10' for classification (VGG)")
    p.add_argument("--model_name", default=None,
                   help="Optional override model: 'unet' or 'vgg16' (if unset, inferred from dataset_type)")
    p.add_argument("--num_classes", type=int, default=1,
                   help="Number of classes (for classification: 10 for CIFAR-10; for segmentation set appropriately)")


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
        finetune_compressed=args.finetune_compressed,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
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
