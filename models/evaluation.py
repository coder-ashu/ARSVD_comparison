# models/evaluation.py
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class Evaluation(ABC):
    """
    Abstract base class for evaluating model performance.
    """

    @abstractmethod
    def calculate_score(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                        device: str = "cpu") -> Dict[str, Any]:
        """
        Calculates metrics for given model and dataloader.

        Returns:
            Dict containing metric values (e.g., Dice, IoU, accuracy, etc.)
        """
        pass


class SegmentationEvaluator(Evaluation):
    """
    Evaluator for segmentation models like U-Net.
    Computes Dice, IoU, pixel accuracy, and optionally compares
    parameter counts between original and compressed models.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @staticmethod
    def dice_score(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + eps) / (preds.sum() + targets.sum() + eps)
        return dice.item()

    @staticmethod
    def iou_score(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection
        iou = (intersection + eps) / (union + eps)
        return iou.item()

    @staticmethod
    def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
        correct = (preds == targets).float().sum()
        total = torch.numel(targets)
        return (correct / total).item()

    @staticmethod
    def count_parameters(model: torch.nn.Module) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def calculate_score(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                        device: str = "cpu", use_tta: bool = False) -> Dict[str, Any]:
        """
        Calculate segmentation metrics.

        Args:
            model: Model to evaluate
            dataloader: Test data loader
            device: Device to run on
            use_tta: Whether to use Test-Time Augmentation

        Returns:
            Dict with dice, iou, pixel_accuracy
        """
        model.eval()
        model.to(device)
        dice_total, iou_total, acc_total = 0.0, 0.0, 0.0
        n_batches = 0

        with torch.no_grad():
            for images, masks in dataloader:
                images, masks = images.to(device), masks.to(device)

                if use_tta:
                    # Use Test-Time Augmentation
                    preds = self._predict_with_tta(model, images, device)
                else:
                    # Standard prediction
                    outputs = model(images)
                    if outputs.shape[1] > 1:
                        preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
                        masks = masks.squeeze(1)
                    else:
                        preds = (torch.sigmoid(outputs) > self.threshold).float()

                dice_total += self.dice_score(preds, masks)
                iou_total += self.iou_score(preds, masks)
                acc_total += self.pixel_accuracy(preds, masks)
                n_batches += 1

        dice_avg = dice_total / n_batches
        iou_avg = iou_total / n_batches
        acc_avg = acc_total / n_batches

        metrics = {
            "dice": dice_avg,
            "iou": iou_avg,
            "pixel_accuracy": acc_avg,
        }

        tta_str = "with TTA" if use_tta else "without TTA"
        logger.info(f"Segmentation Metrics ({tta_str}) → Dice: {dice_avg:.4f}, IoU: {iou_avg:.4f}, Acc: {acc_avg:.4f}")
        return metrics

    def _predict_with_tta(self, model: torch.nn.Module, images: torch.Tensor,
                         device: str) -> torch.Tensor:
        """
        Test-Time Augmentation: Predict on multiple augmented versions and average.

        Augmentations:
        1. Original image
        2. Horizontal flip
        3. Vertical flip
        4. Horizontal + Vertical flip

        Args:
            model: Model to use for prediction
            images: Input images (N, C, H, W)
            device: Device to run on

        Returns:
            Averaged predictions (N, 1, H, W)
        """
        model.eval()
        preds_list = []

        with torch.no_grad():
            # 1. Original
            outputs = model(images)
            if outputs.shape[1] > 1:
                preds_list.append(torch.argmax(F.softmax(outputs, dim=1), dim=1, keepdim=True))
            else:
                preds_list.append((torch.sigmoid(outputs) > self.threshold).float())

            # 2. Horizontal flip
            outputs_h = model(torch.flip(images, [3]))
            if outputs_h.shape[1] > 1:
                pred_h = torch.argmax(F.softmax(outputs_h, dim=1), dim=1, keepdim=True)
            else:
                pred_h = (torch.sigmoid(outputs_h) > self.threshold).float()
            preds_list.append(torch.flip(pred_h, [3]))

            # 3. Vertical flip
            outputs_v = model(torch.flip(images, [2]))
            if outputs_v.shape[1] > 1:
                pred_v = torch.argmax(F.softmax(outputs_v, dim=1), dim=1, keepdim=True)
            else:
                pred_v = (torch.sigmoid(outputs_v) > self.threshold).float()
            preds_list.append(torch.flip(pred_v, [2]))

            # 4. Both flips
            outputs_hv = model(torch.flip(images, [2, 3]))
            if outputs_hv.shape[1] > 1:
                pred_hv = torch.argmax(F.softmax(outputs_hv, dim=1), dim=1, keepdim=True)
            else:
                pred_hv = (torch.sigmoid(outputs_hv) > self.threshold).float()
            preds_list.append(torch.flip(pred_hv, [2, 3]))

        # Average predictions
        stacked = torch.stack(preds_list, dim=0).float()  # (4, N, 1, H, W)
        avg_pred = (stacked.mean(dim=0) > 0.5).float()  # (N, 1, H, W)

        return avg_pred


def compare_models(original_model: torch.nn.Module,
                   compressed_model: torch.nn.Module,
                   evaluator: SegmentationEvaluator,
                   dataloader: torch.utils.data.DataLoader,
                   device: str = "cpu") -> Dict[str, Any]:
    """
    Compare baseline and compressed models in terms of:
    - Dice, IoU, pixel accuracy
    - Parameter count reduction (%)
    - Model size reduction (if available)
    """

    # parameter counts
    orig_params = evaluator.count_parameters(original_model)
    comp_params = evaluator.count_parameters(compressed_model)
    param_reduction = 100 * (1 - comp_params / orig_params)

    # evaluate both
    logger.info("Evaluating baseline (uncompressed) model...")
    base_metrics = evaluator.calculate_score(original_model, dataloader, device)
    logger.info("Evaluating compressed model...")
    comp_metrics = evaluator.calculate_score(compressed_model, dataloader, device)

    results = {
        "baseline_metrics": base_metrics,
        "compressed_metrics": comp_metrics,
        "param_baseline": orig_params,
        "param_compressed": comp_params,
        "param_reduction_%": param_reduction,
    }

    # relative metric changes
    for k in base_metrics.keys():
        baseline_value = base_metrics[k]
        compressed_value = comp_metrics[k]
        diff = compressed_value - baseline_value
        results[f"{k}_delta"] = diff

    logger.info(
        f"Parameter Reduction: {param_reduction:.2f}% | "
        f"Dice Δ: {results['dice_delta']:.4f}, IoU Δ: {results['iou_delta']:.4f}"
    )

    return results
