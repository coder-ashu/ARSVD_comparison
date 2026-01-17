# models/losses.py
"""
Loss functions for medical image segmentation.
Includes Dice Loss and Combined BCE+Dice Loss for better optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    Dice Loss = 1 - Dice Coefficient
    Dice Coefficient = (2 * intersection + smooth) / (sum_pred + sum_target + smooth)

    This loss directly optimizes the Dice metric used for evaluation.
    """

    def __init__(self, smooth: float = 1.0):
        """
        Args:
            smooth: Smoothing factor to prevent division by zero
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (before sigmoid), shape (N, 1, H, W)
            targets: Ground truth binary masks, shape (N, 1, H, W)

        Returns:
            Dice loss value (lower is better)
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions
        probs = probs.view(-1)
        targets = targets.view(-1)

        # Calculate intersection and union
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()

        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # Return loss (1 - Dice)
        return 1 - dice


class CombinedBCEDiceLoss(nn.Module):
    """
    Combined BCE (Binary Cross-Entropy) and Dice Loss.

    BCE Loss:
    - Good for initial training convergence
    - Optimizes pixel-wise classification
    - Can struggle with class imbalance

    Dice Loss:
    - Directly optimizes Dice metric
    - Better for handling class imbalance
    - Can be unstable early in training

    Combined approach:
    - Gets benefits of both losses
    - More stable training
    - Better final performance
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        """
        Args:
            bce_weight: Weight for BCE loss (default: 0.5)
            dice_weight: Weight for Dice loss (default: 0.5)
            smooth: Smoothing factor for Dice loss
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (before sigmoid), shape (N, 1, H, W)
            targets: Ground truth binary masks, shape (N, 1, H, W)

        Returns:
            Combined weighted loss
        """
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        combined = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)

        return combined


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance.

    Focal Loss = -α * (1 - p_t)^γ * log(p_t)

    Where:
    - p_t is the model's estimated probability for the correct class
    - γ (gamma) is the focusing parameter (reduces loss for well-classified examples)
    - α (alpha) is the class weight

    Useful when foreground (tumor) is much smaller than background.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for rare class (tumor)
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (before sigmoid)
            targets: Ground truth binary masks

        Returns:
            Focal loss value
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


class DiceFocalLoss(nn.Module):
    """
    Combined Dice + Focal Loss.

    Combines the shape-aware optimization of Dice loss with the
    class imbalance handling of Focal loss. This is often the
    best choice for highly imbalanced medical segmentation.
    """

    def __init__(self, dice_weight: float = 0.5, focal_weight: float = 0.5,
                 alpha: float = 0.25, gamma: float = 2.0, smooth: float = 1.0):
        """
        Args:
            dice_weight: Weight for Dice loss component
            focal_weight: Weight for Focal loss component
            alpha: Alpha parameter for Focal loss
            gamma: Gamma parameter for Focal loss
            smooth: Smoothing factor for Dice loss
        """
        super().__init__()
        self.dice = DiceLoss(smooth=smooth)
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, targets)
        focal_loss = self.focal(logits, targets)
        return (self.dice_weight * dice_loss) + (self.focal_weight * focal_loss)
