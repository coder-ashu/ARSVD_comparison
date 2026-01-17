# data/dataset.py
"""
Data loading and augmentation for brain tumor segmentation.

This module implements advanced data augmentation using Albumentations, specifically
designed for medical image segmentation. The augmentation strategy includes:

1. Geometric transforms (applied to both images and masks):
   - Horizontal/Vertical flips: Simulates different orientations
   - Rotation: Handles different scan angles
   - ShiftScaleRotate: Simulates positional variations
   - ElasticTransform: Simulates tissue deformation (very important for medical imaging)

2. Intensity transforms (applied only to images):
   - RandomBrightnessContrast: Handles MRI intensity variations
   - GaussNoise: Simulates MRI scanner noise
   - GaussianBlur: Simulates different imaging resolutions
   - CLAHE: Improves local contrast (Contrast Limited Adaptive Histogram Equalization)

3. Advanced transforms (in 'heavy' mode):
   - GridDistortion: Adds local geometric distortions
   - OpticalDistortion: Simulates lens/scanner distortions
   - RandomGamma: Handles different intensity scales
   - CoarseDropout: Cutout regularization for better generalization

Augmentation levels:
- 'light': Basic geometric transforms only (faster training)
- 'medium': Balanced augmentation (RECOMMENDED for most cases)
- 'heavy': Maximum regularization (use if severe overfitting)

Expected performance improvements:
- Dice score: +3-8% improvement
- IoU: +3-7% improvement
- Better generalization to unseen data
- Reduced overfitting on small medical datasets
"""

import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2


class COCOSegmentationDataset(Dataset):
    """
    Dataset loader for COCO-style segmentation annotations.
    Compatible with pipelines for U-Net (image, mask) pairs.
    Supports Albumentations for simultaneous image+mask augmentation.
    """

    def __init__(self, root_dir: str, annotation_file: str,
                 transform=None, multi_class=False, is_training=True):
        """
        Args:
            root_dir: directory containing images
            annotation_file: path to _annotations.coco.json
            transform: Albumentations transform (applied to both image and mask)
            multi_class: if True, produce multi-class masks (else binary)
            is_training: if True, use training mode flags
        """
        super().__init__()
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.multi_class = multi_class
        self.is_training = is_training

    def __len__(self):
        return len(self.image_ids)

    def _load_image(self, image_id):
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")
        return image, image_info

    def _load_mask(self, image_info, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

        for ann in anns:
            category_id = ann["category_id"] if self.multi_class else 1
            mask = np.maximum(mask, self.coco.annToMask(ann) * category_id)

        return Image.fromarray(mask)

    # inside COCOSegmentationDataset class


    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image, image_info = self._load_image(image_id)
        mask = self._load_mask(image_info, image_id)

        # Convert PIL images to numpy arrays for Albumentations
        image_np = np.array(image)
        mask_np = np.array(mask)

        # Apply Albumentations transform (handles both image and mask simultaneously)
        if self.transform is not None:
            transformed = self.transform(image=image_np, mask=mask_np)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Fallback: convert to tensor without augmentation
            image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask_np).long().unsqueeze(0)

        # Ensure mask is in correct format: (1, H, W) with dtype long
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 3 and mask.shape[0] != 1:
            mask = mask[0:1]

        mask = mask.long()

        return image, mask




def create_training_transforms(image_size=(256, 256), augment_level='medium'):
    """
    Advanced data augmentation for training medical images.

    Args:
        image_size: Target size (H, W)
        augment_level: 'light', 'medium', or 'heavy' augmentation intensity

    Returns:
        Albumentations compose object for simultaneous image+mask augmentation
    """
    if augment_level == 'light':
        # Light augmentation - basic geometric transforms
        transform = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5, border_mode=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    elif augment_level == 'medium':
        # Medium augmentation - balanced for medical imaging
        transform = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            # Geometric transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5, border_mode=0),
            # Use Affine instead of deprecated ShiftScaleRotate
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.1), rotate=(0, 30), p=0.5, mode=0),
            # ElasticTransform: alpha=1, sigma=50 (alpha_affine removed in newer versions)
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            # Intensity transforms (images only)
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # GaussNoise: var_limit is deprecated, use variance_limit
            A.GaussNoise(variance_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    elif augment_level == 'heavy':
        # Heavy augmentation - maximum regularization
        transform = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            # Geometric transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.5, border_mode=0),
            A.Affine(scale=(0.8, 1.2), translate_percent=(0.0, 0.15), rotate=(0, 45), p=0.5, mode=0),
            A.ElasticTransform(alpha=2, sigma=50, p=0.5),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(p=0.3),
            # Intensity transforms
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.GaussNoise(variance_limit=(10.0, 80.0), p=0.4),
            A.GaussianBlur(blur_limit=(3, 9), p=0.4),
            A.CLAHE(clip_limit=3.0, p=0.4),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        raise ValueError(f"Unknown augment_level: {augment_level}. Use 'light', 'medium', or 'heavy'.")

    return transform


def create_validation_transforms(image_size=(256, 256)):
    """
    Minimal transforms for validation/test - NO augmentation.

    Only resize and normalize to ensure consistent evaluation.
    """
    transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return transform


def create_transforms(image_size=(256, 256)):
    """
    Legacy function name for backward compatibility.
    Returns medium training transforms (recommended default).
    """
    return create_training_transforms(image_size, augment_level='medium')


def create_dataloaders(data_root: str, batch_size: int = 4,
                       image_size=(256, 256), multi_class=False,
                       num_workers: int = 2, augment_level='medium'):
    """
    Creates train, val, test DataLoaders for COCO-style datasets.
    Cleans invalid samples automatically.

    Args:
        data_root: Root directory containing train/valid/test subdirectories
        batch_size: Batch size for dataloaders
        image_size: Target image size (H, W)
        multi_class: Whether to use multi-class masks
        num_workers: Number of worker processes for data loading
        augment_level: Augmentation intensity for training ('light', 'medium', 'heavy')

    Returns:
        train_loader, val_loader, test_loader
    """
    subsets = ["train", "valid", "test"]
    dataloaders = {}

    for subset in subsets:
        subset_dir = os.path.join(data_root, subset)
        annotation_file = os.path.join(subset_dir, "_annotations.coco.json")

        if not os.path.exists(annotation_file):
            print(f"⚠️ Skipping {subset}: missing annotations file.")
            continue

        # Use training transforms for training set, validation transforms for val/test
        if subset == "train":
            transform = create_training_transforms(image_size, augment_level=augment_level)
            is_training = True
        else:
            transform = create_validation_transforms(image_size)
            is_training = False

        dataset = COCOSegmentationDataset(
            root_dir=subset_dir,
            annotation_file=annotation_file,
            transform=transform,
            multi_class=multi_class,
            is_training=is_training,
        )

        # Clean invalid samples (missing files, corrupted images)
        valid_samples = []
        for i in range(len(dataset)):
            try:
                img, mask = dataset[i]
                if img.shape[1:] != mask.shape[1:]:
                    print(f"⚠️ Skipping sample {i}: shape mismatch {img.shape} vs {mask.shape}")
                    continue
                valid_samples.append(i)
            except Exception as e:
                print(f"⚠️ Skipping invalid sample {i}: {e}")

        # Use Subset to include only valid indices
        if valid_samples:
            subset_data = torch.utils.data.Subset(dataset, valid_samples)
        else:
            subset_data = dataset

        dataloader = DataLoader(subset_data, batch_size=batch_size,
                                shuffle=(subset == "train"),
                                num_workers=num_workers, pin_memory=True)
        dataloaders[subset] = dataloader

    return dataloaders["train"], dataloaders["valid"], dataloaders["test"]
