# data/dataset.py
import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as T


class COCOSegmentationDataset(Dataset):
    """
    Dataset loader for COCO-style segmentation annotations.
    Compatible with pipelines for U-Net (image, mask) pairs.
    """

    def __init__(self, root_dir: str, annotation_file: str,
                 transform=None, target_transform=None, multi_class=False):
        """
        Args:
            root_dir: directory containing images
            annotation_file: path to _annotations.coco.json
            transform: transform to apply to image
            target_transform: transform to apply to mask
            multi_class: if True, produce multi-class masks (else binary)
        """
        super().__init__()
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.multi_class = multi_class

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

        # If user provided transforms, apply; otherwise convert to tensor
        if self.transform is not None:
            image = self.transform(image)
            # ToTensor() already normalizes to [0,1], so no need to divide by 255
            if not isinstance(image, torch.Tensor):
                image = T.ToTensor()(image)
        else:
            # Manual conversion without transforms
            image = torch.from_numpy(np.array(image)).float().permute(2, 0, 1)  # (H,W,3) -> (3,H,W)
            # Normalize to [0,1] only when transforms are None
            image = image / 255.0  # Maps [0, 255] -> [0, 1]

        # For mask: ensure we return a 1 x H x W tensor
        # CRITICAL: For binary segmentation, masks should be binary [0 or 1], not continuous [0,1]
        if self.target_transform is not None:
            mask = self.target_transform(mask)
            # if target_transform returned PIL, convert explicitly
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(np.array(mask)).float()
            else:
                mask = mask.float()
            
            # Ensure proper shape: (H, W) -> (1, H, W)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3 and mask.shape[0] != 1:
                mask = mask[0:1]
        else:
            mask = torch.from_numpy(np.array(mask)).float()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)  # (H, W) -> (1, H, W)
        
        # Binarize mask: values > 0.5 become 1.0, <= 0.5 become 0.0
        # This ensures binary segmentation masks are properly formatted after transforms
        mask = (mask > 0.5).float()

        return image, mask




def create_transforms(image_size=(256, 256), augment=False):
    """
    Basic image & mask transforms for segmentation tasks.

    Args:
        image_size: Target size (H, W)
        augment: If True, apply data augmentation (recommended for training set)
    """
    train_transforms = [
        T.Resize(image_size),
    ]

    # Fix #3: Add data augmentation for training
    # (Gentler augmentation for medical images)
    if augment:
        train_transforms.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            # Removed rotation and color jitter for medical images
        ])

    train_transforms.extend([
        T.ToTensor(),
        # No normalization! Using simple /255.0 in __getitem__ instead
        # This matches the TensorFlow implementation achieving IoU 0.90
    ])

    img_transform = T.Compose(train_transforms)

    mask_transform = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST)
    ])
    return img_transform, mask_transform


def create_dataloaders(data_root: str, batch_size: int = 4,
                       image_size=(256, 256), multi_class=False,
                       num_workers: int = 2, augment: bool = True):
    """
    Creates train, val, test DataLoaders for COCO-style datasets.
    Cleans invalid samples automatically.

    Args:
        augment: If True, apply data augmentation to training set (recommended: True)
    """
    subsets = ["train", "valid", "test"]
    dataloaders = {}

    for subset in subsets:
        subset_dir = os.path.join(data_root, subset)
        annotation_file = os.path.join(subset_dir, "_annotations.coco.json")

        if not os.path.exists(annotation_file):
            print(f"⚠️ Skipping {subset}: missing annotations file.")
            continue

        # Fix #3: Apply augmentation only to training set
        use_augment = (subset == "train") and augment
        img_transform, mask_transform = create_transforms(image_size, augment=use_augment)

        dataset = COCOSegmentationDataset(
            root_dir=subset_dir,
            annotation_file=annotation_file,
            transform=img_transform,
            target_transform=mask_transform,
            multi_class=multi_class,
        )

        # Clean invalid samples (missing files, corrupted images)
        valid_samples = []
        for i in range(len(dataset)):
            try:
                img, mask = dataset[i]
                if img.shape[1:] != mask.shape[1:]:
                    continue
                valid_samples.append(i)
            except Exception as e:
                print(f"Skipping invalid sample {i}: {e}")

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
