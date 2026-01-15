# data/dataset_simple.py
"""
Dataset with simple normalization (/255.0) mirroring the TensorFlow implementation.
This is critical for achieving high performance.
"""
import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as T


class COCOSegmentationDatasetSimple(Dataset):
    """Dataset with simple /255.0 normalization"""
    def __init__(self, root_dir: str, annotation_file: str):
        super().__init__()
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def _load_image(self, image_id):
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")  # (H, W, 3) in [0, 255]
        return image, image_info

    def _load_mask(self, image_info, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann) * 1)

        return Image.fromarray(mask)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image, image_info = self._load_image(image_id)
        mask = self._load_mask(image_info, image_id)

        # Convert to tensor
        image = torch.from_numpy(np.array(image)).float()  # [0, 255] -> float
        image = image.permute(2, 0, 1) / 255.0  # (H, W, 3) -> (3, H, W) and normalize to [0, 1]

        mask = torch.from_numpy(np.array(mask)).float()  # [0, 255] -> float
        mask = mask / 255.0  # Normalize to [0, 1]
        mask = mask.unsqueeze(0)  # (H, W) -> (1, H, W)

        return image, mask


def create_dataloaders_simple(data_root: str, batch_size: int = 16,
                               image_size=(256, 256), num_workers: int = 2):
    """
    Create dataloaders with simple normalization (like the TensorFlow implementation)
    """
    subsets = ["train", "valid", "test"]
    dataloaders = {}

    for subset in subsets:
        subset_dir = os.path.join(data_root, subset)
        annotation_file = os.path.join(subset_dir, "_annotations.coco.json")

        if not os.path.exists(annotation_file):
            print(f"⚠️ Skipping {subset}: missing annotations file.")
            continue

        # Load images and get target size
        from PIL import Image
        sample_img = Image.open(os.path.join(subset_dir, os.listdir(subset_dir)[0]))
        orig_size = sample_img.size  # (W, H)

        # Use custom dataset with simple normalization
        dataset = COCOSegmentationDatasetSimple(
            root_dir=subset_dir,
            annotation_file=annotation_file
        )

        # Add resize transform
        class ResizeDataset(Dataset):
            def __init__(self, dataset, size):
                self.dataset = dataset
                self.size = size

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                img, mask = self.dataset[idx]

                # Resize using interpolation
                img = F.interpolate(img.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)
                mask = F.interpolate(mask.unsqueeze(0), size=self.size, mode='nearest').squeeze(0)

                return img, mask

        dataset = ResizeDataset(dataset, image_size)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(subset == "train"),
            num_workers=num_workers,
            pin_memory=True
        )
        dataloaders[subset] = dataloader

    return dataloaders["train"], dataloaders["valid"], dataloaders["test"]
