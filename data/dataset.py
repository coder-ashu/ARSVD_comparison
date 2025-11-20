# data/dataset.py
import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as T
from torchvision import transforms, datasets


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
            # if transform returns PIL (unlikely), convert
            if not isinstance(image, torch.Tensor):
                image = image.ToTensor()
        else:
            image = image.ToTensor()

        # For mask: ensure we return a 1 x H x W tensor of dtype long (for class ids)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
            # if target_transform returned PIL, convert explicitly
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(np.array(mask)).long().unsqueeze(0)
            else:
                # ensure type & shape
                if mask.ndim == 2:
                    mask = mask.long().unsqueeze(0)
                elif mask.ndim == 3 and mask.shape[0] != 1:
                    # if mask is CxHxW, reduce to single channel if necessary
                    mask = mask[0:1].long()
        else:
            mask = torch.from_numpy(np.array(mask)).long().unsqueeze(0)

        return image, mask




def create_transforms(image_size=(256, 256)):
    """
    Basic image & mask transforms for segmentation tasks.
    """
    img_transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    mask_transform = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST)
    ])
    return img_transform, mask_transform


def create_dataloaders(data_root: str, batch_size: int = 4,
                       image_size=(256, 256), multi_class=False,
                       num_workers: int = 2):
    """
    Creates train, val, test DataLoaders for COCO-style datasets.
    Cleans invalid samples automatically.
    """
    subsets = ["train", "valid", "test"]
    dataloaders = {}

    img_transform, mask_transform = create_transforms(image_size)

    for subset in subsets:
        subset_dir = os.path.join(data_root, subset)
        annotation_file = os.path.join(subset_dir, "_annotations.coco.json")

        if not os.path.exists(annotation_file):
            print(f"⚠️ Skipping {subset}: missing annotations file.")
            continue

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

# ---- append to data/dataset.py ----


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

def get_cifar10_loaders(data_dir: str, batch_size: int = 128,
                        num_workers: int = 4, shuffle_train: bool = True):
    """
    Returns train, val, test DataLoaders for CIFAR-10.
    Expects data_dir to be the directory where CIFAR data will be downloaded/stored.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_ds  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    # Optionally split train into train/valid (common). Here we create a small val split (e.g. 5k).
    val_size = 5000
    if len(train_ds) > val_size:
        train_subset, val_subset = torch.utils.data.random_split(train_ds, [len(train_ds)-val_size, val_size])
    else:
        train_subset, val_subset = train_ds, test_ds  # fallback: use test as val if too small

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle_train,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def create_dataloaders(data_root: str, batch_size: int = 4,
                       image_size=(256, 256), multi_class=False,
                       num_workers: int = 2, dataset_type: str = "coco"):
    """
    Unified factory:
      dataset_type: "coco" (default) or "cifar10"
    Returns: train_loader, val_loader, test_loader
    """
    dataset_type = dataset_type.lower()
    if dataset_type == "cifar10":
        # data_root is root directory for CIFAR download/cache
        return get_cifar10_loaders(data_root, batch_size=max(batch_size, 8), num_workers=num_workers)
    elif dataset_type == "coco":
        # old behavior (segmentation)
        subsets = ["train", "valid", "test"]
        dataloaders = {}

        img_transform, mask_transform = create_transforms(image_size)

        for subset in subsets:
            subset_dir = os.path.join(data_root, subset)
            annotation_file = os.path.join(subset_dir, "_annotations.coco.json")

            if not os.path.exists(annotation_file):
                print(f"⚠️ Skipping {subset}: missing annotations file.")
                continue

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

        # keep same return signature as before (train, valid, test)
        return dataloaders.get("train", None), dataloaders.get("valid", None), dataloaders.get("test", None)
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}. Choose 'coco' or 'cifar10'.")
