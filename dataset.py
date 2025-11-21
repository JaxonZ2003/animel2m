import torch
import torch.nn as nn
import torch.optim as optim
import logging
import json
import wandb  # Optional: for experiment tracking
import argparse
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from preprocess import is_img

from preprocess import parse_fake_path
from baseline_models import get_baseline_model


MODEL_TO_ID = {"SD": 0, "SDXL": 1, "FLUX1": 2, "REAL": 3}


class FakeImageDataset(Dataset):
    """
    This dataset will return repetitive fake images with unique mask labels by default.
    Please use collapse_for_classification(records) method to avoid having repetitive images
    when training classification models.
    """

    def __init__(self, records, img_size=256, with_mask=True):
        self.records = records
        self.with_mask = with_mask
        self.t_img = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),  # Convert to [0, 1]
            ]
        )
        self.t_mask = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),  # Convert to [0, 1]
            ]
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(r["img_path"]).convert("RGB")
        img = self.t_img(img)

        mask = None
        if self.with_mask and r["mask_path"] is not None:
            mask = Image.open(r["mask_path"]).convert("L")
            mask = self.t_mask(mask)

        sample = {
            "image": img,  # Tensor of shape (3, H, W)
            "fake": 1,  # 1 indicates fake image
            "task": r["task"],  # "inpainting" or "txt2img"
            "model_name": r["model"],
            "model_id": MODEL_TO_ID[r["model"]],
            "subset": r["subset"],  # 0000
            "id": r["id"],  # 000000
            "mask": mask,  # Tensor of shape (1, H, W) or None
            "mask_label": r["mask_label"],  # objects
        }
        return sample


class RealImageDataset(Dataset):
    """
    Dataset for real images.
    """

    def __init__(self, real_root: Path, img_size=256):
        if real_root.name != "resized_img":
            logging.warning(
                "Assuming real images are stored in 'resized_img' subdirectory."
            )
            real_root = real_root / "resized_img"

        files = [p for p in real_root.rglob("*") if p.is_file() and is_img(p)]
        self.paths = files
        self.t_img = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),  # Convert to [0, 1]
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        img = self.t_img(img)
        subset = p.parent.stem
        id_ = p.stem
        return {
            "image": img,
            "fake": 0,  # 0 indicates real image
            "task": "real",
            "model_id": MODEL_TO_ID["REAL"],
            "subset": subset,
            "id": id_,
            "mask": None,
            "mask_label": None,
            "info_path": None,
            "img_path": p,
        }


class SimpleFakeDataset(Dataset):
    """
    Simplified dataset that only returns images and labels (no masks).
    For fake images dataset.
    """

    def __init__(self, records, img_size=512, augment=False):
        self.records = records
        self.img_size = img_size
        self.augment = augment

        # Base transforms
        self.base_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Augmentation transforms
        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.1),
                    transforms.RandomRotation(degrees=10),
                    # transforms.ColorJitter(
                    #     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    # ),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = self.base_transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        # Load image
        img = Image.open(record["img_path"]).convert("RGB")
        img = self.transform(img)

        return {
            "image": img,
            "label": 1.0,  # Fake = 1
            "fake": 1,
            "task": record.get("task", "unknown"),
            "model_id": record.get("model", 0),
            "subset": record.get("subset", "unknown"),
            "id": record.get("id", idx),
        }


class SimpleRealDataset(Dataset):
    """
    Simplified dataset that only returns images and labels (no masks).
    For real images dataset.
    """

    def __init__(self, real_root, img_size=512, augment=False):
        self.real_root = Path(real_root)
        self.img_size = img_size
        self.augment = augment

        # Get all image paths
        self.image_paths = (
            list(self.real_root.glob("*.jpg"))
            + list(self.real_root.glob("*.png"))
            + list(self.real_root.glob("*.jpeg"))
            + list(self.real_root.glob("*.JPEG"))
            + list(self.real_root.glob("*.JPG"))
        )

        if not self.image_paths:
            raise ValueError(f"No images found in {real_root}")

        # Base transforms
        self.base_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Augmentation transforms
        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.1),
                    transforms.RandomRotation(degrees=10),
                    # transforms.ColorJitter(
                    #     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    # ),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = self.base_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return {
            "image": img,
            "label": 0.0,  # Real = 0
            "fake": 0,
            "task": "real",
            "model_id": -1,
            "subset": "real",
            "id": str(img_path.stem),
        }


class FakeDetectionDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for fake image detection.
    Handles all data loading logic.
    """

    def __init__(
        self,
        fake_root,
        real_root,
        img_size,
        batch_size,
        num_workers,
        task_filter=None,
        train_val_split=0.8,
        augment_train=True,
        pin_memory=True,
        persistent_workers=True,
        max_fake_samples=None,
        max_real_samples=None,
    ):
        super().__init__()
        self.fake_root = fake_root
        self.real_root = real_root
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task_filter = task_filter
        self.train_val_split = train_val_split
        self.augment_train = augment_train
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.persistent_workers = persistent_workers and num_workers > 0
        self.max_fake_samples = max_fake_samples
        self.max_real_samples = max_real_samples

        # Datasets will be created in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """
        Download data if needed. Called only on 1 GPU/TPU in distributed.
        """
        # Check if paths exist
        if not Path(self.fake_root).exists():
            raise ValueError(f"Fake root path does not exist: {self.fake_root}")
        if not Path(self.real_root).exists():
            raise ValueError(f"Real root path does not exist: {self.real_root}")

    def setup(self, stage=None):
        """
        Data operations on every GPU/TPU.
        Split data, apply transforms, etc.
        """
        # Parse fake records
        parsed = parse_fake_path(self.fake_root, quiet=True)
        records_fakes = parsed["records"]

        # Filter by task if specified
        if self.task_filter:
            records_fakes = [r for r in records_fakes if r["task"] == self.task_filter]
            print(
                f"Filtered to {len(records_fakes)} fake samples for task: {self.task_filter}"
            )

        # Limit fake samples if specified
        if self.max_fake_samples and len(records_fakes) > self.max_fake_samples:
            records_fakes = records_fakes[: self.max_fake_samples]
            print(f"Limited to {self.max_fake_samples} fake samples")

        # Create datasets with augmentation for training
        fake_dataset_train = SimpleFakeDataset(
            records_fakes, img_size=self.img_size, augment=self.augment_train
        )
        fake_dataset_val = SimpleFakeDataset(
            records_fakes, img_size=self.img_size, augment=False
        )

        real_dataset_train = SimpleRealDataset(
            self.real_root, img_size=self.img_size, augment=self.augment_train
        )
        real_dataset_val = SimpleRealDataset(
            self.real_root, img_size=self.img_size, augment=False
        )

        # Balance dataset if needed
        # if self.balance_dataset:
        #     min_samples = min(len(fake_dataset_train), len(real_dataset_train))
        #     if self.max_real_samples:
        #         min_samples = min(min_samples, self.max_real_samples)

        #     # Randomly sample to balance
        #     if len(fake_dataset_train) > min_samples:
        #         indices = np.random.choice(
        #             len(fake_dataset_train), min_samples, replace=False
        #         )
        #         fake_dataset_train = torch.utils.data.Subset(
        #             fake_dataset_train, indices
        #         )
        #         fake_dataset_val = torch.utils.data.Subset(fake_dataset_val, indices)

        #     if len(real_dataset_train) > min_samples:
        #         indices = np.random.choice(
        #             len(real_dataset_train), min_samples, replace=False
        #         )
        #         real_dataset_train = torch.utils.data.Subset(
        #             real_dataset_train, indices
        #         )
        #         real_dataset_val = torch.utils.data.Subset(real_dataset_val, indices)

        # Split datasets
        if stage == "fit" or stage is None:
            # Training/validation split
            fake_train_size = int(self.train_val_split * len(fake_dataset_train))
            fake_val_size = len(fake_dataset_train) - fake_train_size

            # Use the augmented dataset for training split
            fake_train_indices = list(range(fake_train_size))
            fake_val_indices = list(
                range(fake_train_size, fake_train_size + fake_val_size)
            )

            fake_train = torch.utils.data.Subset(fake_dataset_train, fake_train_indices)
            fake_val = torch.utils.data.Subset(fake_dataset_val, fake_val_indices)

            real_train_size = int(self.train_val_split * len(real_dataset_train))
            real_val_size = len(real_dataset_train) - real_train_size

            real_train_indices = list(range(real_train_size))
            real_val_indices = list(
                range(real_train_size, real_train_size + real_val_size)
            )

            real_train = torch.utils.data.Subset(real_dataset_train, real_train_indices)
            real_val = torch.utils.data.Subset(real_dataset_val, real_val_indices)

            # Combine datasets
            self.train_dataset = ConcatDataset([fake_train, real_train])
            self.val_dataset = ConcatDataset([fake_val, real_val])

            print(
                f"Training samples: {len(self.train_dataset)} "
                f"(Fake: {len(fake_train)}, Real: {len(real_train)})"
            )
            print(
                f"Validation samples: {len(self.val_dataset)} "
                f"(Fake: {len(fake_val)}, Real: {len(real_val)})"
            )

        if stage == "test" or stage is None:
            # For testing, use validation set or create a separate test set
            if self.val_dataset is not None:
                self.test_dataset = self.val_dataset
            else:
                # Create test dataset from all data
                fake_dataset_test = SimpleFakeDataset(
                    records_fakes, img_size=self.img_size, augment=False
                )
                real_dataset_test = SimpleRealDataset(
                    self.real_root, img_size=self.img_size, augment=False
                )

                # Take a subset for testing
                test_size = min(1000, len(fake_dataset_test), len(real_dataset_test))
                fake_test = torch.utils.data.Subset(
                    fake_dataset_test,
                    np.random.choice(len(fake_dataset_test), test_size, replace=False),
                )
                real_test = torch.utils.data.Subset(
                    real_dataset_test,
                    np.random.choice(len(real_dataset_test), test_size, replace=False),
                )

                self.test_dataset = ConcatDataset([fake_test, real_test])
                print(f"Test samples: {len(self.test_dataset)}")

    def train_dataloader(self):
        """Return training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=simple_collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        """Return validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=simple_collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        """Return test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=simple_collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=False,  # Don't keep workers for test
        )

    def predict_dataloader(self):
        """Return prediction dataloader (same as test)"""
        return self.test_dataloader()


def collapse_for_classification(records):
    """
    Collapse records to have unique images for classification tasks.
    For each unique image, we keep only one record (the first one encountered).
    """
    groups = {}
    for r in records:
        key = (r["subset"], r["id"], r["task"], r["model"])
        g = groups.setdefault(
            key,
            {
                "subset": r["subset"],
                "id": r["id"],
                "task": r["task"],
                "model": r["model"],
                "img_path": [],
                "masks": [],  # [(mask_path, mask_label)]
                "info": r.get("info"),
            },
        )
        g["img_path"].append(r["img_path"])
        if r["mask_path"] is not None:
            g["masks"].append((r["mask_path"], r["mask_label"]))
    return list(groups.values())


def simple_collate_fn(samples):
    """Simple collate function for baseline models (no masks needed)"""
    images = torch.stack([s["image"] for s in samples], dim=0)
    labels = torch.tensor([s["label"] for s in samples], dtype=torch.float32)

    batch = {
        "image": images,
        "label": labels,
        "fake": torch.tensor([s["fake"] for s in samples], dtype=torch.int64),
        "task": [s["task"] for s in samples],
        "model_id": torch.tensor(
            [s.get("model_id", -1) for s in samples], dtype=torch.int64
        ),
        "subset": [s["subset"] for s in samples],
        "id": [s["id"] for s in samples],
    }
    return batch
