import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from preprocess import is_img, parse_fake_path

from models.Baselines.baselines import get_baseline_model


MODEL_TO_ID = {
    "SD": 0,
    "SDXL": 1,
    "FLUX1": 2,
    "REAL": 3,
    "Flux.1 S": 4,  # for civitai test set only
    "Illustraious": 5,  # for civitai test set only
}


class FakeImageDataset(Dataset):
    """
    This dataset will return repetitive fake images with unique mask labels by default.
    Use collapse_for_classification(records) method to avoid having repetitive images
    when training classification models.
    """

    def __init__(self, records, img_size=256, with_mask=True):
        self.records = records
        self.img_size = img_size
        self.with_mask = with_mask
        self.t_img = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),  # Convert to [0, 1]
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],  # ImageNet stats
                ),
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
        p = r["img_path"]

        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            H, W = self.img_size, self.img_size
            img = torch.zeros((3, H, W), dtype=torch.float32)
            return {
                "image": img,
                "label": -1,  # -1 indicates unknown label due to error
                "task": "invalid",
                "model_id": -1,  # unknown model
                "subset": "",
                "id": None,
                "mask": torch.zeros((1, H, W), dtype=torch.float32),
                "mask_label": None,
                "info_path": None,
                "img_path": p,
            }

        img = self.t_img(img)

        mask = None
        if self.with_mask and r["mask_path"] is not None:
            mask = Image.open(r["mask_path"]).convert("L")
            mask = self.t_mask(mask)  # (1, H, W)
        else:
            _, H, W = img.shape
            mask = torch.zeros((1, H, W), dtype=torch.float32)

        sample = {
            "image": img,  #  tensor (3, H, W)
            "label": 1,  # 1 indicates fake image
            "task": r["task"],  # "inpainting" or "txt2img"
            "model_name": r["model"],
            "model_id": MODEL_TO_ID[r["model"]],
            "subset": r["subset"],  # 0000
            "id": r["id"],  # 000000
            "mask": mask,  # tensor (1, H, W)
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
        valid_files = []
        for p in files:
            try:
                Image.open(p)
                valid_files.append(p)
            except Exception as e:
                pass
        self.paths = valid_files
        self.img_size = img_size
        self.t_img = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),  # Convert to [0, 1]
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]

        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            H, W = self.img_size, self.img_size
            img = torch.zeros((3, H, W), dtype=torch.float32)
            return {
                "image": img,
                "label": -1,  # -1 indicates unknown label due to error
                "task": "invalid",
                "model_id": -1,  # unknown model
                "subset": "",
                "id": None,
                "mask": torch.zeros((1, H, W), dtype=torch.float32),
                "mask_label": None,
                "info_path": None,
                "img_path": p,
            }

        img = self.t_img(img)
        subset = p.parent.stem
        id_ = p.stem
        _, H, W = img.shape
        mask = torch.zeros((1, H, W), dtype=torch.float32)  # mask of 0 for real images
        return {
            "image": img,
            "label": 0,  # 0 indicates real image
            "task": "real",
            "model_id": MODEL_TO_ID["REAL"],
            "subset": subset,
            "id": id_,
            "mask": mask,
            "mask_label": None,
            "info_path": None,
            "img_path": p,
        }


class CivitaiFakeDataset(Dataset):
    """
    Dataset for Civitai fake images without masks for test set only.
    """

    def __init__(self, civitai_root: Path, img_size=256):
        if not civitai_root.exists():
            print(
                f"Civitai root path does not exist: {civitai_root}, trying to use the fallback path.",
                flush=True,
            )
            civitai_root = Path(
                "/home/yz2483/scratch.gerstein/animel2m_dataset/civitai_subset/image"
            )
            if not civitai_root.exists():
                raise FileNotFoundError(
                    f"Fall back path does not exist: {civitai_root}, please follow the instructions to download the civitai test set."
                )

        files = [p for p in civitai_root.rglob("*") if p.is_file() and is_img(p)]
        valid_files = []
        for p in files:
            try:
                Image.open(p)
                valid_files.append(p)
            except Exception as e:
                pass
        self.paths = valid_files
        self.img_size = img_size
        self.t_img = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),  # Convert to [0, 1]
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]  # civitai_subset/image/<model_name>/<id>.jpeg

        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            H, W = self.img_size, self.img_size
            img = torch.zeros((3, H, W), dtype=torch.float32)
            return {
                "image": img,
                "label": -1,  # -1 indicates unknown label due to error
                "task": "invalid",
                "model_id": -1,  # unknown model
                "subset": "",
                "id": None,
                "mask": torch.zeros((1, H, W), dtype=torch.float32),
                "mask_label": None,
                "info_path": None,
                "img_path": p,
            }

        img = self.t_img(img)
        model_name = p.parent.stem
        _, H, W = img.shape
        mask = torch.zeros(
            (1, H, W), dtype=torch.float32
        )  # mask of 0 for civitai images
        return {
            "image": img,
            "label": 1,  # 1 indicates fake image
            "task": "civitai_test_fake",
            "model_id": MODEL_TO_ID[model_name],  # assuming SD for civitai
            "subset": "civitai",
            "id": None,
            "mask": mask,
            "mask_label": None,
            "info_path": None,
            "img_path": p,
        }


class AnimeIMDLDataModule(pl.LightningDataModule):
    """
    Handling both fake and real images using PyTorch Lightning DataModule.
    """

    def __init__(
        self,
        fake_root,
        real_root,
        civitai_root,
        img_size,
        batch_size,
        num_workers,
        train_val_split=0.8,
        pin_memory=True,
        persistent_workers=True,
        with_mask=True,  # whether to load masks for fake images
        fold=0,  # only for 5-fold CV
        seed=4710,
    ):
        super().__init__()
        self.fake_root = fake_root
        self.real_root = real_root
        self.civitai_root = civitai_root
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.persistent_workers = persistent_workers and num_workers > 0
        self.with_mask = with_mask
        if not (0 <= fold < 5):
            raise ValueError(
                "Fold must be between 0 and 4 for 5-fold cross-validation."
            )
        self.fold = fold
        self.seed = seed

        # Datasets will be created in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # check if paths exist
        if not Path(self.fake_root).exists():
            raise FileNotFoundError(f"Fake root path does not exist: {self.fake_root}")
        if not Path(self.real_root).exists():
            raise FileNotFoundError(f"Real root path does not exist: {self.real_root}")
        if not Path(self.civitai_root).exists():
            raise FileNotFoundError(
                f"Civitai root path does not exist: {self.civitai_root}"
            )

    def setup(self, stage=None):
        """
        split the dataset following 5-fold stratified cross-validation
        """
        if (
            self.train_dataset is not None
            and self.val_dataset is not None
            and self.test_dataset is not None
        ):
            return  # already set up

        # parse fake records
        parsed = parse_fake_path(self.fake_root, quiet=True)
        records_fakes = parsed["records"]

        # initiate datasets
        fake_dataset = FakeImageDataset(
            records_fakes, img_size=self.img_size, with_mask=self.with_mask
        )

        real_dataset = RealImageDataset(
            self.real_root,
            img_size=self.img_size,
        )

        civitai_dataset = CivitaiFakeDataset(
            self.civitai_root,
            img_size=self.img_size,
        )

        # [0, num_fake - 1] are fake indices
        # [num_fake, num_fake + num_real - 1] are real indices
        num_fake_total = len(fake_dataset)
        num_real_total = len(real_dataset)
        num_civitai_total = len(civitai_dataset)

        # split 1/3 real images for test set
        rng = np.random.RandomState(self.seed)
        desired_real_test = max(1, num_real_total // 3)

        num_real_test = min(
            desired_real_test, num_civitai_total
        )  # ensure we don't exceed civitai size

        # choose 1/3 real images or num_civitai_total whichever is smaller for test
        all_real_indices = np.arange(num_real_total)
        test_real_indices = rng.choice(
            all_real_indices, size=num_real_test, replace=False
        )
        trainval_real_indices = np.setdiff1d(all_real_indices, test_real_indices)

        # create subsets of real dataset
        real_trainval_dataset = torch.utils.data.Subset(
            real_dataset, trainval_real_indices
        )
        test_real_dataset = torch.utils.data.Subset(real_dataset, test_real_indices)

        num_real_trainval = len(real_trainval_dataset)

        # choose the same number of fake images from civitai for test set
        num_civitai_test = num_real_test
        all_civitai_indices = np.arange(num_civitai_total)
        test_civitai_indices = rng.choice(
            all_civitai_indices, size=num_civitai_test, replace=False
        )

        # create civitai test dataset
        civitai_test_dataset = torch.utils.data.Subset(
            civitai_dataset, test_civitai_indices
        )

        # use the remaining 2/3 real and all fake images for train/val split
        full_dataset = ConcatDataset([fake_dataset, real_trainval_dataset])

        num_fake = len(fake_dataset)
        num_real = len(real_trainval_dataset)

        labels = np.array([1] * num_fake + [0] * num_real)
        indices = np.arange(len(full_dataset))

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        folds = list(skf.split(indices, labels))

        val_fold = self.fold  # fold set for validation
        _, val_idx = folds[val_fold]

        # the other four folds are for training
        train_mask = np.ones(len(indices), dtype=bool)
        train_mask[val_idx] = False
        train_idx = indices[train_mask]

        self.train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        self.val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
        self.test_dataset = ConcatDataset([civitai_test_dataset, test_real_dataset])

        num_fake_train = np.sum(train_idx < num_fake)
        num_real_train = len(train_idx) - num_fake_train

        num_fake_val = np.sum(val_idx < num_fake)
        num_real_val = len(val_idx) - num_fake_val

        num_fake_test = len(civitai_test_dataset)
        num_real_test_final = len(test_real_dataset)

        print(
            f"[Fold {self.fold}] Train samples: {len(self.train_dataset)} "
            f"(Fake: {num_fake_train}, Real: {num_real_train})",
            flush=True,
        )
        print(
            f"[Fold {self.fold}] Validation samples: {len(self.val_dataset)} "
            f"(Fake: {num_fake_val}, Real: {num_real_val})",
            flush=True,
        )
        print(
            f"[Fold {self.fold}] Test samples: {len(self.test_dataset)} "
            f"(Fake: {num_fake_test}, Real: {num_real_test_final})",
            flush=True,
        )

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
    valid = [s for s in samples if s["label"] != -1]
    if len(valid) == 0:
        valid = samples  # all samples are invalid, proceed anyway
        print(f"[Warning] All samples in the batch are invalid.")

    images = torch.stack([s["image"] for s in samples], dim=0)
    labels = torch.tensor([s["label"] for s in samples], dtype=torch.float32)

    batch = {
        "image": images,
        "label": labels,
        "task": [s["task"] for s in samples],
        "model_id": torch.tensor(
            [s.get("model_id", -1) for s in samples], dtype=torch.int64
        ),
        "subset": [s["subset"] for s in samples],
        "id": [s["id"] for s in samples],
    }

    # for AniXplore training with masks
    if "mask" in samples[0] and samples[0]["mask"] is not None:
        batch["mask"] = torch.stack([s["mask"] for s in samples], dim=0)

    return batch
