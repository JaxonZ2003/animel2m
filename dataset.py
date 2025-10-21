import torch
import logging

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

from preprocess import is_img

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
