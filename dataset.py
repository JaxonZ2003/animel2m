import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

MODEL_TO_ID = {"SD": 0, "SDXL": 1, "FLUX1": 2, "REAL": 3}


class FakeImageDataset(Dataset):
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
