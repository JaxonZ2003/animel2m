import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path

from dataset import FakeImageDataset, RealImageDataset
from models.AniXplore.AniXplore import AniXplore
from preprocess import parse_fake_path


# 把 sample list -> batch tensor
def collate_fn(samples):
    """
    samples: list of dict
    每个样本里至少有: "image", "mask", "fake"
    我们要返回:
      - images: (B, 3, H, W)
      - masks : (B, 1, H, W)
      - labels: (B,)   其中 1=真图, 0=假图
    """
    # 所有图片叠在一起
    images = torch.stack([s["image"] for s in samples], dim=0)  # (B, 3, H, W)

    # 统一处理 mask
    masks = []
    for s in samples:
        if s["mask"] is None:
            # 没有 mask 的（真实图），用全 0 mask 代替
            _, H, W = s["image"].shape
            masks.append(torch.zeros(1, H, W, dtype=torch.float32))
        else:
            masks.append(s["mask"].float())
    masks = torch.stack(masks, dim=0)  # (B, 1, H, W)

    # 处理标签:
    # s["fake"] = 0 表示真图, 1 表示假图
    # 我们就直接用它作为 label，不做取反
    labels = torch.tensor([s["fake"] for s in samples], dtype=torch.float32)  # (B,)

    # 你要的话可以把其他信息也一起返回
    batch = {
        "image": images,
        "mask": masks,
        "label": labels,  # 0=real, 1=fake
        "fake": torch.tensor([s["fake"] for s in samples], dtype=torch.int64),
        "task": [s["task"] for s in samples],
        "model_id": torch.tensor([s["model_id"] for s in samples], dtype=torch.int64),
        "subset": [s["subset"] for s in samples],
        "id": [s["id"] for s in samples],
        "mask_label": [s["mask_label"] for s in samples],
    }
    return batch


def load_fake_records(fake_root):
    parsed = parse_fake_path(fake_root, quiet=True)
    records = parsed["records"]

    # 如果你只想要 inpainting，可以在这里筛一筛：
    # records = [r for r in records if r["task"] == "inpainting"]

    return records


def main():
    # ------------------------------
    # 2. 准备数据集 (假 + 真)
    # ------------------------------
    img_size = 512  # 要和 AniXplore 里的 image_size 对上
    real_root = Path(
        "/gpfs/milgram/scratch60/gerstein/yz2483/animel2m_dataset/real_images/resized_img"
    )
    fake_root = Path(
        "/gpfs/milgram/scratch60/gerstein/yz2483/animel2m_dataset/fake_images"
    )

    # 2.1 假图记录 records: 一般是你预处理好的一堆 dict 列表
    # 这里假设你已经有 records_fakes: List[dict]
    # 每个 dict 至少包含: img_path, mask_path, task, model, subset, id, mask_label
    # 这里我放一个占位, 你要替换成你自己的加载方式

    records_fakes = load_fake_records(fake_root)
    fake_dataset = FakeImageDataset(records_fakes, img_size=img_size, with_mask=True)
    real_dataset = RealImageDataset(real_root, img_size=img_size)
    train_dataset = ConcatDataset([fake_dataset, real_dataset])

    print(f"Fake dataset size: {len(fake_dataset)}", flush=True)
    print(f"Real dataset size: {len(real_dataset)}", flush=True)
    print(f"Total training dataset size: {len(train_dataset)}", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seg_pretrain_path = None

    model = AniXplore(
        seg_pretrain_path=seg_pretrain_path,
        conv_pretrain=True,
        image_size=img_size,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # num_epochs = 10

    # for epoch in range(num_epochs):
    #     model.train()
    #     running_loss = 0.0

    #     for step, batch in enumerate(train_loader):
    #         images = batch["image"].to(device)
    #         masks = batch["mask"].to(device)
    #         labels = batch["label"].to(device)

    #         out = model(images, masks, labels)
    #         loss = out["backward_loss"]

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()

    #         if (step + 1) % 20 == 0:
    #             avg_loss = running_loss / 20
    #             running_loss = 0.0
    #             pred_label = out["pred_label"].detach()
    #             acc = (pred_label == labels).float().mean().item()
    #             print(
    #                 f"Epoch [{epoch+1}/{num_epochs}] "
    #                 f"Step [{step+1}] "
    #                 f"Loss: {avg_loss:.4f} "
    #                 f"Cls Acc: {acc:.4f}"
    #             )

    #     torch.save(model.state_dict(), f"anixplore_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    main()
