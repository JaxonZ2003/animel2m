import torch
import pytorch_lightning as pl
import os
import argparse

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import roc_auc_score, accuracy_score

# 确保路径正确，根据你的实际情况调整
from models.Baselines.baselines import get_baseline_model
from dataset import FakeDetectionDataModule
from models.AniXplore.AniXplore import AniXplore


class PrintEpochResultCallback(Callback):
    r"""
    Log train and val metrics at the end of each epoch.
    Log the test metrics at the end of testing.
    """

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        train_loss = metrics.get("train_loss", 0.0)
        train_acc = metrics.get("train_acc", 0.0)
        train_auc = metrics.get("train_auc", 0.0)

        val_loss = metrics.get("val_loss", 0.0)
        val_acc = metrics.get("val_acc", 0.0)
        val_auc = metrics.get("val_auc", 0.0)

        train_loss = (
            train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
        )
        train_acc = (
            train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc
        )
        train_auc = (
            train_auc.item() if isinstance(train_auc, torch.Tensor) else train_auc
        )
        val_loss = val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss
        val_acc = val_acc.item() if isinstance(val_acc, torch.Tensor) else val_acc
        val_auc = val_auc.item() if isinstance(val_auc, torch.Tensor) else val_auc

        print(
            f"[TRAIN] [L:{train_loss:.4f} Acc:{train_acc:.4f} AUC:{train_auc:.4f}] | Epoch {epoch}\n"
            f"[VALID] [L:{val_loss:.4f} Acc:{val_acc:.4f} AUC:{val_auc:.4f}] | Epoch {epoch}",
            flush=True,
        )

    def on_test_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        test_loss = metrics.get("test_loss", 0.0)
        test_acc = metrics.get("test_acc", 0.0)
        test_auc = metrics.get("test_auc", 0.0)

        test_loss = (
            test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss
        )
        test_acc = test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc
        test_auc = test_auc.item() if isinstance(test_auc, torch.Tensor) else test_auc

        print(
            f"\n[TEST ] Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | AUC: {test_auc:.4f}\n",
            flush=True,
        )


class BaseLitModule(pl.LightningModule):
    r"""
    Metrics computation and logging for train/val/test.
    """

    def __init__(self):
        super().__init__()
        # temporary storage for epoch outputs and logs
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _compute_and_log_metrics(self, outputs, stage="val"):
        if not outputs:
            return

        # concatenate all preds and labels
        preds = torch.cat([x["preds"] for x in outputs]).cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).cpu().numpy()
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        # Baselines output raw logits, AniXplore outputs 0 to 1 float
        if self.hparams.is_logits:
            pred_binary = (preds > 0.5).astype(int)
        else:
            pred_binary = preds.astype(int)

        # compute accuracy
        acc = accuracy_score(labels.astype(int), pred_binary)

        # compute AUC
        auc = roc_auc_score(labels, preds)

        # log the metrics
        self.log(f"{stage}_loss", avg_loss, prog_bar=False, logger=True)
        self.log(f"{stage}_acc", acc, prog_bar=False, logger=True)
        self.log(f"{stage}_auc", auc, prog_bar=False, logger=True)

    def on_train_epoch_end(self):
        self._compute_and_log_metrics(self.training_step_outputs, stage="train")
        self.training_step_outputs.clear()  # free memory

    def on_validation_epoch_end(self):
        self._compute_and_log_metrics(self.validation_step_outputs, stage="val")
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        self._compute_and_log_metrics(self.test_step_outputs, stage="test")
        self.test_step_outputs.clear()


# === 3. Baseline Model 实现 ===
class BaselineLitModule(BaseLitModule):
    def __init__(self, model_name="convnext", lr=1e-4, max_epochs=10, img_size=224):
        super().__init__()  # 必须调用父类 init
        self.save_hyperparameters()
        self.hparams.is_logits = True
        # 传入 img_size 修复 ViT 等模型报错
        self.model = get_baseline_model(model_name, pretrained=True, img_size=img_size)

    def forward(self, x):
        return self.model.get_logits(x)

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch["image"])
        loss_dict = self.model(batch["image"], label=batch["label"])
        loss = loss_dict["backward_loss"]

        preds = torch.sigmoid(logits).squeeze()

        # 收集数据用于 Epoch 计算，必须 detach 避免显存爆炸
        self.training_step_outputs.append(
            {
                "preds": preds.detach().cpu(),
                "labels": batch["label"].detach().cpu(),
                "loss": loss.detach().cpu(),
            }
        )
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch["image"])
        loss_dict = self.model(batch["image"], label=batch["label"])
        preds = torch.sigmoid(logits).squeeze()

        self.validation_step_outputs.append(
            {
                "preds": preds.cpu(),
                "labels": batch["label"].cpu(),
                "loss": loss_dict["backward_loss"].cpu(),
            }
        )

    def test_step(self, batch, batch_idx):
        logits = self.forward(batch["image"])
        loss_dict = self.model(batch["image"], label=batch["label"])
        preds = torch.sigmoid(logits).squeeze()

        self.test_step_outputs.append(
            {
                "preds": preds.cpu(),
                "labels": batch["label"].cpu(),
                "loss": loss_dict["backward_loss"].cpu(),
            }
        )

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return [optimizer], [scheduler]


# === 4. AniXplore Model 实现 ===
class AniXploreLitModule(BaseLitModule):
    def __init__(self, seg_pretrain_path, lr=1e-4, max_epochs=10, img_size=224):
        super().__init__()  # 必须调用父类 init
        self.save_hyperparameters()
        self.hparams.is_logits = False
        # 传入 image_size 修复尺寸不匹配问题
        self.model = AniXplore(
            seg_pretrain_path=seg_pretrain_path, conv_pretrain=True, image_size=img_size
        )

    def _get_dummy_mask(self, images):
        return torch.zeros(
            (images.shape[0], 1, images.shape[2], images.shape[3]), device=self.device
        )

    def training_step(self, batch, batch_idx):
        masks = (
            batch["mask"]
            if ("mask" in batch and batch["mask"] is not None)
            else self._get_dummy_mask(batch["image"])
        )
        output = self.model(batch["image"], masks, batch["label"])
        loss = output["backward_loss"]

        self.training_step_outputs.append(
            {
                "preds": output["pred_label"].detach().cpu(),
                "labels": batch["label"].detach().cpu(),
                "loss": loss.detach().cpu(),
            }
        )
        return loss

    def validation_step(self, batch, batch_idx):
        masks = self._get_dummy_mask(batch["image"])
        output = self.model(batch["image"], masks, batch["label"])

        self.validation_step_outputs.append(
            {
                "preds": output["pred_label"].cpu(),
                "labels": batch["label"].cpu(),
                "loss": output["backward_loss"].cpu(),
            }
        )

    def test_step(self, batch, batch_idx):
        masks = self._get_dummy_mask(batch["image"])
        output = self.model(batch["image"], masks, batch["label"])

        self.test_step_outputs.append(
            {
                "preds": output["pred_label"].cpu(),
                "labels": batch["label"].cpu(),
                "loss": output["backward_loss"].cpu(),
            }
        )

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr)


# === 5. Main Execution ===
if __name__ == "__main__":
    IMG_SIZE = 224
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="baseline", choices=["baseline", "anixplore"]
    )
    parser.add_argument(
        "--model_name", type=str, default="convnext", help="For baseline mode only"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--fake_root",
        type=str,
        default="/gpfs/milgram/scratch60/gerstein/yz2483/animel2m_dataset/fake_images",
    )
    parser.add_argument(
        "--real_root",
        type=str,
        default="/gpfs/milgram/scratch60/gerstein/yz2483/animel2m_dataset/real_images/resized_img",
    )
    parser.add_argument(
        "--seg_path",
        type=str,
        default="./segformer_mit-b0.pth",
        help="Path to SegFormer weights",
    )

    args = parser.parse_args()

    pl.seed_everything(42)

    print_callbacks = PrintEpochResultCallback()

    # === 1. 配置输出路径和 Log 名称 ===
    if args.mode == "baseline":
        run_name = args.model_name
    else:
        run_name = "anixplore"

    print(f"=== Task Name: {run_name} ===")

    # Checkpoint 回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("out", "checkpoint", run_name),
        filename="{epoch}-{val_auc:.4f}",
        monitor="val_auc",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=False,
    )

    # Logger: 保存 csv
    logger = CSVLogger(save_dir="out", name="logs", version=run_name)

    # 2. Data
    dm = FakeDetectionDataModule(
        fake_root=args.fake_root,
        real_root=args.real_root,
        img_size=IMG_SIZE,
        batch_size=args.batch_size,
        num_workers=4,
    )

    # 3. Model
    if args.mode == "baseline":
        print(f"Initializing Baseline: {args.model_name}")
        model = BaselineLitModule(
            model_name=args.model_name, max_epochs=args.epochs, img_size=IMG_SIZE
        )
    else:
        print(f"Initializing AniXplore")
        model = AniXploreLitModule(
            seg_pretrain_path=args.seg_path, max_epochs=args.epochs, img_size=IMG_SIZE
        )

    # 4. Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.epochs,
        precision="16-mixed",
        enable_checkpointing=True,
        enable_progress_bar=False,
        callbacks=[checkpoint_callback, print_callbacks],
        logger=logger,
        num_sanity_val_steps=0,
    )

    # 5. Train
    print(f"=== Start Training ===")
    trainer.fit(model, datamodule=dm)

    print(f"Best Checkpoint Path: {checkpoint_callback.best_model_path}")

    # 6. Test
    print("\n=== Start Testing (using best checkpoint) ===")
    trainer.test(model, datamodule=dm, ckpt_path="best")
