import torch
import pytorch_lightning as pl
import os
import argparse

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import auc, roc_auc_score, accuracy_score

from models.Baselines.baselines import *
from dataset import *
from models.AniXplore.AniXplore import AniXplore


class PrintEpochResultCallback(Callback):
    """在每个 Epoch 结束时打印 Train 和 Val 的所有指标"""

    def on_train_epoch_end(self, trainer, pl_module):
        # 从 trainer.callback_metrics 中获取所有 logged 的指标
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        # 使用 .get() 获取，如果取不到（比如第一个 epoch 还没算完）则默认 0.0
        train_loss = metrics.get("train_loss", 0.0)
        train_acc = metrics.get("train_acc", 0.0)
        train_auc = metrics.get("train_auc", 0.0)

        val_loss = metrics.get("val_loss", 0.0)
        val_acc = metrics.get("val_acc", 0.0)
        val_auc = metrics.get("val_auc", 0.0)

        print(
            f"Epoch {epoch} | "
            f"Train [L:{train_loss:.4f} Acc:{train_acc:.4f} AUC:{train_auc:.4f}] | "
            f"Val [L:{val_loss:.4f} Acc:{val_acc:.4f} AUC:{val_auc:.4f}]",
            flush=True,
        )

    def on_test_epoch_end(self, trainer, pl_module):
        """测试结束时打印 Test 指标"""
        metrics = trainer.callback_metrics
        test_loss = metrics.get("test_loss", 0.0)
        test_acc = metrics.get("test_acc", 0.0)
        test_auc = metrics.get("test_auc", 0.0)

        print(
            f"\n[TEST RESULT] Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | AUC: {test_auc:.4f}\n",
            flush=True,
        )


class BaseLitModule(pl.LightningModule):
    """父类，提取公共的 Metric 计算和 Logging 逻辑"""

    def _compute_and_log_metrics(self, outputs, stage="val"):
        if not outputs:
            return

        # 拼接所有 step 的结果
        preds = torch.cat([x["preds"] for x in outputs]).cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).cpu().numpy()
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        # 计算 Loss 平均值
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        # 计算 ACC
        # 注意：AniXplore 输出的是 0/1 float，Baseline 输出的是 logits
        if self.hparams.is_logits:
            pred_binary = preds > 0.5
        else:
            pred_binary = preds

        acc = accuracy_score(labels, pred_binary)

        # 计算 AUC
        try:
            auc = roc_auc_score(labels, preds)
        except ValueError:
            auc = 0.5  # 防止只有一个类别报错

        # 打印和记录 (logger=True 会写入 CSV)
        self.log(f"{stage}_loss", avg_loss, on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{stage}_acc", acc, on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{stage}_auc", auc, on_epoch=True, prog_bar=False, logger=True)

    def on_train_epoch_end(self):
        self._compute_and_log_metrics(self.training_step_outputs, stage="train")
        self.training_step_outputs.clear()  # 释放内存

    def on_validation_epoch_end(self):
        self._compute_and_log_metrics(self.validation_step_outputs, stage="val")
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        self._compute_and_log_metrics(self.test_step_outputs, stage="test")
        self.test_step_outputs.clear()


class BaselineLitModule(BaseLitModule):
    def __init__(self, model_name="convnext", lr=1e-4, max_epochs=10):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.is_logits = True  # 标记这是一个输出 logits 的模型
        self.model = get_baseline_model(model_name, pretrained=True)

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model.get_logits(x)

    def training_step(self, batch, batch_idx):
        output = self.model(batch["image"], label=batch["label"])
        loss = output["backward_loss"]
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch["image"])
        loss_dict = self.model(batch["image"], label=batch["label"])
        preds = torch.sigmoid(logits).squeeze()
        self.validation_step_outputs.append(
            {
                "preds": preds,
                "labels": batch["label"],
                "loss": loss_dict["backward_loss"],
            }
        )

    def on_validation_epoch_end(self):
        self._compute_and_log_metrics(self.validation_step_outputs, stage="val")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        logits = self.forward(batch["image"])
        loss_dict = self.model(batch["image"], label=batch["label"])
        preds = torch.sigmoid(logits).squeeze()
        self.test_step_outputs.append(
            {
                "preds": preds,
                "labels": batch["label"],
                "loss": loss_dict["backward_loss"],
            }
        )

    def on_test_epoch_end(self):
        self._compute_and_log_metrics(self.test_step_outputs, stage="test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return [optimizer], [scheduler]


class AniXploreLitModule(BaseLitModule):
    def __init__(self, seg_pretrain_path, lr=1e-4, max_epochs=10):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.is_logits = False  # AniXplore 输出的是处理过的 0/1
        self.model = AniXplore(seg_pretrain_path=seg_pretrain_path, conv_pretrain=True)

        self.validation_step_outputs = []
        self.test_step_outputs = []

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
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        masks = self._get_dummy_mask(batch["image"])
        output = self.model(batch["image"], masks, batch["label"])
        self.validation_step_outputs.append(
            {
                "preds": output["pred_label"],
                "labels": batch["label"],
                "loss": output["backward_loss"],
            }
        )

    def on_validation_epoch_end(self):
        self._compute_and_log_metrics(self.validation_step_outputs, stage="val")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        masks = self._get_dummy_mask(batch["image"])
        output = self.model(batch["image"], masks, batch["label"])
        self.test_step_outputs.append(
            {
                "preds": output["pred_label"],
                "labels": batch["label"],
                "loss": output["backward_loss"],
            }
        )

    def on_test_epoch_end(self):
        self._compute_and_log_metrics(self.test_step_outputs, stage="test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
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
    # 确定当前任务的名称 (用于文件夹命名)
    if args.mode == "baseline":
        run_name = args.model_name
    else:
        run_name = "anixplore"

    print(f"=== Task Name: {run_name} ===")

    # Checkpoint 回调: 保存 val_auc 最高的模型
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            "out", "checkpoint", run_name
        ),  # 路径: out/checkpoint/convnext/
        filename="{epoch}-{val_auc:.4f}",  # 文件名: epoch=x-val_auc=0.xxxx.ckpt
        monitor="val_auc",  # 监控指标
        mode="max",  # 越大越好
        save_top_k=1,  # 只保存最好的1个
        save_last=True,  # 同时也保存最后一个epoch的结果
        verbose=True,
    )

    # Logger: 保存 loss/acc/auc 到 CSV 文件
    # 最终文件位置: out/logs/<run_name>/version_0/metrics.csv
    logger = CSVLogger(save_dir="out", name="logs", version=run_name)

    # ==============================

    # 2. Data
    dm = FakeDetectionDataModule(
        fake_root=args.fake_root,
        real_root=args.real_root,
        img_size=224,
        batch_size=args.batch_size,
        num_workers=4,
    )

    # 3. Model
    if args.mode == "baseline":
        print(f"Initializing Baseline: {args.model_name}")
        model = BaselineLitModule(model_name=args.model_name, max_epochs=args.epochs)
    else:
        print(f"Initializing AniXplore")
        model = AniXploreLitModule(
            seg_pretrain_path=args.seg_path, max_epochs=args.epochs
        )

    # 4. Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.epochs,
        precision="16-mixed",
        # === 启用 Checkpoint 和 Logger ===
        enable_checkpointing=True,  # 必须为 True
        enable_progress_bar=False,  # 关闭默认进度条
        callbacks=[checkpoint_callback, print_callbacks],
        logger=logger,  # 传入 Logger
        # ==============================
        num_sanity_val_steps=0,
    )

    # 5. Train & Test
    print(f"=== Start Training ===")
    trainer.fit(model, datamodule=dm)

    print(f"Best Checkpoint Path: {checkpoint_callback.best_model_path}")

    print("\n=== Start Testing (using best checkpoint) ===")
    # test 会自动加载最好的 checkpoint (ckpt_path='best')
    trainer.test(model, datamodule=dm, ckpt_path="best")
