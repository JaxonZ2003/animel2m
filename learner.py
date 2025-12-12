import torch
import pytorch_lightning as pl
import numpy as np
import os
import argparse

from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import roc_auc_score, accuracy_score

from models.Baselines.baselines import get_baseline_model
from dataset import AnimeIMDLDataModule
from models.AniXplore.AniXplore import AniXplore

IMG_SIZE = 512
EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-4
SEED = 4710


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

        if np.isnan(preds).any():
            num_nan = np.isnan(preds).sum()
            print(
                f"[WARNING] [{stage}] Found {num_nan} NaN predictions. "
                f"Will drop them before computing metrics.",
                flush=True,
            )
            valid = ~np.isnan(preds)
            preds = preds[valid]
            labels = labels[valid]

            if preds.size == 0:
                print(
                    f"[ERROR] [{stage}] No valid predictions left after dropping NaNs. "
                    f"Skipping metric computation.",
                    flush=True,
                )
                self.log(f"{stage}_loss", avg_loss, prog_bar=False, logger=True)
                return
        # Baselines output raw logits: we have converted it to probs already
        # AniXplore outputs class probabilities directly
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

        if stage == "test":
            loss_float = (
                avg_loss.item()
                if isinstance(avg_loss, torch.Tensor)
                else float(avg_loss)
            )
            print(
                f"\n[TEST ] Loss: {loss_float:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}\n",
                flush=True,
            )

    def on_train_epoch_end(self):
        self._compute_and_log_metrics(self.training_step_outputs, stage="train")
        self.training_step_outputs.clear()  # free memory

    def on_validation_epoch_end(self):
        self._compute_and_log_metrics(self.validation_step_outputs, stage="val")
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        self._compute_and_log_metrics(self.test_step_outputs, stage="test")
        self.test_step_outputs.clear()


class BaselineLitModule(BaseLitModule):
    def __init__(self, model_name="convnext", lr=LR, max_epochs=EPOCHS, img_size=224):
        super().__init__()  # call parent init for metric storage
        self.save_hyperparameters()
        self.hparams.is_logits = True  # baseline models output raw logits
        # pass img_size to fix errors in ViT and other models
        self.model = get_baseline_model(model_name, pretrained=True, img_size=img_size)

    def forward(self, x):
        return self.model.get_logits(x)

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch["image"])
        loss_dict = self.model(batch["image"], label=batch["label"])
        loss = loss_dict["backward_loss"]

        preds = torch.sigmoid(logits).view(-1)

        # for epoch metrics logging
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
        preds = torch.sigmoid(logits).view(-1)

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
        preds = torch.sigmoid(logits).view(-1)

        self.test_step_outputs.append(
            {
                "preds": preds.cpu(),
                "labels": batch["label"].cpu(),
                "loss": loss_dict["backward_loss"].cpu(),
            }
        )

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=5e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return [optimizer], [scheduler]


class AniXploreLitModule(BaseLitModule):
    def __init__(
        self,
        seg_pretrain_path,
        lr=1e-4,
        max_epochs=10,
        img_size=224,
        adv_training=False,
        adv_eps=4 / 255,
        adv_alpha=1 / 255,
        adv_steps=3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.is_logits = True  # AniXplore outputs class probabilities directly
        self.model = AniXplore(
            seg_pretrain_path=seg_pretrain_path, conv_pretrain=True, image_size=img_size
        )

        # adversarial training parameters
        self.adv_training = adv_training
        self.adv_eps = adv_eps
        self.adv_alpha = adv_alpha
        self.adv_steps = adv_steps

    def pgd_attack(self, images, masks, labels):
        r"""
        Perform PGD attack on the input images.
        The masks remain unchanged.
        """
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(
            1, 3, 1, 1
        )  # imagenet std

        eps_norm = self.adv_eps / std
        alpha_norm = self.adv_alpha / std

        x_orig = images.detach()

        x_adv = x_orig.clone().detach()
        # initialize with random noise
        x_adv = x_adv + torch.zeros_like(x_adv).uniform_(-eps_norm, eps_norm)
        x_adv = torch.max(torch.min(x_adv, x_orig + eps_norm), x_orig - eps_norm)
        # ensure within valid pixel range after normalization
        x_adv = torch.clamp(x_adv, x_orig.min(), x_orig.max())
        x_adv.requires_grad = True

        for _ in range(self.adv_steps):
            self.model.zero_grad()

            out_adv = self.model(x_adv, masks, labels)
            loss_adv = out_adv["backward_loss"]

            # gradients of loss w.r.t. adversarial images
            loss_adv.backward(retain_graph=False)
            grad = x_adv.grad.detach()

            x_adv = x_adv + alpha_norm * grad.sign()
            # maps back to [x_orig - eps, x_orig + eps]
            x_adv = torch.max(torch.min(x_adv, x_orig + eps_norm), x_orig - eps_norm)
            # ensure within valid pixel range after normalization
            x_adv = torch.clamp(x_adv, x_orig.min(), x_orig.max())

            # prepare for next step
            x_adv = x_adv.detach()
            x_adv.requires_grad = True

        return x_adv.detach()

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"].to(self.device)
        labels = batch["label"]

        if self.adv_training:
            images_adv = self.pgd_attack(images, masks, labels)
            output = self.model(images_adv, masks, labels)
        else:
            output = self.model(images, masks, labels)

        loss = output["backward_loss"]

        self.training_step_outputs.append(
            {
                "preds": output["pred_prob"].detach().cpu(),
                "labels": batch["label"].detach().cpu(),
                "loss": loss.detach().cpu(),
            }
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"].to(self.device)
        labels = batch["label"]

        output = self.model(images, masks, labels)

        self.validation_step_outputs.append(
            {
                "preds": output["pred_prob"].cpu(),
                "labels": batch["label"].cpu(),
                "loss": output["backward_loss"].cpu(),
            }
        )

    def test_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"].to(self.device)
        labels = batch["label"]

        output = self.model(images, masks, labels)

        self.test_step_outputs.append(
            {
                "preds": output["pred_prob"].cpu(),
                "labels": batch["label"].cpu(),
                "loss": output["backward_loss"].cpu(),
            }
        )

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=5e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="baseline", choices=["baseline", "anixplore"]
    )
    parser.add_argument(
        "--model_name", type=str, default="convnext", help="For baseline mode only"
    )
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
        "--civitai_root",
        type=str,
        default="/gpfs/milgram/scratch60/gerstein/yz2483/animel2m_dataset/civitai_subset/image",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold index for cross-validation (0-4)",
    )
    parser.add_argument(
        "--seg_path",
        type=str,
        default="./segformer_mit-b0.pth",
        help="Path to SegFormer weights",
    )
    parser.add_argument(
        "--test_only", action="store_true", help="If set, only run testing"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Checkpoint path to load for test_only mode.",
    )

    args = parser.parse_args()

    base_dir = os.path.join("out", f"seed{SEED}_fold{args.fold}")
    run_name = args.model_name if args.mode == "baseline" else "anixplore"
    print(f"=== Task Name: {run_name} ===", flush=True)

    pl.seed_everything(SEED)

    print_callbacks = PrintEpochResultCallback()

    logger = CSVLogger(save_dir=os.path.join(base_dir, "logs"), name=run_name)

    dm = AnimeIMDLDataModule(
        fake_root=args.fake_root,
        real_root=Path(args.real_root),
        civitai_root=Path(args.civitai_root),
        fold=args.fold,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=4,
        train_val_split=0.8,
        seed=SEED,
        with_mask=(args.mode == "anixplore"),
    )

    if args.test_only:
        if args.ckpt_path is None:
            raise ValueError("Please provide --ckpt_path for test_only mode.")

        if args.mode == "baseline":
            model = BaselineLitModule.load_from_checkpoint(args.ckpt_path)
        else:
            model = AniXploreLitModule.load_from_checkpoint(args.ckpt_path)

        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            precision="16-mixed",
            enable_checkpointing=False,
            enable_progress_bar=False,
            callbacks=[print_callbacks],
            logger=logger,
            num_sanity_val_steps=0,
        )

        print(
            f"=== TEST ONLY MODE ===\n"
            f"Loaded checkpoint: {args.ckpt_path}\n"
            f"Mode: {args.mode} | Fold: {args.fold}",
            flush=True,
        )
        print("\n=== Start Testing (test_only) ===", flush=True)
        trainer.test(model, datamodule=dm)

    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(base_dir, "checkpoint", run_name),
            filename="{epoch}-{val_auc:.4f}",
            monitor="val_auc",
            mode="max",
            save_top_k=1,
            save_last=True,
            verbose=False,
        )
        early_stopping = EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=10,
            min_delta=0.001,
            verbose=True,
        )
        model = (
            BaselineLitModule(
                model_name=args.model_name, max_epochs=EPOCHS, lr=LR, img_size=IMG_SIZE
            )
            if args.mode == "baseline"
            else AniXploreLitModule(
                seg_pretrain_path=args.seg_path,
                max_epochs=EPOCHS,
                lr=LR,
                img_size=IMG_SIZE,
            )
        )

        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=EPOCHS,
            precision="16-mixed",
            enable_checkpointing=True,
            enable_progress_bar=False,
            callbacks=[checkpoint_callback, print_callbacks, early_stopping],
            logger=logger,
            num_sanity_val_steps=0,
        )

        print(
            f"=== Set up complete ===\n"
            f"Model {args.model_name} | Max Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | Learning Rate: {LR}\n"
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} Trainable Parameters.\n"
            f"Dataset with fold {args.fold} | Seed {SEED}\n"
            f"Results will be save to {os.path.join(base_dir, 'logs', run_name)}",
            flush=True,
        )

        print(f"=== Start Training ===", flush=True)
        trainer.fit(model, datamodule=dm)

        print(
            f"Best Checkpoint Path: {checkpoint_callback.best_model_path}", flush=True
        )

        print("\n=== Start Testing (using best checkpoint) ===", flush=True)
        trainer.test(model, datamodule=dm, ckpt_path="best")
