import torch
import argparse
import warnings
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms

from models.AniXplore.AniXplore import AniXplore

# denormalize constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

warnings.filterwarnings("ignore")


def denormalize(t):
    device = t.device
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    x = t * std + mean
    x = x.clamp(0, 1)
    x = x.permute(1, 2, 0).cpu().numpy()  # HWC
    return x


@torch.no_grad()
def visualize_mask_on_image(model, img_path, device="cuda", save_path="mask_vis.png"):
    model.eval().to(device)

    pil_img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    img = transform(pil_img).unsqueeze(0).to(device)  # [1, 3, 512, 512]

    # create dummy mask just ensure the model runs
    _, _, H, W = img.shape
    dummy_mask = torch.zeros((1, 1, H, W), device=device)
    dummy_label = torch.zeros((1,), device=device)

    output = model(img, dummy_mask, dummy_label)
    pred_mask = output["pred_mask"]  # [1, 1, H, W]

    heatmap = pred_mask[0, 0].cpu().numpy()
    binary_mask = (heatmap > 0.5).astype(np.uint8)

    # maximum explanation point (highest probability)
    max_y, max_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    img_denorm = denormalize(img[0])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img_denorm)
    axes[0].scatter([max_x], [max_y], c="red", s=30, label="Max Explanation Point")
    axes[0].set_title("Max Evidence Point")
    axes[0].axis("off")

    axes[1].imshow(img_denorm)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.5)
    axes[1].set_title("Heatmap Evidence Map")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved visualization to {save_path}")


# Import Captum
from captum.attr import (
    # GradCAM,
    GuidedGradCam,
    IntegratedGradients,
    LayerGradCam,
    visualization as viz,
)
from captum.attr._core.gradient_shap import GradientShap

# Import your modules
import sys

sys.path.append(".")
from models.Baselines.baselines import get_baseline_model
from models.AniXplore.AniXplore import AniXplore
from dataset import SimpleRealDataset, SimpleFakeDataset, parse_fake_path
from learner import BaselineLitModule, AniXploreLitModule

# For visualization
from torchvision import transforms
from PIL import Image
import os


class ModelInterpreter:
    """Class to handle model interpretation using GradCAM and SHAP"""

    def __init__(self, model_name, checkpoint_path, device="cuda", img_size=224):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.img_size = img_size

        # Load model from checkpoint
        self.model = self._load_model()
        self.model.eval()

        # Get the target layer for GradCAM based on model architecture
        self.target_layer = self._get_target_layer()

        # Initialize interpretation methods
        self.gradcam = None
        self.shap = None
        if self.target_layer is not None:
            self.gradcam = LayerGradCam(self.forward_func, self.target_layer)

    def _load_model(self):
        """Load model from checkpoint"""
        print(f"Loading model from: {self.checkpoint_path}")

        if self.model_name == "anixplore":
            # Load AniXplore model
            model = AniXploreLitModule.load_from_checkpoint(
                self.checkpoint_path,
                seg_pretrain_path="./segformer_mit-b0.pth",  # You may need to adjust this
                img_size=self.img_size,
            )
        else:
            # Load baseline model
            model = BaselineLitModule.load_from_checkpoint(
                self.checkpoint_path, model_name=self.model_name, img_size=self.img_size
            )

        return model.to(self.device)

    def _get_target_layer(self):
        """Get the appropriate layer for GradCAM based on model architecture"""
        if self.model_name == "anixplore":
            # For AniXplore, use the fusion layer before classification
            return self.model.model.fusion_layers[-1]

        elif self.model_name == "convnext":
            # For ConvNeXt, use the last stage
            return self.model.model.backbone.stages[-1]

        elif self.model_name == "resnet":
            # For ResNet, use the last conv layer in layer4
            return self.model.model.backbone.layer4[-1]

        elif self.model_name == "vit":
            # For ViT, GradCAM is trickier, use the last attention layer
            # Note: ViT may not work well with GradCAM
            try:
                return self.model.model.backbone.blocks[-1].attn
            except:
                print(
                    f"Warning: Could not find attention layer for ViT, GradCAM may not work properly"
                )
                return None

        elif self.model_name == "efficientnet":
            # For EfficientNet, use the last conv layer
            return self.model.model.backbone.conv_head

        elif self.model_name == "frequency":
            # For frequency-aware model, use the backbone's last layer
            if hasattr(self.model.model.backbone, "stages"):
                return self.model.model.backbone.stages[-1]
            else:
                return self.model.model.backbone.layer4[-1]

        elif self.model_name == "dualstream":
            # For dual-stream, use the fusion layer
            return self.model.model.fusion[0]

        elif self.model_name == "lightweight":
            # For lightweight CNN, use the last conv layer
            return self.model.model.features[-4]  # Last conv before pooling

        else:
            print(
                f"Warning: Unknown model {self.model_name}, cannot determine target layer"
            )
            return None

    def forward_func(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        """Forward function for Captum"""
        # if self.model_name == "anixplore":
        #     # AniXplore needs special handling
        #     batch_size = input_tensor.shape[0]
        #     dummy_mask = torch.zeros((batch_size, 1, self.img_size, self.img_size)).to(
        #         self.device
        #     )
        #     dummy_label = torch.ones(batch_size).to(self.device)
        #     output = self.model.model(input_tensor, dummy_mask, dummy_label)
        #     # Return raw logits for classification
        #     return self.model.model.cls_head(
        #         self.model.model.fusion_layers[-1](
        #             torch.cat(
        #                 [
        #                     self.model.model.convnext.stages[-1](
        #                         self.model.model.convnext.stem(
        #                             torch.cat([input_tensor, input_tensor], dim=1)
        #                         )
        #                     ),
        #                     input_tensor,
        #                 ],
        #                 dim=1,
        #             )
        #         )
        #     ).squeeze()
        # else:
        #     # Baseline models
        #     return self.model.model.get_logits(input_tensor).squeeze()

        if self.model_name == "anixplore":
            # 这里先简单处理：我们暂时不用 AniXplore 做 GradCAM
            # 为了 SHAP / IG 至少有东西用，先用 backward_loss 近似一个标量输出
            batch_size = input_tensor.shape[0]
            dummy_mask = torch.zeros(
                (batch_size, 1, self.img_size, self.img_size),
                device=self.device,
            )
            dummy_label = torch.zeros(batch_size, device=self.device)
            output_dict = self.model.model(input_tensor, dummy_mask, dummy_label)
            # backward_loss 是标量 [B]，我们扩一维成 [B, 1] 让 Captum 好受一点
            loss = output_dict["backward_loss"]
            return loss.view(-1, 1)

        else:
            # Baseline 模型：保留 logits 的 (B, 1) 形状
            logits = self.model.model.get_logits(input_tensor)  # [B, 1]
            return logits

    def interpret_gradcam(self, input_tensor, target_class=None):
        """Generate GradCAM interpretation"""
        if self.gradcam is None:
            return None

        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True

        # If no target class specified, use predicted class
        if target_class is None:
            # output = self.forward_func(input_tensor)
            # target_class = int(torch.sigmoid(output) > 0.5)
            target_class = 0

        # Generate GradCAM attributions
        attributions = self.gradcam.attribute(
            input_tensor, target=target_class, relu_attributions=True
        )

        return attributions

    def interpret_shap(self, input_tensor, n_samples=50):
        """Generate SHAP interpretation using GradientShap"""
        input_tensor = input_tensor.to(self.device)

        # Create baseline (black image)
        baseline = torch.zeros_like(input_tensor).to(self.device)

        # Initialize GradientShap
        gradient_shap = GradientShap(self.forward_func)

        # Generate SHAP values
        attributions = gradient_shap.attribute(
            input_tensor,
            baselines=baseline,
            n_samples=n_samples,
            stdevs=0.0001,
            target=None,  # Will use predicted class
        )

        return attributions

    def interpret_integrated_gradients(self, input_tensor):
        """Generate Integrated Gradients interpretation"""
        input_tensor = input_tensor.to(self.device)

        # Create baseline (black image)
        baseline = torch.zeros_like(input_tensor).to(self.device)

        # Initialize Integrated Gradients
        ig = IntegratedGradients(self.forward_func)

        # Generate attributions
        attributions = ig.attribute(
            input_tensor, baselines=baseline, target=None, n_steps=50
        )

        return attributions


def create_visualization_grid(
    images_data, save_path, title="Model Interpretations", max_cols=4
):
    """
    Create a grid visualization of interpretations
    images_data: list of dicts with keys: 'original', 'gradcam', 'shap', 'ig', 'label', 'pred'
    """
    n_samples = len(images_data)
    n_cols = min(max_cols, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols

    # Each sample gets 4 subplots (original, gradcam, shap, ig)
    fig = plt.figure(figsize=(n_cols * 12, n_rows * 3.5))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.3, wspace=0.2)

    for idx, data in enumerate(images_data):
        row = idx // n_cols
        col = idx % n_cols

        # Create subplot for this sample
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=gs[row, col], wspace=0.1
        )

        # Original image
        ax1 = fig.add_subplot(inner_gs[0])
        ax1.imshow(data["original"])
        ax1.set_title(
            f"Original\nTrue: {data['label']}, Pred: {data['pred']:.2f}", fontsize=10
        )
        ax1.axis("off")

        # GradCAM
        ax2 = fig.add_subplot(inner_gs[1])
        if data["gradcam"] is not None:
            ax2.imshow(data["original"], alpha=0.5)
            ax2.imshow(data["gradcam"], cmap="jet", alpha=0.5)
            ax2.set_title("GradCAM", fontsize=10)
        else:
            ax2.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax2.transAxes)
            ax2.set_title("GradCAM", fontsize=10)
        ax2.axis("off")

        # SHAP
        ax3 = fig.add_subplot(inner_gs[2])
        if data["shap"] is not None:
            ax3.imshow(data["shap"])
            ax3.set_title("SHAP", fontsize=10)
        else:
            ax3.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax3.transAxes)
            ax3.set_title("SHAP", fontsize=10)
        ax3.axis("off")

        # Integrated Gradients
        ax4 = fig.add_subplot(inner_gs[3])
        if data.get("ig") is not None:
            ax4.imshow(data["ig"])
            ax4.set_title("Integrated Gradients", fontsize=10)
        else:
            ax4.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax4.transAxes)
            ax4.set_title("Integrated Gradients", fontsize=10)
        ax4.axis("off")

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    # Save figure
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to {save_path}")


def process_attributions(attributions, input_tensor):
    """Process attributions for visualization"""
    if attributions is None:
        return None

    # Convert to numpy
    attr_np = attributions.squeeze().cpu().detach().numpy()

    # Handle different shapes
    if len(attr_np.shape) == 3:  # [C, H, W]
        # Average across channels
        attr_np = np.mean(np.abs(attr_np), axis=0)
    elif len(attr_np.shape) == 4:  # [1, C, H, W]
        attr_np = np.mean(np.abs(attr_np[0]), axis=0)

    # Normalize to [0, 1]
    if attr_np.max() > attr_np.min():
        attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min())

    return attr_np


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image tensor for visualization"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor


def evaluate_model(
    model_name, checkpoint_path, data_loader, n_samples=8, save_dir="interpretations"
):
    """
    Evaluate a single model and generate interpretations
    """
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    # Initialize interpreter
    interpreter = ModelInterpreter(model_name, checkpoint_path)

    # Collect samples for visualization
    images_data = []

    # Process samples

    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing samples")):
        if len(images_data) >= n_samples:
            break

        # Get first image from batch
        input_tensor = batch["image"][:1]
        label = batch["label"][:1].item()

        # Get model prediction
        output = interpreter.forward_func(input_tensor)
        pred = torch.sigmoid(output).item()

        # Generate interpretations
        print(f"  Generating GradCAM for sample {len(images_data)+1}...")
        gradcam_attr = interpreter.interpret_gradcam(input_tensor)

        print(f"  Generating SHAP for sample {len(images_data)+1}...")
        shap_attr = interpreter.interpret_shap(input_tensor, n_samples=25)

        print(f"  Generating Integrated Gradients for sample {len(images_data)+1}...")
        ig_attr = interpreter.interpret_integrated_gradients(input_tensor)

        # Process for visualization
        original_img = denormalize_image(input_tensor[0]).permute(1, 2, 0).cpu().numpy()
        gradcam_vis = process_attributions(gradcam_attr, input_tensor)
        shap_vis = process_attributions(shap_attr, input_tensor)
        ig_vis = process_attributions(ig_attr, input_tensor)

        # Apply colormap to SHAP and IG for better visualization
        if shap_vis is not None:
            cmap = plt.cm.RdBu_r
            shap_vis = cmap(shap_vis)

        if ig_vis is not None:
            cmap = plt.cm.RdBu_r
            ig_vis = cmap(ig_vis)

        images_data.append(
            {
                "original": original_img,
                "gradcam": gradcam_vis,
                "shap": shap_vis,
                "ig": ig_vis,
                "label": int(label),
                "pred": pred,
            }
        )

    # Create visualization
    save_path = Path(save_dir) / f"{model_name}_interpretations.png"
    create_visualization_grid(
        images_data, save_path, title=f"{model_name.upper()} Model Interpretations"
    )

    return images_data


def main():
    parser = argparse.ArgumentParser(
        description="Interpret trained models using GradCAM and SHAP"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="out/checkpoint",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to evaluate. If None, evaluate all found models",
    )
    parser.add_argument(
        "--fake_root",
        type=str,
        default="/gpfs/milgram/scratch60/gerstein/yz2483/animel2m_dataset/fake_images",
        help="Path to fake images dataset",
    )
    parser.add_argument(
        "--real_root",
        type=str,
        default="/gpfs/milgram/scratch60/gerstein/yz2483/animel2m_dataset/real_images/resized_img",
        help="Path to real images dataset",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=8,
        help="Number of samples to visualize per model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for data loading"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="interpretations",
        help="Directory to save interpretation visualizations",
    )
    parser.add_argument(
        "--img_size", type=int, default=224, help="Image size for model input"
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Available baseline models
    baseline_models = [
        "convnext",
        "resnet",
        "vit",
        "frequency",
        "efficientnet",
        "dualstream",
        "lightweight",
    ]

    # Find checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Determine which models to evaluate
    if args.models:
        models_to_eval = args.models
    else:
        # Find all models with checkpoints
        models_to_eval = []
        for model_name in baseline_models:
            model_dir = checkpoint_dir / model_name
            if model_dir.exists():
                # Find best checkpoint (highest val_auc)
                ckpt_files = list(model_dir.glob("epoch=*-val_auc=*.ckpt"))
                if ckpt_files:
                    models_to_eval.append(model_name)

    print(f"Found models to evaluate: {models_to_eval}")

    # Prepare data
    print("\nPreparing data...")

    # Parse fake records
    parsed = parse_fake_path(args.fake_root, quiet=True)
    fake_records = parsed["records"][:100]  # Use subset for faster processing

    # Create mixed dataset (fake and real)
    fake_dataset = SimpleFakeDataset(
        fake_records, img_size=args.img_size, augment=False
    )
    real_dataset = SimpleRealDataset(
        args.real_root, img_size=args.img_size, augment=False
    )

    # Create a mixed dataset for evaluation
    from torch.utils.data import ConcatDataset, DataLoader, Subset

    # Use subset of data
    n_fake = min(args.n_samples // 2, len(fake_dataset))
    n_real = min(args.n_samples // 2, len(real_dataset))

    fake_subset = Subset(fake_dataset, range(n_fake))
    real_subset = Subset(real_dataset, range(n_real))

    mixed_dataset = ConcatDataset([fake_subset, real_subset])

    # Import the collate function
    from dataset import simple_collate_fn

    data_loader = DataLoader(
        mixed_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=simple_collate_fn,
    )

    # Evaluate each model
    results = {}
    for model_name in models_to_eval:
        model_dir = checkpoint_dir / model_name

        # Find best checkpoint
        ckpt_files = list(model_dir.glob("epoch=*-val_auc=*.ckpt"))
        if not ckpt_files:
            print(f"No checkpoint found for {model_name}, skipping...")
            continue

        # Sort by val_auc (in filename) and take the best
        best_ckpt = sorted(
            ckpt_files, key=lambda x: float(x.stem.split("val_auc=")[1])
        )[-1]

        try:
            # Evaluate model
            results[model_name] = evaluate_model(
                model_name,
                str(best_ckpt),
                data_loader,
                n_samples=args.n_samples,
                save_dir=args.save_dir,
            )

        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    # Create combined visualization
    if len(results) > 1:
        print("\nCreating combined visualization...")
        combined_save_path = Path(args.save_dir) / "all_models_comparison.png"

        # Take first few samples from each model for comparison
        fig, axes = plt.subplots(len(results), 4, figsize=(16, 4 * len(results)))
        if len(results) == 1:
            axes = axes.reshape(1, -1)

        for idx, (model_name, model_results) in enumerate(results.items()):
            if model_results and len(model_results) > 0:
                sample = model_results[0]  # Take first sample

                # Original
                axes[idx, 0].imshow(sample["original"])
                axes[idx, 0].set_title(f"{model_name}\nOriginal", fontsize=10)
                axes[idx, 0].axis("off")

                # GradCAM
                if sample["gradcam"] is not None:
                    axes[idx, 1].imshow(sample["original"], alpha=0.5)
                    axes[idx, 1].imshow(sample["gradcam"], cmap="jet", alpha=0.5)
                else:
                    axes[idx, 1].text(0.5, 0.5, "N/A", ha="center", va="center")
                axes[idx, 1].set_title("GradCAM", fontsize=10)
                axes[idx, 1].axis("off")

                # SHAP
                if sample["shap"] is not None:
                    axes[idx, 2].imshow(sample["shap"])
                else:
                    axes[idx, 2].text(0.5, 0.5, "N/A", ha="center", va="center")
                axes[idx, 2].set_title("SHAP", fontsize=10)
                axes[idx, 2].axis("off")

                # IG
                if sample.get("ig") is not None:
                    axes[idx, 3].imshow(sample["ig"])
                else:
                    axes[idx, 3].text(0.5, 0.5, "N/A", ha="center", va="center")
                axes[idx, 3].set_title("Integrated Gradients", fontsize=10)
                axes[idx, 3].axis("off")

        plt.suptitle("Model Interpretations Comparison", fontsize=14)
        plt.tight_layout()
        plt.savefig(combined_save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved combined visualization to {combined_save_path}")

    print("\nInterpretation complete!")
    print(f"Results saved in {args.save_dir}/")


if __name__ == "__main__":
    main()
