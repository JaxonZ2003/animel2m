import torch
import argparse
import warnings
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from pathlib import Path
from tqdm import tqdm
from torchvision import transforms

from captum.attr import LayerGradCam, IntegratedGradients
from captum.attr._core.gradient_shap import GradientShap
from learner import BaselineLitModule, AniXploreLitModule

from models.AniXplore.AniXplore import AniXplore

ALL_MODELS = [
    "convnext",
    # "resnet",
    # "vit",
    "frequency",
    "efficientnet",
    # "dualstream",
    # "lightweight",
    "anixplore",
]

# denormalize constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

warnings.filterwarnings("ignore")


def _find_best_ckpt(model_name, fold=0, seed=4710):
    ckpt_dir = Path(f"out/seed{seed}_fold{fold}/checkpoint") / model_name
    if not ckpt_dir.exists():
        raise ValueError(f"Checkpoint directory not found: {ckpt_dir}")

    ckpts = list(ckpt_dir.glob("epoch=*-val_auc=*.ckpt"))
    if not ckpts:
        raise ValueError(f"No checkpoints found in {ckpt_dir}")

    def _get_auc(p):
        stem = p.stem  # epoch=xx-val_auc=yy.ckpt
        try:
            return float(stem.split("val_auc=")[1])
        except Exception:
            return -1.0

    best_ckpt = sorted(ckpts, key=_get_auc)[-1]
    print(f"[INFO] Best ckpt for {model_name} (fold {fold}): {best_ckpt}")
    return best_ckpt


def _load_lit_module(model_name, ckpt_path, device="cuda"):
    if model_name == "anixplore":
        lit_model = AniXploreLitModule.load_from_checkpoint(str(ckpt_path))
    else:
        lit_model = BaselineLitModule.load_from_checkpoint(str(ckpt_path))

    lit_model.to(device)
    lit_model.eval()
    return lit_model


def _get_target_layer_for_gradcam(model_name, lit_model):
    m = lit_model.model

    if model_name == "anixplore":
        return m.fusion_layers[-1]

    if model_name == "convnext":
        return m.backbone.stages[-1]

    if model_name == "resnet":
        return m.backbone.layer4[-1]

    if model_name == "vit":
        try:
            return m.backbone.blocks[-1].attn
        except:
            print(
                f"Warning: Could not find attention layer for ViT, GradCAM may not work properly"
            )
            return None

    if model_name == "frequency":
        if hasattr(m.backbone, "stages"):
            return m.backbone.stages[-1]
        elif hasattr(m.backbone, "layer4"):
            return m.backbone.layer4[-1]
        else:
            return None

    if model_name == "dualstream":
        return m.fusion[0]

    if model_name == "lightweight":
        return m.features[-4]  # Last conv before pooling


def _make_forward_func(model_name, lit_model, device="cuda"):
    m = lit_model.model

    if model_name == "anixplore":

        def forward(x):
            B, _, H, W = x.shape
            x = x.to(device)
            dummy_mask = torch.zeros((B, 1, H, W), device=device)
            dummy_label = torch.zeros((B,), device=device)
            out = m(x, dummy_mask, dummy_label)
            prob = out["pred_prob"].view(B, 1)
            return prob

        return forward

    else:

        def forward(x):
            x = x.to(device)
            logits = m.get_logits(x)  # [B, 1]
            return logits

        return forward


def _attr_to_heatmap(attr_tensor, H, W):
    if attr_tensor is None:
        return None

    if attr_tensor.dim() == 4:
        # [1, C, h, w]
        if attr_tensor.shape[2:] != (H, W):
            attr_tensor = F.interpolate(
                attr_tensor, size=(H, W), mode="bilinear", align_corners=False
            )
        attr_tensor = attr_tensor[0]  # [C, H, W]

    if attr_tensor.dim() == 3:
        # [C, H, W]
        attr_tensor = attr_tensor.abs().mean(dim=0)  # [H, W]

    heat = attr_tensor.detach().cpu().numpy()
    if heat.max() > heat.min():
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    return heat


def denormalize(t):
    device = t.device
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    x = t * std + mean
    x = x.clamp(0, 1)
    x = x.permute(1, 2, 0).cpu().numpy()  # HWC
    return x


def _load_and_preprocess_image(img_path, img_size=512, device="cuda"):
    pil_img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    img = transform(pil_img).unsqueeze(0).to(device)  # [1, 3, img_size, img_size]
    img_denorm = denormalize(img[0])
    return img, img_denorm


def generate_all_models_gradcam(
    img_path,
    fold=0,
    seed=4710,
    img_size=512,
    device="cuda",
    save_path="all_models_gradcam.png",
    model_list=None,
):
    if model_list is None:
        model_list = ALL_MODELS

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    img_tensor, img_denorm = _load_and_preprocess_image(img_path, img_size, device)
    _, _, H, W = img_tensor.shape

    results = []

    for model_name in model_list:
        run_name = model_name if model_name != "anixplore" else "anixplore"
        ckpt_path = _find_best_ckpt(model_name, fold, seed)
        if ckpt_path is None:
            print(
                f"[WARN] No checkpoint found for {model_name}, skipping...", flush=True
            )
            continue

        lit_model = _load_lit_module(model_name, ckpt_path, device)
        target_layer = _get_target_layer_for_gradcam(model_name, lit_model)
        if target_layer is None:
            print(
                f"[WARNING] No target layer found for GradCAM: {model_name}",
                flush=True,
            )
            results.append((model_name, None, None))
            continue

        forward_func = _make_forward_func(model_name, lit_model, device)
        with torch.no_grad():
            pred_out = forward_func(img_tensor)
            pred_prob = torch.sigmoid(pred_out).item()

        gradcam = LayerGradCam(forward_func, target_layer)
        attr = gradcam.attribute(img_tensor, target=0, relu_attributions=True)
        heat = _attr_to_heatmap(attr, H, W)

        results.append((model_name, heat, pred_prob))

    if not results:
        print("[ERROR] No results to visualize.", flush=True)
        return

    n_models = len(results)
    fig, axes = plt.subplots(n_models, 2, figsize=(8, 3 * n_models))
    if n_models == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (model_name, heat, prob) in enumerate(results):
        ax_img = axes[i, 0]
        ax_cam = axes[i, 1]

        ax_img.imshow(img_denorm)
        ax_img.set_title(f"{model_name} - Input")
        ax_img.axis("off")

        ax_cam.imshow(img_denorm)
        if heat is not None:
            ax_cam.imshow(heat, cmap="jet", alpha=0.5)
        else:
            ax_cam.text(0.5, 0.5, "N/A", ha="center", va="center")
        if prob is not None:
            ax_cam.set_title(f"{model_name} - GradCAM (Prob: {prob:.4f})")
        else:
            ax_cam.set_title(f"{model_name} - GradCAM")
        ax_cam.axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved all models GradCAM visualization to {save_path}", flush=True)


def generate_all_models_integrated_gradients(
    img_path,
    fold=0,
    seed=4710,
    img_size=512,
    device="cuda",
    save_path="all_models_ig.png",
    model_list=None,
):
    if model_list is None:
        model_list = ALL_MODELS

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    img_tensor, img_denorm = _load_and_preprocess_image(img_path, img_size, device)
    _, _, H, W = img_tensor.shape

    results = []

    for model_name in model_list:
        run_name = model_name if model_name != "anixplore" else "anixplore"
        ckpt_path = _find_best_ckpt(run_name, fold=fold, seed=seed)
        if ckpt_path is None:
            continue

        lit_model = _load_lit_module(model_name, ckpt_path, device=device)
        forward_func = _make_forward_func(model_name, lit_model, device)

        baseline = torch.zeros_like(img_tensor).to(device)
        ig = IntegratedGradients(forward_func)

        with torch.no_grad():
            pred_out = forward_func(img_tensor)
            pred_prob = torch.sigmoid(pred_out).item()

        attr = ig.attribute(img_tensor, baselines=baseline, n_steps=50)
        heat = _attr_to_heatmap(attr, H, W)

        results.append((model_name, heat, pred_prob))

    if not results:
        print("[ERROR] No model results, nothing to plot.", flush=True)
        return

    n_models = len(results)
    fig, axes = plt.subplots(n_models, 2, figsize=(8, 3 * n_models))
    if n_models == 1:
        axes = np.expand_dims(axes, 0)

    for i, (model_name, heat, prob) in enumerate(results):
        ax_img = axes[i, 0]
        ax_attr = axes[i, 1]

        ax_img.imshow(img_denorm)
        ax_img.set_title(f"{model_name} - Input")
        ax_img.axis("off")

        ax_attr.imshow(img_denorm)
        if heat is not None:
            ax_attr.imshow(heat, cmap="RdBu_r", alpha=0.5)
        else:
            ax_attr.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax_attr.set_title(f"{model_name} - Integrated Gradients (Prob: {prob:.3f})")
        ax_attr.axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved IG grid to {save_path}")


def generate_all_models_shap(
    img_path,
    fold=0,
    seed=4710,
    img_size=512,
    device="cuda",
    save_path="all_models_shap.png",
    model_list=None,
    n_samples=50,
):
    if model_list is None:
        model_list = ALL_MODELS

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    img_tensor, img_denorm = _load_and_preprocess_image(img_path, img_size, device)
    _, _, H, W = img_tensor.shape

    results = []

    for model_name in model_list:
        run_name = model_name if model_name != "anixplore" else "anixplore"
        ckpt_path = _find_best_ckpt(run_name, fold=fold, seed=seed)
        if ckpt_path is None:
            continue

        lit_model = _load_lit_module(model_name, ckpt_path, device=device)
        forward_func = _make_forward_func(model_name, lit_model, device)

        baseline = torch.zeros_like(img_tensor).to(device)
        gshap = GradientShap(forward_func)

        with torch.no_grad():
            pred_out = forward_func(img_tensor)
            pred_prob = torch.sigmoid(pred_out).item()

        attr = gshap.attribute(
            img_tensor,
            baselines=baseline,
            n_samples=n_samples,
            stdevs=0.0001,
        )
        heat = _attr_to_heatmap(attr, H, W)

        results.append((model_name, heat, pred_prob))

    if not results:
        print("[ERROR] No model results, nothing to plot.")
        return

    n_models = len(results)
    fig, axes = plt.subplots(n_models, 2, figsize=(8, 3 * n_models))
    if n_models == 1:
        axes = np.expand_dims(axes, 0)

    for i, (model_name, heat, prob) in enumerate(results):
        ax_img = axes[i, 0]
        ax_attr = axes[i, 1]

        ax_img.imshow(img_denorm)
        ax_img.set_title(f"{model_name} - Input")
        ax_img.axis("off")

        ax_attr.imshow(img_denorm)
        if heat is not None:
            ax_attr.imshow(heat, cmap="RdBu_r", alpha=0.5)
        else:
            ax_attr.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax_attr.set_title(f"{model_name} - SHAP (GradientShap) (Prob: {prob:.3f})")
        ax_attr.axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved SHAP grid to {save_path}")


def generate_anixplore_mask_and_explanations(
    img_path,
    fold=0,
    seed=4710,
    img_size=512,
    device="cuda",
    save_path="anixplore_mask_and_explanations.png",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 1. 加载最佳 anixplore ckpt
    ckpt_path = _find_best_ckpt("anixplore", fold=fold, seed=seed)
    if ckpt_path is None:
        print("[ERROR] No AniXplore checkpoint found.")
        return

    lit_model = _load_lit_module("anixplore", ckpt_path, device=device)
    m = lit_model.model

    # 2. 处理图片
    img_tensor, img_denorm = _load_and_preprocess_image(img_path, img_size, device)
    B, _, H, W = img_tensor.shape

    dummy_mask = torch.zeros((B, 1, H, W), device=device)
    dummy_label = torch.zeros((B,), device=device)

    # 3. 得到 mask prediction
    with torch.no_grad():
        out = m(img_tensor, dummy_mask, dummy_label)
        pred_mask = out["pred_mask"]  # [1,1,H,W]
        prob = out["pred_prob"].item()

    heat_mask = pred_mask[0, 0].detach().cpu().numpy()
    if heat_mask.max() > heat_mask.min():
        heat_mask = (heat_mask - heat_mask.min()) / (heat_mask.max() - heat_mask.min())
    max_y, max_x = np.unravel_index(np.argmax(heat_mask), heat_mask.shape)

    # 4. AniXplore 的三种 explanation
    forward_func = _make_forward_func("anixplore", lit_model, device)
    target_layer = _get_target_layer_for_gradcam("anixplore", lit_model)

    # GradCAM
    if target_layer is not None:
        gradcam = LayerGradCam(forward_func, target_layer)
        attr_gc = gradcam.attribute(img_tensor, target=0, relu_attributions=True)
        heat_gc = _attr_to_heatmap(attr_gc, H, W)
    else:
        heat_gc = None

    # IG
    baseline = torch.zeros_like(img_tensor).to(device)
    ig = IntegratedGradients(forward_func)
    attr_ig = ig.attribute(img_tensor, baselines=baseline, n_steps=50)
    heat_ig = _attr_to_heatmap(attr_ig, H, W)

    # SHAP (GradientShap)
    gshap = GradientShap(forward_func)
    attr_shap = gshap.attribute(
        img_tensor, baselines=baseline, n_samples=50, stdevs=0.0001
    )
    heat_shap = _attr_to_heatmap(attr_shap, H, W)

    # 5. 拼图：原图、mask、GradCAM、IG、SHAP
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # (0,0) 原图
    axes[0, 0].imshow(img_denorm)
    axes[0, 0].set_title(f"Input Image\nAniXplore pred={prob:.3f}")
    axes[0, 0].axis("off")

    # (0,1) mask prediction + max evidence point
    axes[0, 1].imshow(img_denorm)
    axes[0, 1].imshow(heat_mask, cmap="jet", alpha=0.5)
    axes[0, 1].scatter([max_x], [max_y], c="red", s=30)
    axes[0, 1].set_title("AniXplore Mask Prediction")
    axes[0, 1].axis("off")

    # (0,2) GradCAM
    axes[0, 2].imshow(img_denorm)
    if heat_gc is not None:
        axes[0, 2].imshow(heat_gc, cmap="jet", alpha=0.5)
    else:
        axes[0, 2].text(0.5, 0.5, "GradCAM N/A", ha="center", va="center")
    axes[0, 2].set_title("AniXplore GradCAM")
    axes[0, 2].axis("off")

    # (1,0) IG
    axes[1, 0].imshow(img_denorm)
    if heat_ig is not None:
        axes[1, 0].imshow(heat_ig, cmap="RdBu_r", alpha=0.5)
    else:
        axes[1, 0].text(0.5, 0.5, "IG N/A", ha="center", va="center")
    axes[1, 0].set_title("AniXplore Integrated Gradients")
    axes[1, 0].axis("off")

    # (1,1) SHAP
    axes[1, 1].imshow(img_denorm)
    if heat_shap is not None:
        axes[1, 1].imshow(heat_shap, cmap="RdBu_r", alpha=0.5)
    else:
        axes[1, 1].text(0.5, 0.5, "SHAP N/A", ha="center", va="center")
    axes[1, 1].set_title("AniXplore SHAP (GradientShap)")
    axes[1, 1].axis("off")

    # (1,2) 留空 or 再画一次 mask/其它
    axes[1, 2].axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved AniXplore explanations to {save_path}")


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


def main():
    img_path = "/home/yz2483/scratch.gerstein/animel2m_dataset/fake_images/0000/text2image/255000/SD_text2image_255000.png"

    generate_all_models_gradcam(
        img_path,
        fold=2,
        img_size=512,
        save_path="viz/all_models_gradcam_fold2.png",
    )

    generate_all_models_integrated_gradients(
        img_path,
        fold=2,
        img_size=512,
        save_path="viz/all_models_ig_fold2.png",
    )

    generate_all_models_shap(
        img_path,
        fold=2,
        img_size=512,
        save_path="viz/all_models_shap_fold2.png",
    )

    generate_anixplore_mask_and_explanations(
        img_path,
        fold=2,
        img_size=512,
        save_path="viz/anixplore_mask_and_expl_fold2.png",
    )


if __name__ == "__main__":
    main()
