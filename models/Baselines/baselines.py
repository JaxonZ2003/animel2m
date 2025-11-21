import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models


class BaselineModel(nn.Module):
    """Base class for all baseline models"""

    def forward(self, image, mask=None, label=None, *args, **kwargs):
        # All baselines ignore mask input
        logits = self.get_logits(image)

        output_dict = {
            "backward_loss": torch.tensor(0.0).to(
                image.device
            ),  # Will be computed in training
            "pred_label": torch.tensor(0.0).to(
                image.device
            ),  # Will be computed in training
            "raw_logits": logits,
        }

        if label is not None:
            label = label.float()
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), label)
            pred_prob = torch.sigmoid(logits.squeeze(-1))
            pred_label = (pred_prob > 0.5).float()

            output_dict.update(
                {
                    "backward_loss": loss,
                    "pred_label": pred_label,
                    "visual_loss": {
                        "cls_loss": loss,
                    },
                }
            )

        return output_dict

    def get_logits(self, image):
        raise NotImplementedError


# ============== Baseline 1: Simple ConvNeXt Classifier ==============
class ConvNeXtBaseline(BaselineModel):
    """
    Simple ConvNeXt-based classifier without frequency decomposition or mask prediction.
    Uses pretrained ConvNeXt-Tiny as backbone.
    """

    def __init__(self, pretrained=True, num_classes=1):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny", pretrained=pretrained, num_classes=0
        )
        # Get the feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            feat_dim = self.backbone(dummy_input).shape[-1]

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def get_logits(self, image):
        features = self.backbone(image)
        logits = self.classifier(features)
        return logits


# ============== Baseline 2: ResNet50 Classifier ==============
class ResNetBaseline(BaselineModel):
    """
    ResNet50-based classifier, a classic baseline for image classification.
    """

    def __init__(self, pretrained=True, num_classes=1):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        # Replace the final FC layer
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def get_logits(self, image):
        features = self.backbone(image)
        logits = self.classifier(features)
        return logits


# ============== Baseline 3: Vision Transformer (ViT) Classifier ==============
class ViTBaseline(BaselineModel):
    """
    Vision Transformer baseline for comparison with CNN-based approaches.
    """

    def __init__(self, pretrained=True, num_classes=1, img_size=512):
        super().__init__()
        # Use ViT-Small for efficiency
        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
        )

        # Get the feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            feat_dim = self.backbone(dummy_input).shape[-1]

        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def get_logits(self, image):
        features = self.backbone(image)
        logits = self.classifier(features)
        return logits


# ============== Baseline 4: Frequency-Aware Classifier ==============
class FrequencyAwareBaseline(BaselineModel):
    """
    A baseline that uses frequency analysis (DCT/DWT) but only for classification.
    Simpler than AniXplore as it doesn't do mask prediction or dual-stream fusion.
    """

    def __init__(self, backbone="convnext_tiny", pretrained=True, num_classes=1):
        super().__init__()
        from models.AniXplore.dct_extractor import DctFrequencyExtractor
        from models.AniXplore.dwt_extractor import DwtFrequencyExtractor

        self.dct = DctFrequencyExtractor()
        self.dwt = DwtFrequencyExtractor()

        # Create backbone with 6 input channels (RGB + frequency)
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)

        # Modify first conv layer to accept 6 channels
        if hasattr(self.backbone, "stem"):  # ConvNeXt
            original_conv = self.backbone.stem[0]
            self.backbone.stem[0] = nn.Conv2d(
                6,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False,
            )
            if pretrained:
                # Initialize with pretrained weights for RGB channels
                with torch.no_grad():
                    self.backbone.stem[0].weight[:, :3] = original_conv.weight
                    self.backbone.stem[0].weight[
                        :, 3:
                    ] = original_conv.weight  # Duplicate for frequency

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 6, 224, 224)
            feat_dim = self.backbone(dummy_input).shape[-1]

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def get_logits(self, image):
        # Extract frequency components
        high_dct = self.dct.forward_high(image)
        high_dwt = self.dwt.forward(image)
        high_freq = high_dct * 0.5 + high_dwt * 0.5

        # Concatenate with original image
        x = torch.cat([image, high_freq], dim=1)

        # Pass through backbone
        features = self.backbone(x)

        # Handle different output shapes
        if len(features.shape) == 4:  # Conv output [B, C, H, W]
            logits = self.classifier(features)
        else:  # Transformer output [B, D]
            logits = self.classifier[2:](features)  # Skip pooling and flatten

        return logits


# ============== Baseline 5: EfficientNet Classifier ==============
class EfficientNetBaseline(BaselineModel):
    """
    EfficientNet-B0 baseline, known for good accuracy-efficiency trade-off.
    """

    def __init__(self, pretrained=True, num_classes=1):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0
        )

        # Get the feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            feat_dim = self.backbone(dummy_input).shape[-1]

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def get_logits(self, image):
        features = self.backbone(image)
        logits = self.classifier(features)
        return logits


# ============== Baseline 6: Simple Dual-Stream (RGB + Frequency) ==============
class DualStreamBaseline(BaselineModel):
    """
    Simplified dual-stream architecture processing RGB and frequency separately,
    then fusing for classification. No mask prediction.
    """

    def __init__(self, backbone="convnext_tiny", pretrained=True, num_classes=1):
        super().__init__()
        from models.AniXplore.dct_extractor import DctFrequencyExtractor
        from models.AniXplore.dwt_extractor import DwtFrequencyExtractor

        self.dct = DctFrequencyExtractor()
        self.dwt = DwtFrequencyExtractor()

        # Two separate backbones
        self.rgb_backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0
        )
        self.freq_backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0
        )

        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            feat_dim = self.rgb_backbone(dummy_input).shape[-1]

        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, 512), nn.ReLU(), nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_classes)
        )

    def get_logits(self, image):
        # Extract frequency components
        high_dct = self.dct.forward_high(image)
        high_dwt = self.dwt.forward(image)
        high_freq = high_dct * 0.5 + high_dwt * 0.5

        # Process streams separately
        rgb_features = self.rgb_backbone(image)
        freq_features = self.freq_backbone(high_freq)

        # Fuse features
        fused = torch.cat([rgb_features, freq_features], dim=1)
        fused = self.fusion(fused)

        # Classify
        logits = self.classifier(fused)
        return logits


# ============== Baseline 7: Lightweight CNN ==============
class LightweightCNNBaseline(BaselineModel):
    """
    A very simple and fast CNN baseline for quick experiments.
    Much smaller than other models.
    """

    def __init__(self, num_classes=1):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Global pooling
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def get_logits(self, image):
        features = self.features(image)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits


def get_baseline_model(model_name, **kwargs):
    """Factory function to get baseline models by name"""
    models = {
        "convnext": ConvNeXtBaseline,
        "resnet": ResNetBaseline,
        "vit": ViTBaseline,
        "frequency": FrequencyAwareBaseline,
        "efficientnet": EfficientNetBaseline,
        "dualstream": DualStreamBaseline,
        "lightweight": LightweightCNNBaseline,
    }

    if model_name not in models:
        raise ValueError(
            f"Model {model_name} not found. Available: {list(models.keys())}"
        )

    return models[model_name](**kwargs)
