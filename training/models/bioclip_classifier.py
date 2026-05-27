"""BioCLIP-backed classifier for the two-stage upgrade track.

Stage A (pretrain): species-level head over the PlantNet pretraining pool.
Stage B (finetune): replace the head with the 47-class output head.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class BioClipClassifier(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        model_name: str = "hf-hub:imageomics/bioclip",
        pretrained: bool = True,
        drop_rate: float = 0.1,
        normalize_features: bool = True,
    ) -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "BioCLIP training requires open-clip-torch. "
                "Install training/requirements-bioclip.txt on the RunPod image."
            ) from exc

        self.model_name = model_name
        self.normalize_features = normalize_features
        self.backbone, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        embed_dim = self.backbone.visual.output_dim
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.encode_image(x)
        if self.normalize_features:
            features = F.normalize(features, dim=-1)
        return self.head(features)

    def replace_head(self, num_classes: int, *, drop_rate: float = 0.1) -> None:
        embed_dim = self.head[-1].in_features
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, num_classes),
        )

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True


def build_bioclip_classifier(config: dict[str, Any], num_classes: int) -> BioClipClassifier:
    model_cfg = config.get("model", {})
    return BioClipClassifier(
        num_classes=num_classes,
        model_name=model_cfg.get("bioclip_name", "hf-hub:imageomics/bioclip"),
        pretrained=bool(model_cfg.get("pretrained", True)),
        drop_rate=float(model_cfg.get("drop_rate", 0.1)),
        normalize_features=bool(model_cfg.get("normalize_features", True)),
    )
