"""timm-based species classifier."""

from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn


class SpeciesClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        *,
        pretrained: bool = True,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        if hasattr(self.backbone, "forward_head"):
            try:
                feats = self.backbone.forward_head(feats, pre_logits=True)
            except TypeError:
                pass
        if feats.ndim > 2:
            feats = feats.mean(dim=tuple(range(2, feats.ndim)))
        return feats

    def freeze_backbone(self) -> None:
        for name, param in self.backbone.named_parameters():
            if not name.startswith("head") and not name.startswith("classifier") and "fc" not in name.split(".")[-1]:
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


def build_classifier(config: dict[str, Any], num_classes: int) -> SpeciesClassifier:
    model_cfg = config.get("model", {})
    return SpeciesClassifier(
        model_name=model_cfg.get("name", "convnext_tiny.fb_in22k_ft_in1k"),
        num_classes=num_classes,
        pretrained=bool(model_cfg.get("pretrained", True)),
        drop_rate=float(model_cfg.get("drop_rate", 0.0)),
    )
