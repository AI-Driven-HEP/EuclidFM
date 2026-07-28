"""Three-class classifier built on the pretrained EuclidFM v2 encoder."""

import torch
import torch.nn as nn

from .transformer_v2 import PhotometryTransformerV2


class EuclidFMClassifierV2(nn.Module):
    def __init__(self, cfg, n_band, n_survey, n_type):
        super().__init__()
        self.backbone = PhotometryTransformerV2(
            cfg, n_band=n_band, n_survey=n_survey, n_type=n_type
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(0.20),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )

    def encode(self, values, band, survey, type_id, global_feat, valid_mask):
        """Encode every valid measurement and return the CLS representation."""
        ids_keep = [
            torch.where(row)[0].to(values.device) for row in valid_mask
        ]
        encoded, kept_mask = self.backbone.embedding(
            values, band, survey, type_id, global_feat, ids_keep
        )
        encoded = self.backbone.encoder(
            encoded, src_key_padding_mask=~kept_mask
        )
        encoded = self.backbone.encoder_norm(encoded)
        return encoded[:, 0]

    def forward(self, values, band, survey, type_id, global_feat, valid_mask):
        cls_embedding = self.encode(
            values, band, survey, type_id, global_feat, valid_mask
        )
        return self.classifier(cls_embedding)

    def freeze_encoder(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

