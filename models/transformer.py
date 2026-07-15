"""
Transformer model with fixed masking (Baseline Version v1)

Key characteristics (BERT-style masking):
1. All tokens are passed to the encoder (including masked ones)
2. Mask tokens are replaced with mask_token in the embedding layer
3. Decoder is a simple MLP (no transformer decoder)
4. Loss computed only on masked positions
"""

import logging
import torch
import torch.nn as nn

from .embedding import PhotometryEmbedding

logger = logging.getLogger(__name__)


class PhotometryTransformer(nn.Module):
    """
    Transformer model for photometry time series with masked autoencoding.
    Uses BERT-style masking: mask tokens remain in the encoder input.
    """
    def __init__(self, cfg, n_band, n_survey, n_type):
        super().__init__()
        logger.info("Initializing Transformer (Baseline v1 - BERT-style masking)...")
        logger.info(f"CLS token enabled: {cfg.use_cls_token}")
        logger.info(f"Global features dimension: {cfg.global_features}")
        logger.info("Using separate paths for flux and error")
        logger.info("MASK tokens remain in encoder input (BERT-style)")

        self.embedding = PhotometryEmbedding(
            d_model=cfg.d_model,
            n_band=n_band,
            n_survey=n_survey,
            n_type=n_type,
            max_len=cfg.seq_len,
            use_global=True,
            global_dim=cfg.global_features,
            use_cls_token=cfg.use_cls_token
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

        # Decoder: Simple 3-layer MLP (no transformer decoder)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 2)
        )

        self.use_cls_token = cfg.use_cls_token
        logger.info("Model ready.")

    def forward(self, values, band, survey, type_id, global_feat, valid_mask, mask_positions):
        """
        Forward pass with BERT-style masking.

        Args:
            values: (B, L, 2) - [flux, error] after arcsinh
            band: (B, L) - band IDs
            survey: (B, L) - survey IDs
            type_id: (B, L) - type IDs
            global_feat: (B, global_dim) - global features
            valid_mask: (B, L) - True where measurement is valid
            mask_positions: (B, L) - True where to mask

        Returns:
            pred: (B, L, 2) - predicted [flux, error]
            cls: (B, d_model) - CLS token for downstream tasks
        """
        B = values.shape[0]
        x = self.embedding(values, band, survey, type_id, global_feat, mask_positions)

        if self.use_cls_token:
            cls_ignore = torch.zeros(B, 1, device=valid_mask.device, dtype=torch.bool)
            src_key_padding_mask = torch.cat([cls_ignore, ~valid_mask], dim=1)
        else:
            src_key_padding_mask = ~valid_mask

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        if self.use_cls_token:
            pred = self.decoder(x[:, 1:, :])
            cls = x[:, 0, :]
        else:
            pred = self.decoder(x)
            cls = x.mean(dim=1)

        return pred, cls


def reconstruction_loss(pred, target, mask_positions, valid_mask):
    """
    MSE loss ONLY on masked positions.
    """
    if mask_positions.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    diff = (pred - target).pow(2).mean(dim=-1)
    masked_diff = diff[mask_positions]

    return masked_diff.mean()
