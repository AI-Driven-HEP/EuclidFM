"""
Photometry Transformer model
"""
import logging
import torch
import torch.nn as nn
from .embedding import PhotometryEmbedding

logger = logging.getLogger(__name__)


class PhotometryTransformer(nn.Module):
    """
    Transformer model for photometry time series with masked autoencoding.
    
    Uses separate embedding paths for flux and error to improve error reconstruction.
    Decoder is implemented as an MLP with at least 3 layers.
    """
    def __init__(self, cfg, n_band, n_survey, n_type):
        super().__init__()
        logger.info("Initializing Transformer...")
        logger.info(f"CLS token enabled: {cfg.use_cls_token}")
        logger.info(f"Global features dimension: {cfg.global_features}")
        logger.info(f"Using separate paths for flux and error")
        
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
        
        # Decoder as MLP with 3 layers
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
        Forward pass with correct key padding mask.
        
        Args:
            values: (B, L, 2) - [flux, error] after arcsinh
            band: (B, L) - band IDs
            survey: (B, L) - survey IDs
            type_id: (B, L) - type IDs
            global_feat: (B, global_dim) - global features
            valid_mask: (B, L) - True where measurement is valid
            mask_positions: (B, L) - True where to mask (for training)
        
        Returns:
            pred: (B, L, 2) - predicted [flux, error]
            cls: (B, d_model) - CLS token for downstream tasks
        """
        B = values.shape[0]
        x = self.embedding(values, band, survey, type_id, global_feat, mask_positions)
        
        if self.use_cls_token:
            # Create padding mask: True for positions to ignore in attention
            # cls_ignore: never ignore the CLS token (all False)
            cls_ignore = torch.zeros(B, 1, device=valid_mask.device, dtype=torch.bool)
            # For measurements, ignore if NOT valid (~valid_mask)
            src_key_padding_mask = torch.cat([cls_ignore, ~valid_mask], dim=1)
        else:
            src_key_padding_mask = ~valid_mask
        
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        
        if self.use_cls_token:
            # Use MLP decoder instead of linear head
            pred = self.decoder(x[:, 1:, :])  # Remove CLS token for prediction
            cls = x[:, 0, :]                  # CLS token for downstream tasks
        else:
            pred = self.decoder(x)
            cls = x.mean(dim=1)
        
        return pred, cls
