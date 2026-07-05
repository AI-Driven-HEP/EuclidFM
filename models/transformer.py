"""
Transformer model for photometry time series - CORRECTED VERSION
Correct sequence: Embedding → Mask → Encoder → Restore → Decoder
"""
import logging
import torch
import torch.nn as nn
from .embedding import PhotometryEmbedding
from .masking import random_masking_embeddings

logger = logging.getLogger(__name__)


class PhotometryTransformer(nn.Module):
    """
    Transformer model with correct masking sequence:
    1. Embedding → Convert raw measurements to embeddings
    2. Random Masking → Remove tokens from valid positions
    3. Encoder → Process unmasked tokens
    4. Restore → Put tokens back with mask tokens
    5. Decoder → Reconstruct all tokens
    """
    def __init__(self, cfg, n_band, n_survey, n_type):
        super().__init__()
        logger.info("Initializing Transformer with correct masking sequence...")
        logger.info(f"CLS token enabled: {cfg.use_cls_token}")
        logger.info(f"Global features dimension: {cfg.global_features}")
        logger.info(f"Using separate paths for flux and error")
        
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.d_model = cfg.d_model
        self.use_cls_token = cfg.use_cls_token
        
        # 1. Embedding layer
        self.embedding = PhotometryEmbedding(
            d_model=cfg.d_model, 
            n_band=n_band, 
            n_survey=n_survey, 
            n_type=n_type,
            max_len=cfg.seq_len, 
            use_global=True, 
            global_dim=cfg.global_features,
            use_cls_token=False  # CLS token will be added separately
        )
        
        # 2. Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, 
            nhead=cfg.n_heads, 
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, cfg.n_layers)
        self.encoder_norm = nn.LayerNorm(cfg.d_model)
        
        # 3. Decoder
        self.decoder_embed = nn.Linear(cfg.d_model, cfg.d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, cfg.n_layers)
        self.decoder_norm = nn.LayerNorm(cfg.d_model)
        
        # 4. Mask token for decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # 5. CLS token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            nn.init.normal_(self.cls_token, std=0.02)
            self.global_proj = nn.Sequential(
                nn.Linear(cfg.global_features, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, cfg.d_model)
            )
        
        # 6. Prediction head (3-layer MLP)
        self.pred_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 2)
        )
        
        logger.info("Model ready.")

    def forward(self, values, band, survey, type_id, global_feat, valid_mask):
        """
        Forward pass with correct masking sequence.
        
        Sequence:
        1. Embedding → (B, L, D)
        2. Random Masking → Remove tokens based on mask_ratio
        3. Encoder → Process unmasked tokens
        4. Restore + Mask Tokens → Fill removed positions
        5. Decoder → Reconstruct all positions
        6. Prediction → Predict original values
        
        Args:
            values: (B, L, 2) - [flux, error]
            band: (B, L) - band IDs
            survey: (B, L) - survey IDs
            type_id: (B, L) - type IDs
            global_feat: (B, global_dim) - global features
            valid_mask: (B, L) - True where measurement is valid
        
        Returns:
            pred: (B, L, 2) - predicted [flux, error]
            cls_token: (B, d_model) - CLS token for downstream tasks
            mask: (B, L) - Binary mask (1=masked, 0=kept)
        """
        B, L, _ = values.shape
        device = values.device
        
        # ========== STEP 1: EMBEDDING ==========
        x = self.embedding(values, band, survey, type_id, global_feat)  # (B, L, D)
        
        # ========== STEP 2: RANDOM MASKING ==========
        # Apply masking to valid positions only
        mask_ratio = self.cfg.num_mask / self.seq_len        x_masked, mask, ids_keep, ids_shuffle = random_masking_embeddings(x, mask_ratio)
        # x_masked: (B, N_keep, D) - only unmasked tokens
        # mask: (B, L) - 1=masked, 0=kept
        # ids_keep: (B, N_keep) - indices of kept tokens
        # ids_shuffle: (B, L) - shuffled indices for restoration
        
        # ========== STEP 3: ENCODER ==========
        # Create padding mask for valid positions
        valid_mask_keep = valid_mask.gather(dim=1, index=ids_keep)
        src_key_padding_mask = ~valid_mask_keep
        
        # Pass through encoder
        x_encoded = self.encoder(x_masked, src_key_padding_mask=src_key_padding_mask)
        x_encoded = self.encoder_norm(x_encoded)  # (B, N_keep, D)
        
        # ========== STEP 4: RESTORE + MASK TOKENS ==========
        # Prepare for decoder: need to restore full sequence
        # Create tensor with mask tokens for all positions
        mask_tokens = self.mask_token.repeat(B, L, 1)  # (B, L, D)
        
        # Fill in encoded tokens at their original positions
        # First, scatter encoded tokens back to their positions
        x_restored = mask_tokens.clone()
        # Use scatter to place encoded tokens at their keep positions
        for b in range(B):
            x_restored[b, ids_keep[b]] = x_encoded[b]
        
        # Restore original ordering
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        x_restored = torch.gather(
            x_restored, 
            dim=1, 
            index=ids_restore.unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        
        # ========== STEP 5: DECODER ==========
        # Create padding mask for decoder
        tgt_key_padding_mask = ~valid_mask
        
        # Add CLS token if enabled
        if self.use_cls_token:
            cls = self.cls_token.expand(B, 1, -1)
            g = self.global_proj(global_feat)
            cls = cls + g.unsqueeze(1)
            x_decoder_input = torch.cat([cls, x_restored], dim=1)
            tgt_key_padding_mask = torch.cat([
                torch.zeros(B, 1, device=device, dtype=torch.bool),
                tgt_key_padding_mask
            ], dim=1)
        else:
            x_decoder_input = x_restored
        
        # Decoder self-attention uses x_decoder_input
        # Cross-attention uses x_encoded (or x_masked) from encoder
        x_decoded = self.decoder(
            tgt=x_decoder_input,
            memory=x_masked,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=~valid_mask_keep
        )
        x_decoded = self.decoder_norm(x_decoded)
        
        # ========== STEP 6: PREDICTION ==========
        if self.use_cls_token:
            # Remove CLS token for prediction
            x_for_pred = x_decoded[:, 1:, :]  # (B, L, D)
            cls_token = x_decoded[:, 0, :]    # (B, D)
        else:
            x_for_pred = x_decoded
            cls_token = x_decoded.mean(dim=1)  # (B, D)
        
        # Predict flux and error
        pred = self.pred_head(x_for_pred)  # (B, L, 2)
        
        return pred, cls_token, mask
    
    def reconstruct(self, values, band, survey, type_id, global_feat, valid_mask):
        """Reconstruction with fixed masking."""
        return self.forward(values, band, survey, type_id, global_feat, valid_mask)