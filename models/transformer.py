"""
Based on the original notebook: pretraining_fixed_mask_final.ipynb
"""
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PhotometryEmbedding(nn.Module):

    def __init__(self, d_model, n_band, n_survey, n_type, max_len=512,
                 cont_dim=64, band_dim=8, survey_dim=8, type_dim=4,
                 use_global=True, global_dim=4, use_cls_token=True):
        super().__init__()
        self.d_model = d_model
        self.use_global = use_global
        self.use_cls_token = use_cls_token
        
        # SEPARATE paths for flux and error (each 1-dimensional input)
        self.flux_proj = nn.Sequential(
            nn.Linear(1, cont_dim), 
            nn.GELU(), 
            nn.Linear(cont_dim, cont_dim)
        )
        
        self.err_proj = nn.Sequential(
            nn.Linear(1, cont_dim), 
            nn.GELU(), 
            nn.Linear(cont_dim, cont_dim)
        )
        
        self.band_embed = nn.Embedding(n_band, band_dim)
        self.survey_embed = nn.Embedding(n_survey, survey_dim)
        self.type_embed = nn.Embedding(n_type, type_dim)
        
        # Total dimension: flux_proj(64) + err_proj(64) + band(8) + survey(8) + type(4) = 148
        total_dim = cont_dim + cont_dim + band_dim + survey_dim + type_dim
        self.feature_fuser = nn.Sequential(
            nn.Linear(total_dim, d_model), 
            nn.GELU(), 
            nn.Linear(d_model, d_model)
        )
        
        if use_global:
            self.global_proj = nn.Sequential(
                nn.Linear(global_dim, d_model), 
                nn.GELU(), 
                nn.Linear(d_model, d_model)
            )
        
        self.max_len = max_len
        self.pos_embed = nn.Embedding(max_len + 1, d_model)
        self.register_buffer("pos_ids", torch.arange(max_len + 1))
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, values, band, survey, type_id, global_feat, mask_positions):
        """
        Forward pass with separate paths for flux and error.
        
        Args:
            values: (B, L, 2) - [flux, error] after arcsinh transformation
            band: (B, L) - band IDs
            survey: (B, L) - survey IDs
            type_id: (B, L) - type IDs
            global_feat: (B, global_dim) - global features
            mask_positions: (B, L) - positions to mask with mask_token
        
        Returns:
            x: (B, L + 1, d_model) if use_cls_token else (B, L, d_model)
        """
        B, L, _ = values.shape
        
        # Split flux and error (each becomes (B, L, 1))
        flux = values[:, :, 0:1]  # Keep last dimension
        error = values[:, :, 1:2]
        
        # Project through separate paths
        flux_feat = self.flux_proj(flux)      # (B, L, cont_dim)
        err_feat = self.err_proj(error)       # (B, L, cont_dim)
        
        # Embed categorical features
        band_e = self.band_embed(band)        # (B, L, band_dim)
        survey_e = self.survey_embed(survey)  # (B, L, survey_dim)
        type_e = self.type_embed(type_id)     # (B, L, type_dim)
        
        # Concatenate all features
        features = torch.cat([flux_feat, err_feat, band_e, survey_e, type_e], dim=-1)
        # shape: (B, L, cont_dim*2 + band_dim + survey_dim + type_dim)
        
        # Fuse into d_model dimension
        x = self.feature_fuser(features)      # (B, L, d_model)
        
        # Apply mask token to masked positions
        if mask_positions is not None:
            mask_token = self.mask_token.expand(B, L, -1)
            x = torch.where(mask_positions.unsqueeze(-1), mask_token, x)
        
        # Add CLS token if enabled
        if self.use_cls_token:
            cls = self.cls_token.expand(B, 1, -1)
            if self.use_global:
                g = self.global_proj(global_feat)
                cls = cls + g.unsqueeze(1)
            x = torch.cat([cls, x], dim=1)    # (B, L+1, d_model)
        else:
            if self.use_global:
                g = self.global_proj(global_feat).unsqueeze(1)
                x = x + g
        
        # Add positional embeddings
        seq_len = x.size(1)
        pos_emb = self.pos_embed(self.pos_ids[:seq_len].unsqueeze(0))
        x = x + pos_emb
        x = self.norm(x)
        
        return x


class PhotometryTransformer(nn.Module):
    """
    Transformer model for photometry time series with masked autoencoding.
    
    Uses separate embedding paths for flux and error to improve error reconstruction.
    Decoder is implemented as an MLP with at least 3 layers.
    Masking happens inside the embedding layer (before encoder).
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
        
        # ========== Decoder as MLP with 3 layers ==========
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
