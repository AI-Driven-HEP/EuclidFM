"""
Embedding layer for photometry data - CORRECTED
Now masking happens AFTER embedding
"""
import torch
import torch.nn as nn


class PhotometryEmbedding(nn.Module):
    """
    Embedding layer for photometry time series with separate paths for flux and error.
    """
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
        
        # Mask token for decoder (not used in encoder)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, values, band, survey, type_id, global_feat):
        """
        Forward pass WITHOUT masking - just create embeddings.
        Masking happens AFTER this layer.
        
        Args:
            values: (B, L, 2) - [flux, error] after arcsinh transformation
            band: (B, L) - band IDs
            survey: (B, L) - survey IDs
            type_id: (B, L) - type IDs
            global_feat: (B, global_dim) - global features
        
        Returns:
            x: (B, L, d_model) - Token embeddings
        """
        B, L, _ = values.shape
        
        # Split flux and error
        flux = values[:, :, 0:1]
        error = values[:, :, 1:2]
        
        # Project through separate paths
        flux_feat = self.flux_proj(flux)
        err_feat = self.err_proj(error)
        
        # Embed categorical features
        band_e = self.band_embed(band)
        survey_e = self.survey_embed(survey)
        type_e = self.type_embed(type_id)
        
        # Concatenate and fuse
        features = torch.cat([flux_feat, err_feat, band_e, survey_e, type_e], dim=-1)
        x = self.feature_fuser(features)
        
        # Add global features to each token if enabled
        if self.use_global and not self.use_cls_token:
            g = self.global_proj(global_feat).unsqueeze(1)
            x = x + g
        
        # Add positional embeddings
        pos_emb = self.pos_embed(self.pos_ids[:L].unsqueeze(0))
        x = x + pos_emb
        x = self.norm(x)
        
        return x