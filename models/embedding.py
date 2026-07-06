"""
Embedding layer for photometry data.

Important masking design:
    This module only converts each measurement into an embedding.
    It does NOT replace masked measurements with a mask token.
    The MAE-style removal/restoration logic is handled in transformer.py.
"""
import torch
import torch.nn as nn


class PhotometryEmbedding(nn.Module):
    """
    Embedding layer for photometry measurements with separate paths for flux and error.

    Each measurement has:
        - continuous values: flux and error
        - categorical IDs: band, survey, type

    The output still has one token per original measurement position.
    Masked measurements are removed later, before the encoder, in PhotometryTransformer.
    """

    def __init__(self, d_model, n_band, n_survey, n_type, max_len=512,
                 cont_dim=64, band_dim=8, survey_dim=8, type_dim=4,
                 use_global=True, global_dim=4, use_cls_token=True):
        super().__init__()
        self.d_model = d_model
        self.use_global = use_global
        self.use_cls_token = use_cls_token

        # Separate paths for flux and error.
        # This keeps flux and uncertainty information independent before fusion.
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

        # Categorical embeddings for metadata of each measurement.
        self.band_embed = nn.Embedding(n_band, band_dim)
        self.survey_embed = nn.Embedding(n_survey, survey_dim)
        self.type_embed = nn.Embedding(n_type, type_dim)

        # Fuse flux, error, band, survey, and type into one d_model-dimensional token.
        total_dim = cont_dim + cont_dim + band_dim + survey_dim + type_dim
        self.feature_fuser = nn.Sequential(
            nn.Linear(total_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Project global object-level features, used only to enrich the CLS token.
        if use_global:
            self.global_proj = nn.Sequential(
                nn.Linear(global_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            )

        # Encoder-side positional embeddings.
        # +1 is for the optional CLS token.
        self.max_len = max_len
        self.pos_embed = nn.Embedding(max_len + 1, d_model)
        self.register_buffer("pos_ids", torch.arange(max_len + 1), persistent=False)

        # CLS token gives one global representation per object.
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, values, band, survey, type_id, global_feat):
        """
        Embed all original measurement positions.

        Args:
            values: (B, L, 2), transformed [flux, error]
            band: (B, L), band IDs
            survey: (B, L), survey IDs
            type_id: (B, L), type IDs
            global_feat: (B, global_dim), object-level features

        Returns:
            x: (B, L + 1, d_model) if CLS is used, otherwise (B, L, d_model)

        Note:
            No masking happens here. The encoder must not see the masked measurements;
            therefore masking/removal is done after this embedding step in transformer.py.
        """
        B, L, _ = values.shape

        # Split flux and error into two one-dimensional inputs.
        flux = values[:, :, 0:1]
        error = values[:, :, 1:2]

        # Independent continuous projections.
        flux_feat = self.flux_proj(flux)
        err_feat = self.err_proj(error)

        # Categorical projections.
        band_e = self.band_embed(band)
        survey_e = self.survey_embed(survey)
        type_e = self.type_embed(type_id)

        # Measurement token before positional information.
        features = torch.cat([flux_feat, err_feat, band_e, survey_e, type_e], dim=-1)
        x = self.feature_fuser(features)

        # Optional CLS token. Global object features are added to CLS only.
        if self.use_cls_token:
            cls = self.cls_token.expand(B, 1, -1)
            if self.use_global:
                g = self.global_proj(global_feat)
                cls = cls + g.unsqueeze(1)
            x = torch.cat([cls, x], dim=1)
        else:
            if self.use_global:
                g = self.global_proj(global_feat).unsqueeze(1)
                x = x + g

        # Add encoder-side positional embeddings.
        seq_len = x.size(1)
        pos_emb = self.pos_embed(self.pos_ids[:seq_len].unsqueeze(0))
        x = self.norm(x + pos_emb)

        return x
