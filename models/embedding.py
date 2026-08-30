"""Embedding layer for the MAE-style encoder.

Only unmasked valid measurements are sent to the encoder. Their positional
embeddings are gathered from their original sequence positions, rather than
being reassigned according to the compacted/padded order.
"""

import torch
import torch.nn as nn


class PhotometryEmbedding(nn.Module):
    def __init__(
        self,
        d_model,
        n_band,
        n_survey,
        n_type,
        max_len=512,
        cont_dim=64,
        band_dim=8,
        survey_dim=8,
        type_dim=4,
        global_dim=4,
        use_cls_token=True,
        use_global_features=True,  # NEW: Enable/disable global features
    ):
        super().__init__()
        self.d_model = d_model
        self.use_cls_token = use_cls_token
        self.use_global_features = use_global_features  # NEW: Store the flag
        
        self.flux_proj = nn.Sequential(
            nn.Linear(1, cont_dim), nn.GELU(), nn.Linear(cont_dim, cont_dim)
        )
        self.err_proj = nn.Sequential(
            nn.Linear(1, cont_dim), nn.GELU(), nn.Linear(cont_dim, cont_dim)
        )
        self.band_embed = nn.Embedding(n_band, band_dim)
        self.survey_embed = nn.Embedding(n_survey, survey_dim)
        self.type_embed = nn.Embedding(n_type, type_dim)
        combined_dim = 2 * cont_dim + band_dim + survey_dim + type_dim
        self.feature_fuser = nn.Sequential(
            nn.Linear(combined_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # NEW: Only create global projection if global features are used
        if use_global_features:
            self.global_proj = nn.Sequential(
                nn.Linear(global_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.global_proj = None
        
        self.pos_embed = nn.Embedding(max_len + 1, d_model)
        self.register_buffer("pos_ids", torch.arange(max_len + 1))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, values, band, survey, type_id, global_feat, ids_keep):
        batch_size = values.shape[0]
        maximum_kept = max((indices.numel() for indices in ids_keep), default=0)

        padded_values = values.new_zeros((batch_size, maximum_kept, 2))
        padded_band = band.new_zeros((batch_size, maximum_kept))
        padded_survey = survey.new_zeros((batch_size, maximum_kept))
        padded_type = type_id.new_zeros((batch_size, maximum_kept))
        kept_position_ids = band.new_zeros((batch_size, maximum_kept))
        valid_kept = torch.zeros(
            batch_size,
            maximum_kept,
            dtype=torch.bool,
            device=values.device,
        )

        position_offset = 1 if self.use_cls_token else 0
        for row, indices in enumerate(ids_keep):
            number_kept = indices.numel()
            if number_kept:
                padded_values[row, :number_kept] = values[row, indices]
                padded_band[row, :number_kept] = band[row, indices]
                padded_survey[row, :number_kept] = survey[row, indices]
                padded_type[row, :number_kept] = type_id[row, indices]
                kept_position_ids[row, :number_kept] = indices + position_offset
                valid_kept[row, :number_kept] = True

        features = torch.cat(
            [
                self.flux_proj(padded_values[:, :, 0:1]),
                self.err_proj(padded_values[:, :, 1:2]),
                self.band_embed(padded_band),
                self.survey_embed(padded_survey),
                self.type_embed(padded_type),
            ],
            dim=-1,
        )
        embedded = self.feature_fuser(features)
        if maximum_kept:
            embedded = embedded + self.pos_embed(kept_position_ids)

        if self.use_cls_token:
            cls = self.cls_token.expand(batch_size, 1, -1)
            
            # NEW: Only add global features if enabled
            if self.use_global_features and self.global_proj is not None:
                cls = cls + self.global_proj(global_feat).unsqueeze(1)
            
            cls = cls + self.pos_embed(
                torch.zeros(batch_size, 1, dtype=torch.long, device=values.device)
            )
            embedded = torch.cat([cls, embedded], dim=1)
            cls_valid = torch.ones(
                batch_size, 1, dtype=torch.bool, device=values.device
            )
            valid_kept = torch.cat([cls_valid, valid_kept], dim=1)
        else:
            # NEW: Only add global features if enabled
            if self.use_global_features and self.global_proj is not None:
                embedded = embedded + self.global_proj(global_feat).unsqueeze(1)

        return self.norm(embedded), valid_kept
