"""Encoder embedding that sends only unmasked measurements into the v2 encoder."""

import torch
import torch.nn as nn


class PhotometryEmbeddingV2(nn.Module):
    def __init__(self, d_model, n_band, n_survey, n_type, max_len=512,
                 cont_dim=64, band_dim=8, survey_dim=8, type_dim=4,
                 global_dim=4, use_cls_token=True):
        super().__init__()
        self.d_model = d_model
        self.use_cls_token = use_cls_token
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
            nn.Linear(combined_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.pos_embed = nn.Embedding(max_len + 1, d_model)
        self.register_buffer("pos_ids", torch.arange(max_len + 1))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, values, band, survey, type_id, global_feat, ids_keep):
        batch_size = values.shape[0]
        maximum_kept = max((len(indices) for indices in ids_keep), default=0)
        padded_values = torch.zeros(batch_size, maximum_kept, 2, device=values.device)
        padded_band = torch.zeros(batch_size, maximum_kept, dtype=torch.long, device=values.device)
        padded_survey = torch.zeros_like(padded_band)
        padded_type = torch.zeros_like(padded_band)
        valid_kept = torch.zeros(batch_size, maximum_kept, dtype=torch.bool, device=values.device)

        for row, indices in enumerate(ids_keep):
            number_kept = len(indices)
            if number_kept:
                padded_values[row, :number_kept] = values[row, indices]
                padded_band[row, :number_kept] = band[row, indices]
                padded_survey[row, :number_kept] = survey[row, indices]
                padded_type[row, :number_kept] = type_id[row, indices]
                valid_kept[row, :number_kept] = True

        features = torch.cat([
            self.flux_proj(padded_values[:, :, 0:1]),
            self.err_proj(padded_values[:, :, 1:2]),
            self.band_embed(padded_band),
            self.survey_embed(padded_survey),
            self.type_embed(padded_type),
        ], dim=-1)
        embedded = self.feature_fuser(features)

        if self.use_cls_token:
            cls = self.cls_token.expand(batch_size, 1, -1)
            cls = cls + self.global_proj(global_feat).unsqueeze(1)
            embedded = torch.cat([cls, embedded], dim=1)
            cls_valid = torch.ones(batch_size, 1, dtype=torch.bool, device=values.device)
            valid_kept = torch.cat([cls_valid, valid_kept], dim=1)
        else:
            embedded = embedded + self.global_proj(global_feat).unsqueeze(1)

        embedded += self.pos_embed(self.pos_ids[:embedded.size(1)].unsqueeze(0)) #the positional is added after masking????
        return self.norm(embedded), valid_kept

