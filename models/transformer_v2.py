"""MAE-style v2 Transformer: encoder sees only unmasked valid measurements."""

import logging

import torch
import torch.nn as nn

from .embedding_v2 import PhotometryEmbeddingV2

logger = logging.getLogger(__name__)


class PhotometryTransformerV2(nn.Module):
    def __init__(self, cfg, n_band, n_survey, n_type):
        super().__init__()
        self.seq_len = cfg.seq_len
        self.d_model = cfg.d_model
        self.use_cls_token = cfg.use_cls_token
        decoder_layers = getattr(cfg, "decoder_layers", 4)

        self.embedding = PhotometryEmbeddingV2(
            d_model=cfg.d_model,
            n_band=n_band,
            n_survey=n_survey,
            n_type=n_type,
            max_len=cfg.seq_len,
            global_dim=cfg.global_features,
            use_cls_token=cfg.use_cls_token,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, cfg.n_layers)
        self.encoder_norm = nn.LayerNorm(cfg.d_model)
        self.decoder_embed = nn.Linear(cfg.d_model, cfg.d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder_pos_embed = nn.Embedding(cfg.seq_len + 1, cfg.d_model)
        self.register_buffer("decoder_pos_ids", torch.arange(cfg.seq_len + 1))
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, decoder_layers)
        self.decoder_norm = nn.LayerNorm(cfg.d_model)
        self.decoder_pred = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 2),
        )
        logger.info("Initialized MAE-style Photometry Transformer v2")

    def forward(self, values, band, survey, type_id, global_feat, valid_mask,
                mask_positions, ids_keep, ids_shuffle):
        encoded, valid_kept = self.embedding(
            values, band, survey, type_id, global_feat, ids_keep
        )
        encoded = self.encoder(encoded, src_key_padding_mask=~valid_kept)
        encoded = self.encoder_norm(encoded)

        if self.use_cls_token:
            cls_embedding = encoded[:, 0]
            encoded = encoded[:, 1:]
        else:
            cls_embedding = encoded.mean(dim=1)

        encoded = self.decoder_embed(encoded)
        restored_batch = []
        for row, kept_indices in enumerate(ids_keep):
            restored = self.mask_token.expand(1, self.seq_len, -1).squeeze(0).clone()
            if len(kept_indices):
                restored[kept_indices] = encoded[row, :len(kept_indices)]
            ids_restore = torch.argsort(ids_shuffle[row])
            restored_batch.append(restored[ids_restore])

        decoded = torch.stack(restored_batch)
        decoded += self.decoder_pos_embed(
            self.decoder_pos_ids[:self.seq_len].unsqueeze(0)
        )
        decoded = self.decoder(decoded, src_key_padding_mask=~valid_mask)
        decoded = self.decoder_norm(decoded)
        return self.decoder_pred(decoded), cls_embedding

