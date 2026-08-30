"""MAE-style Transformer: encoder sees only unmasked valid measurements."""

import logging

import torch
import torch.nn as nn

from .embedding import PhotometryEmbedding

logger = logging.getLogger(__name__)


class PhotometryTransformer(nn.Module):
    def __init__(self, cfg, n_band, n_survey, n_type):
        super().__init__()
        self.seq_len = cfg.seq_len
        self.d_model = cfg.d_model
        self.use_cls_token = cfg.use_cls_token
        decoder_layers = getattr(cfg, "decoder_layers", 4)
        
        # Get flags from config
        use_global_features = getattr(cfg, "use_global_features", True)

        self.embedding = PhotometryEmbedding(
            d_model=cfg.d_model,
            n_band=n_band,
            n_survey=n_survey,
            n_type=n_type,
            max_len=cfg.seq_len,
            global_dim=cfg.global_features,
            use_cls_token=cfg.use_cls_token,
            use_global_features=use_global_features,  
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
        self.decoder_position_offset = 1 if cfg.use_cls_token else 0
        self.register_buffer(
            "decoder_pos_ids", torch.arange(cfg.seq_len + 1)
        )
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
        
        logger.info(f"Initialized MAE-style Photometry Transformer")
        logger.info(f"  - Use global features: {use_global_features}")

    def forward(
        self,
        values,
        band,
        survey,
        type_id,
        global_feat,
        valid_mask,
        mask_positions,
        ids_keep,
        ids_shuffle,
    ):
        # ids_shuffle is kept in the public API for compatibility with existing
        # training code and checkpoints. Direct scatter by original indices is
        # less error-prone than scattering and applying a second permutation.
        del mask_positions, ids_shuffle

        encoded, valid_kept = self.embedding(
            values, band, survey, type_id, global_feat, ids_keep
        )
        encoded = self.encoder(encoded, src_key_padding_mask=~valid_kept)
        encoded = self.encoder_norm(encoded)

        if self.use_cls_token:
            cls_embedding = encoded[:, 0]
            encoded_measurements = encoded[:, 1:]
        else:
            cls_embedding = encoded.mean(dim=1)
            encoded_measurements = encoded

        encoded_measurements = self.decoder_embed(encoded_measurements)
        batch_size = values.shape[0]
        restored = self.mask_token.expand(
            batch_size, self.seq_len, self.d_model
        ).clone()

        # Restore each encoded token directly to its original sequence index.
        for row, kept_indices in enumerate(ids_keep):
            number_kept = kept_indices.numel()
            if number_kept:
                restored[row, kept_indices] = encoded_measurements[
                    row, :number_kept
                ]

        decoded = restored + self.decoder_pos_embed(
            self.decoder_pos_ids[
                self.decoder_position_offset:
                self.decoder_position_offset + self.seq_len
            ].unsqueeze(0)
        )

        decoder_padding_mask = ~valid_mask
        rows_without_valid_tokens = decoder_padding_mask.all(dim=1)
        if rows_without_valid_tokens.any() and self.seq_len > 0:
            decoder_padding_mask = decoder_padding_mask.clone()
            decoder_padding_mask[rows_without_valid_tokens, 0] = False

        decoded = self.decoder(
            decoded, src_key_padding_mask=decoder_padding_mask
        )
        decoded = self.decoder_norm(decoded)
        return self.decoder_pred(decoded), cls_embedding
