"""
Photometry Transformer model.

MAE-style masking requested by the supervisor:

    1. Embed all measurements.
    2. Randomly remove/mask 20 VALID measurements.
    3. Give ONLY the unmasked valid measurements to the encoder.
    4. After the encoder, add mask tokens back.
    5. Restore everything to the original measurement positions.
    6. Decoder predicts the full sequence.
    7. Loss is computed only on the masked positions.

This follows the same logic as:

    random_masking()
    forward_encoder()
    forward_decoder()

from the advisor's example.
"""

import logging
import torch
import torch.nn as nn

from .embedding import PhotometryEmbedding

logger = logging.getLogger(__name__)


class PhotometryTransformer(nn.Module):
    """
    Transformer model for photometry masked autoencoding.

    Important:
        The encoder must NOT see the masked measurements.
        Masked measurements are removed before the encoder.
        Mask tokens are inserted only after the encoder.
    """

    def __init__(self, cfg, n_band, n_survey, n_type):
        super().__init__()

        logger.info("Initializing Transformer...")
        logger.info(f"CLS token enabled: {cfg.use_cls_token}")
        logger.info("Using MAE-style remove-and-restore masking")

        self.use_cls_token = cfg.use_cls_token
        self.seq_len = cfg.seq_len
        self.d_model = cfg.d_model

        # Embedding layer.
        # This only converts measurements into tokens.
        # It should NOT do masking.
        self.embedding = PhotometryEmbedding(
            d_model=cfg.d_model,
            n_band=n_band,
            n_survey=n_survey,
            n_type=n_type,
            max_len=cfg.seq_len,
            use_global=True,
            global_dim=cfg.global_features,
            use_cls_token=cfg.use_cls_token,
        )

        # Encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, cfg.n_layers)
        self.encoder_norm = nn.LayerNorm(cfg.d_model)

        # Mask token.
        # This is used AFTER the encoder to fill the removed positions.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # Decoder.
        # Kept the same MLP decoder structure.
        self.decoder_pred = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 2),
        )

        logger.info("Model ready.")

    def random_masking_fixed(self, x, valid_mask, mask_positions):
        """
        Remove masked valid measurements before the encoder.

        This is the photometry version of the advisor's random_masking().

        Args:
            x:
                Embedded measurement tokens.
                Shape: (B, N, D)

            valid_mask:
                True where the measurement is real/valid.
                Shape: (B, N)

            mask_positions:
                True where the selected 20 valid measurements are masked.
                Shape: (B, N)

        Returns:
            x_masked:
                Only unmasked valid measurement tokens.
                Shape: (B, K, D)

            mask:
                Mask for loss computation.
                1 means masked, 0 means unmasked.
                Shape: (B, N)

            ids_keep:
                Original indices of kept measurements.
                Shape: (B, K)

            keep_padding_mask:
                Padding mask for the encoder.
                True means ignore this token.
                Shape: (B, K)
        """

        B, N, D = x.shape
        device = x.device

        # Safety:
        # Never mask invalid measurements.
        mask_positions = mask_positions & valid_mask

        # Keep only valid measurements that are NOT masked.
        keep_positions = valid_mask & (~mask_positions)

        # Number of kept measurements per object.
        # Example: if N=94 and 20 valid measurements are masked,
        # then len_keep = 74.
        len_keep = keep_positions.sum(dim=1)

        # Different objects may have different numbers of valid measurements.
        # So we use the maximum kept length in the batch.
        max_len_keep = len_keep.max().clamp_min(1)

        # Random noise for shuffling.
        noise = torch.rand(B, N, device=device)

        # We want the order to be:
        #   1. kept valid measurements
        #   2. masked valid measurements
        #   3. invalid measurements
        #
        # This is similar to ids_shuffle in the advisor's code,
        # but it respects valid_mask.
        scores = noise.clone()
        scores = scores + (~keep_positions).float() * 10.0
        scores = scores + mask_positions.float() * 5.0
        scores = scores + (~valid_mask).float() * 20.0

        # Randomly shuffled indices.
        ids_shuffle = torch.argsort(scores, dim=1)

        # Keep only the unmasked valid measurements.
        ids_keep = ids_shuffle[:, :max_len_keep]

        # Gather kept tokens.
        x_masked = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D),
        )

        # Padding mask:
        # Some objects may have fewer kept valid measurements than max_len_keep.
        arange_keep = torch.arange(max_len_keep, device=device).unsqueeze(0)
        keep_padding_mask = arange_keep >= len_keep.unsqueeze(1)

        # For loss computation:
        # 1 means this measurement was masked.
        # 0 means it was not masked.
        mask = mask_positions.float()

        return x_masked, mask, ids_keep, keep_padding_mask

    def forward_encoder(self, values, band, survey, type_id, global_feat, valid_mask, mask_positions):
        """
        Encoder forward pass.

        Steps:
            1. Embed all measurements.
            2. Remove the masked valid measurements.
            3. Encoder sees only unmasked valid measurements.

        This matches the advisor's:

            x = patch_embed(data)
            x_masked, mask, ids_keep, ids_shuffle = random_masking(x, mask_ratio)
            x_masked = encoder(x_masked)
        """

        B = values.shape[0]

        # 1. Embed all measurements.
        # No masking is done inside the embedding layer.
        x = self.embedding(values, band, survey, type_id, global_feat)

        # Separate CLS token from measurement tokens.
        if self.use_cls_token:
            cls_token = x[:, :1, :]
            x_meas = x[:, 1:, :]
        else:
            cls_token = None
            x_meas = x

        # 2. Remove masked valid measurements before encoder.
        x_masked, mask, ids_keep, keep_padding_mask = self.random_masking_fixed(
            x=x_meas,
            valid_mask=valid_mask,
            mask_positions=mask_positions,
        )

        # 3. Add CLS token back only for encoder input.
        if self.use_cls_token:
            x_encoder = torch.cat([cls_token, x_masked], dim=1)

            cls_ignore = torch.zeros(
                B, 1,
                device=values.device,
                dtype=torch.bool,
            )

            src_key_padding_mask = torch.cat(
                [cls_ignore, keep_padding_mask],
                dim=1,
            )
        else:
            x_encoder = x_masked
            src_key_padding_mask = keep_padding_mask

        # 4. Encoder sees only visible/unmasked valid measurements.
        x_encoder = self.encoder(
            x_encoder,
            src_key_padding_mask=src_key_padding_mask,
        )
        x_encoder = self.encoder_norm(x_encoder)

        # Separate CLS output from measurement outputs.
        if self.use_cls_token:
            cls = x_encoder[:, 0, :]
            x_encoded = x_encoder[:, 1:, :]
        else:
            x_encoded = x_encoder

            # CLS-like representation from visible tokens only.
            nonpad = (~keep_padding_mask).unsqueeze(-1).to(x_encoded.dtype)
            denom = nonpad.sum(dim=1).clamp_min(1.0)
            cls = (x_encoded * nonpad).sum(dim=1) / denom

        return x_encoded, mask, ids_keep, keep_padding_mask, cls

    def forward_decoder(self, x, ids_keep, keep_padding_mask, seq_len):
        """
        Decoder forward pass.

        This follows the advisor's idea:

            x = decoder_embed(x)
            mask_tokens = self.mask_token.repeat(...)
            x_ = torch.cat([x, mask_tokens], dim=1)
            x_ = restore original ordering
            logits = decoder_pred(x_)

        In our case, because different objects can have different valid masks,
        we restore with scatter instead of a Python loop.
        """

        B, K, D = x.shape
        device = x.device

        # Start with mask tokens everywhere.
        # This means every position is hidden by default.
        x_full = self.mask_token.expand(B, seq_len, D).clone()

        # Do not scatter padded encoder tokens.
        x_visible = x.masked_fill(keep_padding_mask.unsqueeze(-1), 0.0)

        # Put the encoded visible measurements back into their original positions.
        x_full = x_full.scatter(
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D),
            src=torch.where(
                keep_padding_mask.unsqueeze(-1),
                self.mask_token.expand(B, K, D),
                x_visible,
            ),
        )

        # Decoder predicts flux and error for every original measurement position.
        logits = self.decoder_pred(x_full)

        return logits

    def forward(self, values, band, survey, type_id, global_feat, valid_mask, mask_positions):
        """
        Full forward pass.

        Args:
            values:
                Shape: (B, L, 2)

            band, survey, type_id:
                Shape: (B, L)

            global_feat:
                Shape: (B, global_features)

            valid_mask:
                True where measurement is valid.
                Shape: (B, L)

            mask_positions:
                True for the selected 20 valid masked measurements.
                Shape: (B, L)

        Returns:
            pred:
                Predicted flux and error.
                Shape: (B, L, 2)

            cls:
                CLS representation.
                Shape: (B, d_model)
        """

        B, L, _ = values.shape

        # Encoder:
        # masked valid measurements are removed here.
        x_encoded, mask, ids_keep, keep_padding_mask, cls = self.forward_encoder(
            values=values,
            band=band,
            survey=survey,
            type_id=type_id,
            global_feat=global_feat,
            valid_mask=valid_mask,
            mask_positions=mask_positions,
        )

        # Decoder:
        # mask tokens are inserted and original order is restored here.
        pred = self.forward_decoder(
            x=x_encoded,
            ids_keep=ids_keep,
            keep_padding_mask=keep_padding_mask,
            seq_len=L,
        )

        return pred, cls
