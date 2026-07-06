"""
Photometry Transformer model.

Minimum MAE-style edit requested by the supervisor:
    - choose fixed masked valid measurements before the model call, as before;
    - remove those masked valid measurements before the encoder;
    - encoder sees only unmasked valid measurements;
    - restore the original measurement positions after the encoder using mask tokens;
    - decoder predicts the full sequence;
    - loss is still computed only on masked positions in models/masking.py.
"""
import logging
import torch
import torch.nn as nn
from .embedding import PhotometryEmbedding

logger = logging.getLogger(__name__)


class PhotometryTransformer(nn.Module):
    """
    Transformer model for photometry time series with masked autoencoding.

    This follows the advisor's MAE example:
        original valid measurements -> remove 20 masked valid measurements -> encoder
        -> append mask tokens -> restore original order -> decoder.
    """
    def __init__(self, cfg, n_band, n_survey, n_type):
        super().__init__()
        logger.info("Initializing Transformer...")
        logger.info(f"CLS token enabled: {cfg.use_cls_token}")
        logger.info(f"Global features dimension: {cfg.global_features}")
        logger.info("Using MAE-style remove-and-restore masking")

        self.use_cls_token = cfg.use_cls_token
        self.seq_len = cfg.seq_len
        self.d_model = cfg.d_model

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

        # MINIMUM REQUIRED EDIT:
        # Mask token is now used AFTER the encoder, when restoring removed positions.
        # It is not given to the encoder.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # Decoder as MLP with 3 layers; unchanged from the previous architecture.
        self.decoder = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 2)
        )

        logger.info("Model ready.")

    def _remove_masked_measurements(self, x_meas, valid_mask, mask_positions):
        """
        Remove masked valid measurements before the encoder.

        Args:
            x_meas: (B, L, D), embedded measurement tokens, without CLS
            valid_mask: (B, L), True where a measurement is valid
            mask_positions: (B, L), True for the fixed 20 valid measurements to hide

        Returns:
            x_keep: (B, Kmax, D), padded unmasked valid tokens for the encoder
            keep_padding_mask: (B, Kmax), True for padded positions to ignore
            ids_keep: Python list; original indices kept for each object

        Example:
            If one object has 94 valid measurements and 20 are masked,
            ids_keep has length 74 and the encoder sees only those 74 tokens.
        """
        B, L, D = x_meas.shape
        device = x_meas.device

        # Keep valid measurements except the selected masked ones.
        keep_mask = valid_mask & (~mask_positions)
        ids_keep = [torch.where(keep_mask[i])[0] for i in range(B)]

        # Different objects may have different numbers of valid measurements.
        # We pad inside the batch so the Transformer can process them together.
        keep_lengths = [len(idx) for idx in ids_keep]
        max_keep = max(max(keep_lengths), 1)

        x_keep = torch.zeros(B, max_keep, D, device=device, dtype=x_meas.dtype)
        keep_padding_mask = torch.ones(B, max_keep, device=device, dtype=torch.bool)

        for i, idx in enumerate(ids_keep):
            n_keep = len(idx)
            if n_keep == 0:
                continue
            x_keep[i, :n_keep] = x_meas[i, idx]
            keep_padding_mask[i, :n_keep] = False

        return x_keep, keep_padding_mask, ids_keep

    def _restore_original_positions(self, encoded_keep, ids_keep, batch_size, seq_len):
        """
        Restore encoder output to the original L measurement positions.

        This is the photometry equivalent of the advisor's code:
            x_ = torch.cat([x, mask_tokens], dim=1)
            x_ = torch.gather(x_, 1, ids_restore...)

        Because every object can have a different valid_mask pattern, we restore with
        per-object indices instead of one shared ids_restore tensor.

        Args:
            encoded_keep: (B, Kmax, D), encoder outputs for unmasked valid tokens
            ids_keep: original kept indices for each object
            batch_size: B
            seq_len: original number of measurements L

        Returns:
            x_full: (B, L, D), full sequence for the decoder
        """
        D = encoded_keep.shape[-1]

        # Start with mask tokens everywhere.
        # Then overwrite only unmasked valid positions with encoder outputs.
        # Therefore masked positions are genuinely new to the decoder.
        x_full = self.mask_token.expand(batch_size, seq_len, D).clone()

        for i, idx in enumerate(ids_keep):
            n_keep = len(idx)
            if n_keep == 0:
                continue
            x_full[i, idx] = encoded_keep[i, :n_keep]

        return x_full

    def forward(self, values, band, survey, type_id, global_feat, valid_mask, mask_positions):
        """
        Forward pass with MAE-style remove-and-restore masking.

        Args:
            values: (B, L, 2) - [flux, error] after arcsinh
            band: (B, L) - band IDs
            survey: (B, L) - survey IDs
            type_id: (B, L) - type IDs
            global_feat: (B, global_dim) - global features
            valid_mask: (B, L) - True where measurement is valid
            mask_positions: (B, L) - True for masked valid measurements

        Returns:
            pred: (B, L, 2) - predicted [flux, error]
            cls: (B, d_model) - CLS token for downstream tasks
        """
        B, L, _ = values.shape
        device = values.device

        # Safety: never hide invalid measurements.
        mask_positions = mask_positions & valid_mask

        # 1) Embed all positions, but do NOT mask inside the embedding.
        x = self.embedding(values, band, survey, type_id, global_feat, mask_positions=None)

        # Separate CLS token from measurement tokens.
        if self.use_cls_token:
            cls_token = x[:, :1, :]
            x_meas = x[:, 1:, :]
        else:
            cls_token = None
            x_meas = x

        # 2) Remove the masked valid measurements before the encoder.
        x_keep, keep_padding_mask, ids_keep = self._remove_masked_measurements(
            x_meas=x_meas,
            valid_mask=valid_mask,
            mask_positions=mask_positions
        )

        # 3) Encoder receives only CLS + unmasked valid measurements.
        if self.use_cls_token:
            x_encoder = torch.cat([cls_token, x_keep], dim=1)
            cls_ignore = torch.zeros(B, 1, device=device, dtype=torch.bool)
            src_key_padding_mask = torch.cat([cls_ignore, keep_padding_mask], dim=1)
        else:
            x_encoder = x_keep
            src_key_padding_mask = keep_padding_mask

        x_encoded = self.encoder(x_encoder, src_key_padding_mask=src_key_padding_mask)
        x_encoded = self.norm(x_encoded)

        # Separate CLS output and encoded unmasked measurement outputs.
        if self.use_cls_token:
            cls = x_encoded[:, 0, :]
            encoded_keep = x_encoded[:, 1:, :]
        else:
            encoded_keep = x_encoded
            # Mean over non-padded kept tokens for a CLS-like global representation.
            nonpad = (~keep_padding_mask).unsqueeze(-1).to(encoded_keep.dtype)
            denom = nonpad.sum(dim=1).clamp_min(1.0)
            cls = (encoded_keep * nonpad).sum(dim=1) / denom

        # 4) Restore to original measurement positions.
        #    Removed/masked measurements receive mask_token here.
        x_full = self._restore_original_positions(
            encoded_keep=encoded_keep,
            ids_keep=ids_keep,
            batch_size=B,
            seq_len=L
        )

        # 5) Decode the full sequence.
        #    The existing reconstruction_loss selects only mask_positions.
        pred = self.decoder(x_full)

        return pred, cls
