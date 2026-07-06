"""
Photometry Transformer model.

MAE-style masking implementation requested by the supervisor:
    1. Embed all measurements.
    2. Randomly chosen valid measurements are REMOVED before the encoder.
    3. The encoder only sees unmasked valid measurements.
    4. After the encoder, mask tokens are inserted back into the original positions.
    5. The decoder predicts the full sequence; the loss is computed only on masked positions.
"""
import logging
import torch
import torch.nn as nn
from .embedding import PhotometryEmbedding

logger = logging.getLogger(__name__)


class PhotometryTransformer(nn.Module):
    """
    Transformer model for photometry time series with masked autoencoding.

    The key difference from the previous version is that masked measurements are not
    replaced before the encoder. They are completely removed, exactly like MAE patch
    masking. This means that if an object has 94 valid measurements and 20 are masked,
    the encoder receives only 74 measurement tokens.
    """

    def __init__(self, cfg, n_band, n_survey, n_type):
        super().__init__()
        logger.info("Initializing Transformer with MAE-style remove-and-restore masking...")
        logger.info(f"CLS token enabled: {cfg.use_cls_token}")
        logger.info(f"Global features dimension: {cfg.global_features}")
        logger.info("Masked valid measurements are removed before the encoder.")

        self.seq_len = cfg.seq_len
        self.d_model = cfg.d_model
        self.use_cls_token = cfg.use_cls_token

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

        # Decoder-side mask token.
        # This is inserted after the encoder to represent positions that were removed.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # Decoder-side positional embeddings.
        # Important: after restoring the original order, masked positions need their own
        # original position information, just like decoder_pos_embed in MAE.
        self.decoder_pos_embed = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.register_buffer("decoder_pos_ids", torch.arange(cfg.seq_len), persistent=False)

        # MLP decoder. It predicts [flux, error] for every original measurement position.
        self.decoder = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 2)
        )

        logger.info("Model ready.")

    def _pack_unmasked_tokens(self, x_meas, valid_mask, mask_positions):
        """
        Remove masked and invalid measurements before the encoder.

        Args:
            x_meas: (B, L, D), embedded measurement tokens without CLS
            valid_mask: (B, L), True for physically valid measurements
            mask_positions: (B, L), True for valid measurements selected for masking

        Returns:
            x_keep: (B, Kmax, D), padded batch of unmasked valid tokens
            keep_padding_mask: (B, Kmax), True where padding should be ignored
            ids_keep: list of length B, each item has original indices kept for that object

        Example:
            If L=94 and 20 valid positions are masked, then that object contributes
            74 tokens to the encoder. Because different objects may have different
            numbers of valid measurements, we pad to Kmax inside the batch.
        """
        B, L, D = x_meas.shape
        device = x_meas.device

        # Keep only valid measurements that were NOT selected for reconstruction.
        keep_mask = valid_mask & ~mask_positions
        ids_keep = [torch.where(keep_mask[i])[0] for i in range(B)]

        # Transformer batches need rectangular tensors, so we pad to the largest
        # number of kept tokens in the current batch.
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

    def _restore_full_sequence(self, encoded_keep, ids_keep, batch_size, seq_len):
        """
        Restore encoder outputs to the original measurement order.

        This mirrors the MAE idea:
            - encoded unmasked tokens go back to their original indices;
            - removed/masked positions receive learned mask tokens;
            - invalid positions also receive mask tokens, but they do not affect the loss.

        Args:
            encoded_keep: (B, Kmax, D), encoder output for kept tokens
            ids_keep: original measurement indices kept for each object
            batch_size: B
            seq_len: original L

        Returns:
            x_full: (B, L, D), restored sequence ready for decoder
        """
        device = encoded_keep.device
        D = encoded_keep.shape[-1]

        # Start with mask tokens everywhere.
        # Then overwrite only the unmasked valid positions with encoder outputs.
        x_full = self.mask_token.expand(batch_size, seq_len, D).clone()

        for i, idx in enumerate(ids_keep):
            n_keep = len(idx)
            if n_keep == 0:
                continue
            x_full[i, idx] = encoded_keep[i, :n_keep]

        # Add decoder positional information after restoring the original order.
        # This tells the decoder which original measurement position each token belongs to.
        pos = self.decoder_pos_embed(self.decoder_pos_ids[:seq_len].unsqueeze(0))
        x_full = x_full + pos

        return x_full

    def forward(self, values, band, survey, type_id, global_feat, valid_mask, mask_positions=None):
        """
        Forward pass using fixed MAE-style measurement removal.

        Args:
            values: (B, L, 2), transformed [flux, error]
            band: (B, L), band IDs
            survey: (B, L), survey IDs
            type_id: (B, L), type IDs
            global_feat: (B, global_dim), global object features
            valid_mask: (B, L), True where measurement is valid
            mask_positions: (B, L), True for valid measurements removed before encoder

        Returns:
            pred: (B, L, 2), reconstructed [flux, error] for all original positions
            cls: (B, d_model), global object representation
        """
        B, L, _ = values.shape
        device = values.device

        if mask_positions is None:
            # Used for feature extraction / t-SNE: remove nothing, keep all valid tokens.
            mask_positions = torch.zeros(B, L, device=device, dtype=torch.bool)
        else:
            # Safety: never mask invalid measurements.
            mask_positions = mask_positions & valid_mask

        # 1. Embed all original measurement positions.
        #    No mask token is applied in the embedding layer.
        x = self.embedding(values, band, survey, type_id, global_feat)

        # Split optional CLS token from measurement tokens.
        if self.use_cls_token:
            cls_token = x[:, :1, :]
            x_meas = x[:, 1:, :]
        else:
            cls_token = None
            x_meas = x

        # 2. Remove masked valid measurements before the encoder.
        #    The encoder sees only unmasked valid tokens.
        x_keep, keep_padding_mask, ids_keep = self._pack_unmasked_tokens(
            x_meas=x_meas,
            valid_mask=valid_mask,
            mask_positions=mask_positions
        )

        # Add CLS token back only for the encoder, if enabled.
        if self.use_cls_token:
            x_encoder = torch.cat([cls_token, x_keep], dim=1)
            cls_padding = torch.zeros(B, 1, device=device, dtype=torch.bool)
            src_key_padding_mask = torch.cat([cls_padding, keep_padding_mask], dim=1)
        else:
            x_encoder = x_keep
            src_key_padding_mask = keep_padding_mask

        # 3. Encoder processes only CLS + unmasked valid measurement tokens.
        x_encoded = self.encoder(x_encoder, src_key_padding_mask=src_key_padding_mask)
        x_encoded = self.norm(x_encoded)

        # Separate CLS and encoded measurement tokens.
        if self.use_cls_token:
            cls = x_encoded[:, 0, :]
            encoded_keep = x_encoded[:, 1:, :]
        else:
            encoded_keep = x_encoded
            # Mean over non-padding kept tokens for a global representation.
            nonpad = (~keep_padding_mask).unsqueeze(-1).to(encoded_keep.dtype)
            denom = nonpad.sum(dim=1).clamp_min(1.0)
            cls = (encoded_keep * nonpad).sum(dim=1) / denom

        # 4. Restore the full original length.
        #    Masked/removed positions are represented by mask_token, not their true values.
        x_full = self._restore_full_sequence(
            encoded_keep=encoded_keep,
            ids_keep=ids_keep,
            batch_size=B,
            seq_len=L
        )

        # 5. Decode all positions. The training loss will select only mask_positions.
        pred = self.decoder(x_full)

        return pred, cls
