"""
Transformer model for photometry time series - MAE-style with FIXED masking
Sequence: Embedding → Encoder → Mask → Decoder → Prediction
"""
import logging
import torch
import torch.nn as nn
from .embedding import PhotometryEmbedding
from .masking import apply_mask_fixed

logger = logging.getLogger(__name__)


class PhotometryTransformer(nn.Module):
    def __init__(self, cfg, n_band, n_survey, n_type):
        super().__init__()
        logger.info("Initializing Transformer with MAE-style masking...")
        logger.info(f"Fixed masking: {cfg.num_mask} masks per sample")  # ✅ لاگ جدید
        
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.d_model = cfg.d_model
        self.use_cls_token = cfg.use_cls_token
        self.num_mask = cfg.num_mask  # ✅ 20
        
        # 1. Embedding layer (NO masking inside)
        self.embedding = PhotometryEmbedding(
            d_model=cfg.d_model, 
            n_band=n_band, 
            n_survey=n_survey, 
            n_type=n_type,
            max_len=cfg.seq_len, 
            use_global=True, 
            global_dim=cfg.global_features,
            use_cls_token=False
        )
        
        # 2. Encoder (processes ALL tokens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, 
            nhead=cfg.n_heads, 
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, cfg.n_layers)
        self.encoder_norm = nn.LayerNorm(cfg.d_model)
        
        # 3. Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, cfg.n_layers)
        self.decoder_norm = nn.LayerNorm(cfg.d_model)
        
        # 4. Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # 5. CLS token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            nn.init.normal_(self.cls_token, std=0.02)
            self.global_proj = nn.Sequential(
                nn.Linear(cfg.global_features, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, cfg.d_model)
            )
        
        # 6. Prediction head (3-layer MLP)
        self.pred_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 2)
        )
        
        logger.info("Model ready.")

    def forward(self, values, band, survey, type_id, global_feat, valid_mask):
        B, L, _ = values.shape
        device = values.device
        
        # ========== STEP 1: EMBEDDING ==========
        x = self.embedding(values, band, survey, type_id, global_feat)
        
        # ========== STEP 2: ENCODER (ALL TOKENS - NO MASKING) ==========
        src_key_padding_mask = ~valid_mask
        x_encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x_encoded = self.encoder_norm(x_encoded)
        
        # ========== STEP 3: FIXED MASKING (AFTER ENCODER) ==========
        # ✅ استفاده از apply_mask_fixed به جای random_masking_embeddings
        mask_positions = apply_mask_fixed(valid_mask, num_mask=self.num_mask)
        
        # Get indices of kept and removed tokens
        ids_keep_list = []
        ids_shuffle_list = []
        for i in range(B):
            all_ids = torch.arange(L, device=device)
            mask_bool = mask_positions[i]
            
            ids_keep = all_ids[~mask_bool]
            ids_remove = all_ids[mask_bool]
            ids_shuffle = torch.cat([ids_keep, ids_remove], dim=0)
            
            ids_keep_list.append(ids_keep)
            ids_shuffle_list.append(ids_shuffle)
        
        # Create masked version for decoder memory
        max_keep = max([len(ids) for ids in ids_keep_list])
        x_masked = torch.zeros(B, max_keep, self.d_model, device=device)
        for i in range(B):
            if len(ids_keep_list[i]) > 0:
                x_masked[i, :len(ids_keep_list[i])] = x_encoded[i, ids_keep_list[i]]
        
        # ========== STEP 4: RESTORE + MASK TOKENS ==========
        mask_tokens = self.mask_token.repeat(B, L, 1)
        x_restored = mask_tokens.clone()
        for i in range(B):
            if len(ids_keep_list[i]) > 0:
                x_restored[i, ids_keep_list[i]] = x_encoded[i, ids_keep_list[i]]
        
        # Restore original ordering
        ids_restore_list = [torch.argsort(ids_shuffle, dim=0) for ids_shuffle in ids_shuffle_list]
        for i in range(B):
            ids_restore = ids_restore_list[i]
            x_restored[i] = x_restored[i, ids_restore]
        
        # ========== STEP 5: DECODER ==========
        tgt_key_padding_mask = ~valid_mask
        
        if self.use_cls_token:
            cls = self.cls_token.expand(B, 1, -1)
            g = self.global_proj(global_feat)
            cls = cls + g.unsqueeze(1)
            x_decoder_input = torch.cat([cls, x_restored], dim=1)
            tgt_key_padding_mask = torch.cat([
                torch.zeros(B, 1, device=device, dtype=torch.bool),
                tgt_key_padding_mask
            ], dim=1)
        else:
            x_decoder_input = x_restored
        
        x_decoded = self.decoder(
            tgt=x_decoder_input,
            memory=x_masked,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None
        )
        x_decoded = self.decoder_norm(x_decoded)
        
        # ========== STEP 6: PREDICTION ==========
        if self.use_cls_token:
            x_for_pred = x_decoded[:, 1:, :]
            cls_token = x_decoded[:, 0, :]
        else:
            x_for_pred = x_decoded
            cls_token = x_decoded.mean(dim=1)
        
        pred = self.pred_head(x_for_pred)
        mask = mask_positions.float()
        
        return pred, cls_token, mask
