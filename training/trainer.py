"""
Training functions (Baseline Version v1)
"""

import logging
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def apply_mask_fixed(valid_mask, num_mask=20):
    """
    Generate exactly `num_mask` mask positions per sample (only from valid positions).
    """
    B, L = valid_mask.shape
    device = valid_mask.device
    mask_positions = torch.zeros(B, L, dtype=torch.bool, device=device)

    for i in range(B):
        valid_indices = torch.where(valid_mask[i])[0]
        n_valid = len(valid_indices)

        if n_valid == 0:
            continue

        k = min(num_mask, n_valid)

        if k > 0:
            perm = torch.randperm(n_valid, device=device)[:k]
            selected = valid_indices[perm]
            mask_positions[i, selected] = True

    return mask_positions


def train_epoch(model, dataloader, optimizer, cfg, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        values = batch["values"].to(device)
        band = batch["band_ids"].to(device)
        survey = batch["survey_ids"].to(device)
        type_id = batch["type_ids"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        global_feat = batch["global_features"].to(device)

        mask_positions = apply_mask_fixed(valid_mask, num_mask=20)
        pred, _ = model(values, band, survey, type_id, global_feat, valid_mask, mask_positions)
        loss = model.reconstruction_loss(pred, values, mask_positions, valid_mask)

        if torch.isnan(loss):
            logger.warning("NaN loss detected, skipping batch")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate_epoch(model, dataloader, cfg, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val", leave=False):
            values = batch["values"].to(device)
            band = batch["band_ids"].to(device)
            survey = batch["survey_ids"].to(device)
            type_id = batch["type_ids"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            global_feat = batch["global_features"].to(device)

            mask_positions = apply_mask_fixed(valid_mask, num_mask=20)
            pred, _ = model(values, band, survey, type_id, global_feat, valid_mask, mask_positions)
            loss = model.reconstruction_loss(pred, values, mask_positions, valid_mask)

            if not torch.isnan(loss):
                total_loss += loss.item()
                num_batches += 1

    return total_loss / max(num_batches, 1)
