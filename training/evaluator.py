"""
Evaluation functions (Baseline Version v1)
"""

import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)


def evaluate_model(model, dataloader, device, num_mask=20):
    """Evaluate model and collect predictions."""
    model.eval()
    all_real = []
    all_pred = []

    with torch.no_grad():
        for batch in dataloader:
            values = batch["values"].to(device)
            band = batch["band_ids"].to(device)
            survey = batch["survey_ids"].to(device)
            type_id = batch["type_ids"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            global_feat = batch["global_features"].to(device)

            mask_positions = model.apply_mask_fixed(valid_mask, num_mask=num_mask)
            pred, _ = model(values, band, survey, type_id, global_feat, valid_mask, mask_positions)

            all_real.append(values[mask_positions].cpu().numpy())
            all_pred.append(pred[mask_positions].cpu().numpy())

    return np.concatenate(all_real, axis=0), np.concatenate(all_pred, axis=0)