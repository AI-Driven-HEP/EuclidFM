"""
Evaluation script
"""
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from models.masking import apply_mask_fixed
import logging

logger = logging.getLogger(__name__)

def evaluate_model(model, dataloader, device, num_mask):
    """
    Evaluate model on validation set.
    
    Returns:
        all_real: (N, 2) - ground truth [flux, error]
        all_pred: (N, 2) - predictions [flux, error]
        metrics: dict with evaluation metrics
    """
    all_real = []
    all_pred = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            values = batch["values"].to(device)
            band = batch["band_ids"].to(device)
            survey = batch["survey_ids"].to(device)
            type_id = batch["type_ids"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            global_feat = batch["global_features"].to(device)

            # Use fixed masking for evaluation
            mask_positions = apply_mask_fixed(valid_mask, num_mask=num_mask)
            pred, _, _ = model(values, band, survey, type_id, global_feat, valid_mask)
            
            # Only evaluate on masked positions
            all_real.append(values[mask_positions].cpu().numpy())
            all_pred.append(pred[mask_positions].cpu().numpy())

    all_real = np.concatenate(all_real, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    metrics = {
        'Flux': {
            'RMSE': np.sqrt(mean_squared_error(all_real[:, 0], all_pred[:, 0])),
            'MAE': mean_absolute_error(all_real[:, 0], all_pred[:, 0]),
            'R2': r2_score(all_real[:, 0], all_pred[:, 0]),
            'Explained Variance': explained_variance_score(all_real[:, 0], all_pred[:, 0])
        },
        'Error': {
            'RMSE': np.sqrt(mean_squared_error(all_real[:, 1], all_pred[:, 1])),
            'MAE': mean_absolute_error(all_real[:, 1], all_pred[:, 1]),
            'R2': r2_score(all_real[:, 1], all_pred[:, 1]),
            'Explained Variance': explained_variance_score(all_real[:, 1], all_pred[:, 1])
        }
    }

    return all_real, all_pred, metrics


def compute_statistics(all_real, all_pred):
    """Compute detailed statistics for reconstruction."""
    stats = {
        'Flux': {
            'Real Range': [all_real[:, 0].min(), all_real[:, 0].max()],
            'Pred Range': [all_pred[:, 0].min(), all_pred[:, 0].max()],
            'Mean Real': all_real[:, 0].mean(),
            'Mean Pred': all_pred[:, 0].mean(),
            'Std Real': all_real[:, 0].std(),
            'Std Pred': all_pred[:, 0].std(),
            'Correlation': np.corrcoef(all_real[:, 0], all_pred[:, 0])[0, 1]
        },
        'Error': {
            'Real Range': [all_real[:, 1].min(), all_real[:, 1].max()],
            'Pred Range': [all_pred[:, 1].min(), all_pred[:, 1].max()],
            'Mean Real': all_real[:, 1].mean(),
            'Mean Pred': all_pred[:, 1].mean(),
            'Std Real': all_real[:, 1].std(),
            'Std Pred': all_pred[:, 1].std(),
            'Correlation': np.corrcoef(all_real[:, 1], all_pred[:, 1])[0, 1]
        }
    }
    return stats