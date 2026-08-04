"""Detailed masked-position evaluation for the MAE-style model."""

import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from models.masking import apply_mask_fixed

logger = logging.getLogger(__name__)


def compute_detailed_metrics(true_values, predicted_values, prefix=""):
    true_flat = true_values.ravel()
    predicted_flat = predicted_values.ravel()
    residuals = predicted_flat - true_flat
    total_variance = np.sum((true_flat - np.mean(true_flat)) ** 2)
    return {
        f"{prefix}rmse": float(np.sqrt(np.mean(residuals ** 2))),
        f"{prefix}mae": float(np.mean(np.abs(residuals))),
        f"{prefix}bias": float(np.mean(residuals)),
        f"{prefix}std": float(np.std(residuals)),
        f"{prefix}median_abs": float(np.median(np.abs(residuals))),
        f"{prefix}p90_abs": float(np.percentile(np.abs(residuals), 90)),
        f"{prefix}p95_abs": float(np.percentile(np.abs(residuals), 95)),
        f"{prefix}p99_abs": float(np.percentile(np.abs(residuals), 99)),
        f"{prefix}r2": float(
            1 - np.sum(residuals ** 2) / (total_variance + 1e-12)
        ),
    }


def evaluate(model, dataloader, cfg):
    model.eval()
    true_rows = []
    predicted_rows = []
    band_rows = []
    cls_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="evaluation"):
            batch = {key: value.to(cfg.device) for key, value in batch.items()}
            mask_positions, ids_keep, ids_shuffle = apply_mask_fixed(
                batch["valid_mask"], cfg.num_mask
            )
            prediction, cls = model(
                batch["values"],
                batch["band_ids"],
                batch["survey_ids"],
                batch["type_ids"],
                batch["global_features"],
                batch["valid_mask"],
                mask_positions,
                ids_keep,
                ids_shuffle,
            )
            for row in range(len(mask_positions)):
                selected = mask_positions[row]
                if selected.any():
                    true_rows.append(batch["values"][row, selected].cpu().numpy())
                    predicted_rows.append(prediction[row, selected].cpu().numpy())
                    band_rows.append(batch["band_ids"][row, selected].cpu().numpy())
                    cls_embeddings.append(cls[row].cpu().numpy())

    if not true_rows:
        raise ValueError("No valid masked positions were available for evaluation.")

    true_values = np.concatenate(true_rows)
    predicted_values = np.concatenate(predicted_rows)
    band_ids = np.concatenate(band_rows)
    cls_embeddings = np.asarray(cls_embeddings)
    metrics = {
        **compute_detailed_metrics(
            true_values[:, 0], predicted_values[:, 0], "flux_"
        ),
        **compute_detailed_metrics(
            true_values[:, 1], predicted_values[:, 1], "error_"
        ),
    }
    pd.DataFrame([metrics]).to_csv(
        cfg.evaluation_dir / "detailed_metrics.csv", index=False
    )
    pd.DataFrame({
        "true_flux": true_values[:, 0],
        "predicted_flux": predicted_values[:, 0],
        "flux_residual": predicted_values[:, 0] - true_values[:, 0],
        "true_error": true_values[:, 1],
        "predicted_error": predicted_values[:, 1],
        "error_residual": predicted_values[:, 1] - true_values[:, 1],
        "band_id": band_ids,
    }).to_csv(cfg.evaluation_dir / "predictions.csv", index=False)
    logger.info("Evaluated %d masked measurements", len(true_values))
    return {
        "true": true_values,
        "predicted": predicted_values,
        "bands": band_ids,
        "embeddings": cls_embeddings,
        "metrics": metrics,
    }

