"""Detailed masked-position evaluation for the MAE-style model.

UPDATED:
  - Tracks which ORIGINAL object each masked sample belongs to.
  - Collects one CLS embedding PER OBJECT (one vector per batch row).
  - Returns per-sample and per-object metadata needed by visualize.py.
"""

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
    """
    Evaluate the model on a dataloader and return all arrays needed by
    the visualization stage.

    Returns
    -------
    dict with keys:
        true          : (N_samples, 2)  — true [flux, error] per masked sample
        predicted     : (N_samples, 2)  — predicted [flux, error]
        bands         : (N_samples,)    — band id per masked sample
        embeddings    : (N_obj, D)      — one CLS vector per OBJECT
        object_ids    : (N_obj,)        — original object id per embedding row
        sample_object : (N_samples,)    — index of the owning object for each sample
        metrics       : dict            — global metrics
    """
    model.eval()

    true_rows = []
    predicted_rows = []
    band_rows = []
    sample_object_indices = []   # for each masked sample, which OBJECT (by position)
    object_embeddings = []       # CLS vector per object, one row per dataset row
    object_ids_collected = []    # original object id per row of the dataset

    running_offset = 0            # cumulative row counter across batches

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

            B = len(mask_positions)

            # ---- One CLS embedding per row (= per object) ----
            object_embeddings.append(cls.cpu().numpy())

            # ---- Original object id per row (if dataset provides it) ----
            if "object_id" in batch:
                obj_ids = batch["object_id"].cpu().numpy()
            else:
                obj_ids = np.arange(
                    running_offset, running_offset + B, dtype=np.int64
                )
            object_ids_collected.append(obj_ids)

            # ---- Per-sample data (only masked positions) ----
            for row in range(B):
                selected = mask_positions[row]
                if selected.any():
                    true_rows.append(batch["values"][row, selected].cpu().numpy())
                    predicted_rows.append(prediction[row, selected].cpu().numpy())
                    band_rows.append(batch["band_ids"][row, selected].cpu().numpy())

                    # All masked samples from this row belong to the SAME object
                    global_object_index = running_offset + row
                    n_selected = int(selected.sum())
                    sample_object_indices.extend([global_object_index] * n_selected)

            running_offset += B

    if not true_rows:
        raise ValueError("No valid masked positions were available for evaluation.")

    # ---- Stack everything ----
    true_values = np.concatenate(true_rows)                        # (N_samples, 2)
    predicted_values = np.concatenate(predicted_rows)              # (N_samples, 2)
    band_ids = np.concatenate(band_rows)                           # (N_samples,)
    embeddings = np.concatenate(object_embeddings, axis=0)         # (N_obj, D)
    object_ids_arr = np.concatenate(object_ids_collected, axis=0)  # (N_obj,)
    sample_object_indices = np.array(sample_object_indices, dtype=np.int64)

    # ---- Global metrics ----
    metrics = {
        **compute_detailed_metrics(
            true_values[:, 0], predicted_values[:, 0], "flux_"
        ),
        **compute_detailed_metrics(
            true_values[:, 1], predicted_values[:, 1], "error_"
        ),
    }

    # ---- Save CSVs ----
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
        "object_index": sample_object_indices,   # NEW — per-sample → object map
    }).to_csv(cfg.evaluation_dir / "predictions.csv", index=False)

    logger.info(
        "Evaluated %d masked measurements from %d objects",
        len(true_values), len(embeddings),
    )

    return {
        "true": true_values,
        "predicted": predicted_values,
        "bands": band_ids,
        "embeddings": embeddings,
        "object_ids": object_ids_arr,
        "sample_object": sample_object_indices,
        "metrics": metrics,
    }
