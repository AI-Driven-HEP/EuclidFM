"""
Evaluation metrics (Baseline Version v1)
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score


def compute_metrics(y_true, y_pred):
    """
    Compute comprehensive metrics for reconstruction.
    """
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'Explained Variance': explained_variance_score(y_true, y_pred)
    }


def compute_correlation(y_true, y_pred):
    """Compute Pearson correlation coefficient."""
    return np.corrcoef(y_true, y_pred)[0, 1]