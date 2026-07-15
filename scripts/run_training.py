"""
Main training script (Baseline Version v1)
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from data import (
    PhotometryDataset,
    find_measurements,
    build_measurement_dataset_arcsinh,
    extract_global_features,
    fit_global_scaler,
    transform_global
)
from models import PhotometryTransformer
from utils.helpers import set_global_seed
from training.trainer import train_epoch, validate_epoch

logger = logging.getLogger(__name__)

def main():
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    set_global_seed(cfg['data']['seed'])

    # Load and preprocess data
    df_raw = pd.read_csv(cfg['data']['input_csv'])

    # Flag-based filtering
    det_col, spur_col = "MER__DET_QUALITY_FLAG", "MER__SPURIOUS_FLAG"
    df_raw[det_col] = pd.to_numeric(df_raw[det_col], errors='coerce')
    df_raw[spur_col] = pd.to_numeric(df_raw[spur_col], errors='coerce')
    mask_det = df_raw[det_col].isin([0, 2, 512])
    mask_spur = df_raw[spur_col] == 0
    df_flagged = df_raw[mask_det & mask_spur].copy()

    # Remove specific IDs
    remove_ids = [-516833347281158785, -524325240264858394, -526763971264021159, -531358091287592116,
                  -601550136483347596, -605798685503436276, -621407362500663238, -632954073493539985,
                  -633889436492517261, -635092693485500394, -643401708482019153, 2664991879655406635,
                  2675618383640896787, 2676463652656186488, 2676932786640872754, 2679138096642491394,
                  2690071563641296650]
    df_clean = df_flagged[~df_flagged["OBJECT_ID"].isin(remove_ids)]
    df_clean = df_clean.dropna(axis=1, how="all")

    # Sample and split
    n_samples = cfg['data']['n_samples']
    if len(df_clean) > n_samples:
        df_sampled = df_clean.sample(n=n_samples, random_state=cfg['data']['seed'])
    else:
        df_sampled = df_clean.copy()

    train_df, val_df = train_test_split(df_sampled, test_size=cfg['data']['test_size'],
                                        random_state=cfg['data']['seed'])

    # Build datasets
    measurements = find_measurements(df_clean)
    (train_values, train_mask, train_band, train_survey, train_type,
     band_vocab, survey_vocab, type_vocab) = build_measurement_dataset_arcsinh(train_df, measurements)
    (val_values, val_mask, val_band, val_survey, val_type,
     _, _, _) = build_measurement_dataset_arcsinh(val_df, measurements)

    # Global features
    train_global = extract_global_features(train_df)
    val_global = extract_global_features(val_df)
    global_scaler = fit_global_scaler(train_global)
    train_global_norm = transform_global(train_global, global_scaler)
    val_global_norm = transform_global(val_global, global_scaler)

    # Create datasets and dataloaders
    train_dataset = PhotometryDataset(train_values, train_mask, train_band, train_survey, train_type, train_global_norm)
    val_dataset = PhotometryDataset(val_values, val_mask, val_band, val_survey, val_type, val_global_norm)

    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'],
                              shuffle=True, num_workers=cfg['training']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'],
                            shuffle=False, num_workers=cfg['training']['num_workers'])

    # Initialize model
    device = torch.device(cfg['training']['device'])
    n_band = len(band_vocab)
    n_survey = len(survey_vocab)
    n_type = len(type_vocab)

    # Create config object
    class Config:
        pass
    config = Config()
    config.seq_len = len(measurements)
    config.d_model = cfg['model']['d_model']
    config.n_heads = cfg['model']['n_heads']
    config.n_layers = cfg['model']['n_layers']
    config.ff_dim = cfg['model']['ff_dim']
    config.dropout = cfg['model']['dropout']
    config.use_cls_token = cfg['model']['use_cls_token']
    config.global_features = len(cfg['global_features'])

    model = PhotometryTransformer(config, n_band, n_survey, n_type).to(device)

    # Attach utils to model
    def recon_loss(pred, target, mask_positions, valid_mask):
        if mask_positions.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        diff = (pred - target).pow(2).mean(dim=-1)
        return diff[mask_positions].mean()
    model.reconstruction_loss = recon_loss
    model.apply_mask_fixed = apply_mask_fixed

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'],
                                  weight_decay=cfg['training']['weight_decay'])

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(cfg['training']['epochs']):
        logger.info(f"========== EPOCH {epoch+1}/{cfg['training']['epochs']} ==========")

        train_loss = train_epoch(model, train_loader, optimizer, cfg, device)
        val_loss = validate_epoch(model, val_loader, cfg, device)

        logger.info(f"Train Loss: {train_loss:.6f}")
        logger.info(f"Validation Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "photometry_transformer_best.pt")
            logger.info(f"Best model saved with val_loss: {best_val_loss:.6f}")

    torch.save(model.state_dict(), "photometry_transformer_pretrained.pt")
    logger.info(f"Final model saved. Best validation loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()