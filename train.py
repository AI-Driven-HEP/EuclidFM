"""
Training entry point - standalone training script
"""
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config import cfg
from data.dataset import (
    find_measurements, build_measurement_dataset_arcsinh,
    extract_global_features, fit_global_scaler, transform_global,
    PhotometryDataset, filter_and_clean_data
)
from models.transformer import PhotometryTransformer #?? must be v2 ?
from training.trainer import Trainer #?? must be v2 or not ?
from utils.helpers import set_global_seed, setup_logging

logger = setup_logging()
set_global_seed(cfg.seed)

def train():
    # Load cleaned data
    df_clean = pd.read_csv(cfg.cleaned_csv)
    measurements = find_measurements(df_clean)
    cfg.seq_len = len(measurements)
    
    # Sample and split
    df_sampled = df_clean.sample(n=10000, random_state=cfg.seed)
    train_df, val_df = train_test_split(df_sampled, test_size=0.2, random_state=cfg.seed) #??? check the inter-object dont shuffle 
    
    # Build datasets
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
    
    # Dataloaders
    train_dataset = PhotometryDataset(train_values, train_mask, train_band, train_survey, train_type, train_global_norm)
    val_dataset = PhotometryDataset(val_values, val_mask, val_band, val_survey, val_type, val_global_norm)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    
    # Model
    device = torch.device(cfg.device)
    model = PhotometryTransformer(cfg, len(band_vocab), len(survey_vocab), len(type_vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # Train
    trainer = Trainer(model, optimizer, device, cfg.best_model_path)
    train_losses, val_losses = trainer.train(train_loader, val_loader, cfg.epochs)
    
    # Save final model
    torch.save(model.state_dict(), cfg.model_save_path)
    logger.info(f"Final model saved to {cfg.model_save_path}")
    
    return train_losses, val_losses

if __name__ == "__main__":
    train()
