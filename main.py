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
from models.transformer import PhotometryTransformer
from training.trainer import Trainer
from evaluate import evaluate_model
from visualize import plot_training_curves, plot_reconstruction_scatter, plot_tsne
from utils.helpers import set_global_seed, setup_logging

logger = setup_logging()
set_global_seed(cfg.seed)

def main():
    # ========== 1. LOAD AND CLEAN DATA ==========
    logger.info("Loading and cleaning data...")
    df_clean = filter_and_clean_data(cfg.input_csv, cfg.cleaned_csv)
    
    # ========== 2. FIND MEASUREMENTS ==========
    measurements = find_measurements(df_clean)
    cfg.seq_len = len(measurements)
    logger.info(f"Found {cfg.seq_len} measurements")
    
    # ========== 3. SAMPLE AND SPLIT ==========
    logger.info("Sampling and splitting data...")
    n_samples = 10000
    if len(df_clean) > n_samples:
        df_sampled = df_clean.sample(n=n_samples, random_state=cfg.seed)
    else:
        df_sampled = df_clean.copy()
    
    train_df, val_df = train_test_split(df_sampled, test_size=0.2, random_state=cfg.seed)
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # ========== 4. BUILD DATASETS ==========
    logger.info("Building training dataset...")
    (train_values, train_mask, train_band, train_survey, train_type, 
     band_vocab, survey_vocab, type_vocab) = build_measurement_dataset_arcsinh(train_df, measurements)
    
    logger.info("Building validation dataset...")
    (val_values, val_mask, val_band, val_survey, val_type, 
     _, _, _) = build_measurement_dataset_arcsinh(val_df, measurements)
    
    # Save NPZ
    np.savez(cfg.output_npz,
             train_values=train_values, train_mask=train_mask,
             train_band=train_band, train_survey=train_survey, train_type=train_type,
             val_values=val_values, val_mask=val_mask,
             val_band=val_band, val_survey=val_survey, val_type=val_type,
             band_vocab=np.array(list(band_vocab.items()), dtype=object),
             survey_vocab=np.array(list(survey_vocab.items()), dtype=object),
             type_vocab=np.array(list(type_vocab.items()), dtype=object),
             measurements=np.array(measurements, dtype=object))
    logger.info(f"Dataset saved to {cfg.output_npz}")
    
    # ========== 5. GLOBAL FEATURES ==========
    logger.info("Extracting global features...")
    train_global = extract_global_features(train_df)
    val_global = extract_global_features(val_df)
    
    global_scaler = fit_global_scaler(train_global)
    train_global_norm = transform_global(train_global, global_scaler)
    val_global_norm = transform_global(val_global, global_scaler)
    
    # ========== 6. DATALOADERS ==========
    train_dataset = PhotometryDataset(train_values, train_mask, train_band, train_survey, train_type, train_global_norm)
    val_dataset = PhotometryDataset(val_values, val_mask, val_band, val_survey, val_type, val_global_norm)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    
    # ========== 7. MODEL ==========
    device = torch.device(cfg.device)
    model = PhotometryTransformer(cfg, len(band_vocab), len(survey_vocab), len(type_vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Device: {device}")
    
    # ========== 8. TRAINING ==========
    trainer = Trainer(model, optimizer, device, cfg.best_model_path)
    train_losses, val_losses = trainer.train(train_loader, val_loader, cfg.epochs)
    
    # Save final model
    torch.save(model.state_dict(), cfg.model_save_path)
    logger.info(f"Final model saved to {cfg.model_save_path}")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, save_path="training_curves.pdf")
    
    # ========== 9. EVALUATION ==========
    logger.info("Evaluating model...")
    all_real, all_pred, metrics = evaluate_model(model, val_loader, device, cfg.num_mask)
    
    logger.info("="*50)
    logger.info("EVALUATION METRICS:")
    logger.info(f"Flux RMSE: {metrics['Flux']['RMSE']:.4f}")
    logger.info(f"Error RMSE: {metrics['Error']['RMSE']:.4f}")
    logger.info(f"Flux R²: {metrics['Flux']['R2']:.4f}")
    logger.info(f"Error R²: {metrics['Error']['R2']:.4f}")
    logger.info("="*50)
    
    plot_reconstruction_scatter(all_real, all_pred, 
                                metrics['Flux']['RMSE'], 
                                metrics['Error']['RMSE'],
                                save_path="reconstruction_scatter.pdf")
    
    # ========== 10. T-SNE VISUALIZATION ==========
    logger.info("Generating t-SNE visualization...")
    plot_tsne(model, val_loader, device, save_path="tsne_fixed.pdf")

if __name__ == "__main__":
    main()