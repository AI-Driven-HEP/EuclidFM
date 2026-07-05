"""
Visualization utilities
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")

def plot_training_curves(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.title('Training and Validation Loss Curves', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_reconstruction_scatter(all_real, all_pred, rmse_flux, rmse_error, save_path=None):
    """Plot scatter plots for flux and error reconstruction."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(all_real[:, 0], all_pred[:, 0], alpha=0.3, s=2, c='steelblue', edgecolors='none')
    axes[0].plot([-4, 4], [-4, 4], 'r--', linewidth=2, label='Perfect Reconstruction')
    axes[0].set_xlabel('True Flux (arcsinh)', fontsize=12)
    axes[0].set_ylabel('Predicted Flux (arcsinh)', fontsize=12)
    axes[0].set_title(f'Flux: RMSE = {rmse_flux:.4f}', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(all_real[:, 1], all_pred[:, 1], alpha=0.3, s=2, c='coral', edgecolors='none')
    axes[1].plot([-4, 4], [-4, 4], 'r--', linewidth=2, label='Perfect Reconstruction')
    axes[1].set_xlabel('True Error (arcsinh)', fontsize=12)
    axes[1].set_ylabel('Predicted Error (arcsinh)', fontsize=12)
    axes[1].set_title(f'Error: RMSE = {rmse_error:.4f}', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_residual_histograms(all_real, all_pred, save_path=None):
    """Plot residual histograms for flux and error."""
    residual_flux = all_real[:, 0] - all_pred[:, 0]
    residual_error = all_real[:, 1] - all_pred[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(residual_flux, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Residual (True - Predicted)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Flux Residuals (μ={residual_flux.mean():.4f}, σ={residual_flux.std():.4f})', fontsize=12)

    axes[1].hist(residual_error, bins=100, alpha=0.7, color='coral', edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual (True - Predicted)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'Error Residuals (μ={residual_error.mean():.4f}, σ={residual_error.std():.4f})', fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_tsne(model, dataloader, device, save_path=None):
    """Plot t-SNE visualization of CLS embeddings."""
    cls_tokens = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            values = batch["values"].to(device)
            band = batch["band_ids"].to(device)
            survey = batch["survey_ids"].to(device)
            type_id = batch["type_ids"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            global_feat = batch["global_features"].to(device)

            _, cls, _ = model(values, band, survey, type_id, global_feat, valid_mask)
            cls_tokens.append(cls.cpu().numpy())

    cls_tokens = np.concatenate(cls_tokens, axis=0)
    
    # Normalize
    scaler = StandardScaler()
    cls_normalized = scaler.fit_transform(cls_tokens)

    # Subsample for speed
    n_samples = min(5000, len(cls_normalized))
    sample_idx = np.random.choice(len(cls_normalized), n_samples, replace=False)
    cls_sample = cls_normalized[sample_idx]

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', n_iter=1000)
    cls_tsne = tsne.fit_transform(cls_sample)

    plt.figure(figsize=(12, 10))
    plt.scatter(cls_tsne[:, 0], cls_tsne[:, 1], alpha=0.6, s=15, c='steelblue', edgecolors='none')
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.title('t-SNE Visualization of CLS Embeddings', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_masking_example(valid_mask, mask_positions, save_path=None):
    """Visualize masking positions for a single sample."""
    fig, ax = plt.subplots(figsize=(15, 3))
    ax.bar(range(len(valid_mask)), valid_mask.astype(float), width=0.8, label='Valid', color='lightblue')
    ax.bar(range(len(mask_positions)), mask_positions.astype(float) * 0.8, 
           width=0.8, label='Masked', color='darkred', alpha=0.7)
    ax.set_xlabel('Measurement Index', fontsize=12)
    ax.set_ylabel('Status', fontsize=12)
    ax.set_title('Fixed Masking: 20 Randomly Masked Positions (Red) Among Valid Measurements (Blue)', fontsize=12)
    ax.legend()
    ax.set_ylim(0, 1.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()