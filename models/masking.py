"""
Masking utilities for training - CORRECTED VERSION
Masking happens on embeddings AFTER embedding layer
"""
import torch


def random_masking_embeddings(x, mask_ratio):
    """
    Randomly mask tokens from embeddings.
    
    Args:
        x: (B, N, D) - Token embeddings
        mask_ratio: float - Ratio of tokens to mask
    
    Returns:
        x_masked: (B, N_keep, D) - Unmasked tokens
        mask: (B, N) - Binary mask (1=removed, 0=kept)
        ids_keep: (B, N_keep) - Indices of kept tokens
        ids_shuffle: (B, N) - Shuffled indices for restoration
    """
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))
    
    # Randomly shuffle indices
    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    
    # Keep unmasked patches
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    
    # Generate mask for loss computation (1=removed, 0=kept)
    mask = torch.ones([B, N], device=x.device)
    mask[:, :len_keep] = 0
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return x_masked, mask, ids_keep, ids_shuffle


def apply_mask_fixed(valid_mask, num_mask=20):
    """
    Fixed masking for training: Generate mask positions.
    This is used BEFORE embedding to determine which positions to mask.
    
    Args:
        valid_mask: (B, L) boolean tensor, True where measurement is valid
        num_mask: number of positions to mask per sample (default: 20)
    
    Returns:
        mask_positions: (B, L) boolean tensor, True for positions to mask
    """
    B, L = valid_mask.shape
    device = valid_mask.device
    mask_positions = torch.zeros(B, L, dtype=torch.bool, device=device)
    
    for i in range(B):
        # Get indices of valid positions for this sample
        valid_indices = torch.where(valid_mask[i])[0]
        n_valid = len(valid_indices)
        
        if n_valid == 0:
            continue
        
        # Number of positions to mask (min between num_mask and n_valid)
        k = min(num_mask, n_valid)
        
        # Randomly select k indices from valid_indices without replacement
        if k > 0:
            perm = torch.randperm(n_valid, device=device)[:k]
            selected = valid_indices[perm]
            mask_positions[i, selected] = True
    
    return mask_positions


def reconstruction_loss(pred, target, mask, valid_mask):
    """
    MSE loss ONLY on masked positions.
    
    Args:
        pred: (B, L, 2) - predicted [flux, error]
        target: (B, L, 2) - ground truth [flux, error]
        mask: (B, L) - Binary mask (1=masked, 0=kept)
        valid_mask: (B, L) - True where measurement is valid
    
    Returns:
        loss: scalar MSE loss averaged over masked positions
    """
    # Only compute loss on positions that are:
    # 1. Masked (mask == 1)
    # 2. Valid (valid_mask == True)
    valid_mask_bool = valid_mask & (mask == 1)
    
    if valid_mask_bool.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    
    # MSE per feature, then mean over features, then select masked positions
    diff = (pred - target).pow(2).mean(dim=-1)  # (B, L)
    masked_diff = diff[valid_mask_bool]
    
    return masked_diff.mean()