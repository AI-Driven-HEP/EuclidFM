"""
Masking utilities for training - FIXED MASKING VERSION
Based on original notebook: pretraining_fixed_mask_final.ipynb
"""
import torch


def apply_mask_fixed(valid_mask, num_mask=20):
    """
    Generate exactly `num_mask` mask positions per sample (only from valid positions).
    If a sample has fewer valid positions than num_mask, mask all valid positions.
    
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
            # No valid positions, skip
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
        mask: (B, L) - Binary mask (1=masked, 0=kept) - می‌تواند Float یا Bool باشد
        valid_mask: (B, L) - True where measurement is valid
    
    Returns:
        loss: scalar MSE loss averaged over masked positions
    """
    # تبدیل mask به boolean (اگر Float باشد)
    if mask.dtype == torch.bool:
        mask_bool = mask
    else:
        mask_bool = mask.bool()  # ✅ تبدیل Float به Bool
    
    # Only compute loss on positions that are:
    # 1. Masked (mask == 1 or True)
    # 2. Valid (valid_mask == True)
    valid_mask_bool = valid_mask & mask_bool
    
    if valid_mask_bool.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    
    # MSE per feature, then mean over features, then select masked positions
    diff = (pred - target).pow(2).mean(dim=-1)  # (B, L)
    masked_diff = diff[valid_mask_bool]
    
    return masked_diff.mean()
