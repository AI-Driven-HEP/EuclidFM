"""
Masking utilities for training
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
        mask_positions: (B, L) boolean tensor
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


def reconstruction_loss(pred, target, mask_positions, valid_mask):
    """
    MSE loss ONLY on masked positions.
    
    Args:
        pred: (B, L, 2) - predicted [flux, error]
        target: (B, L, 2) - ground truth [flux, error]
        mask_positions: (B, L) - True for positions that were masked
        valid_mask: (B, L) - True for valid positions (unused but kept for API consistency)
    
    Returns:
        loss: scalar MSE loss averaged over masked positions
    """
    # mask_positions already only includes valid positions (guaranteed by apply_mask_fixed)
    if mask_positions.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    
    # MSE per feature, then mean over features, then select masked positions
    diff = (pred - target).pow(2).mean(dim=-1)  # (B, L)
    masked_diff = diff[mask_positions]
    
    return masked_diff.mean()
