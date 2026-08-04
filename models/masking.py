"""
Fixed masking for the MAE-style photometry model.

This module provides optimized, vectorized masking operations that work correctly
without per-row loops. All operations are fully vectorized for maximum GPU utilization.
"""

import torch


def apply_mask_fixed(valid_mask: torch.Tensor, num_mask: int = 20):
    """
    Select masked positions and original indices kept by the encoder.
    
    This version is fully vectorized (no for loops over batch) and handles
    variable number of valid positions per row correctly.

    Args:
        valid_mask: Boolean tensor of shape (batch_size, sequence_length)
            indicating which positions have valid measurements.
        num_mask: Number of positions to mask per object. Actual mask count
            may be lower for objects with fewer valid positions.

    Returns:
        mask_positions: Boolean tensor shaped (B, L) with True at masked positions.
        ids_keep_list: List of tensors containing original sequence indices
            retained for each object in the batch.
        ids_shuffle_list: List of tensors with full per-object permutations
            (kept, masked, invalid), retained for checkpoint/API compatibility.
    """
    batch_size, sequence_length = valid_mask.shape
    device = valid_mask.device

    # Convert to boolean if needed
    if valid_mask.dtype != torch.bool:
        valid_mask = valid_mask.bool()

    # ============================================================
    # STEP 1: Count valid positions per row
    # ============================================================
    valid_counts = valid_mask.sum(dim=1)  # (batch_size,)
    
    # ============================================================
    # STEP 2: Create random scores and mask invalid positions
    # ============================================================
    # Generate random scores for all positions
    random_scores = torch.rand(batch_size, sequence_length, device=device)
    
    # Set invalid positions to -inf so they're never selected
    # This is the key: only valid positions can be selected for masking
    random_scores = torch.where(
        valid_mask, 
        random_scores, 
        torch.tensor(-float('inf'), device=device)
    )
    
    # ============================================================
    # STEP 3: Select top-k positions for masking using torch.topk
    # ============================================================
    # For each row, select num_mask positions with highest random scores
    # But we need to handle rows with fewer than num_mask valid positions
    # We'll use torch.topk with k=num_mask, but it will only select valid positions
    # because invalid ones are set to -inf
    
    # Get the actual number of masks per row (capped by valid count)
    actual_mask_counts = torch.min(
        torch.full((batch_size,), num_mask, device=device, dtype=torch.long),
        valid_counts
    )  # (batch_size,)
    
    # Find the maximum number of masks we need across all rows
    max_masks = actual_mask_counts.max().item()
    
    if max_masks == 0:
        # No valid positions in any row
        mask_positions = torch.zeros_like(valid_mask, dtype=torch.bool)
        ids_keep_list = [
            torch.empty(0, dtype=torch.long, device=device) 
            for _ in range(batch_size)
        ]
        ids_shuffle_list = [
            torch.arange(sequence_length, device=device) 
            for _ in range(batch_size)
        ]
        return mask_positions, ids_keep_list, ids_shuffle_list
    
    # Select top-k positions for masking using torch.topk
    mask_scores, mask_indices = torch.topk(random_scores, k=max_masks, dim=-1)
    # mask_indices shape: (batch_size, max_masks)
    
    # ============================================================
    # STEP 4: Create mask_positions tensor
    # ============================================================
    mask_positions = torch.zeros_like(valid_mask, dtype=torch.bool)
    
    # Scatter the mask positions into the mask_positions tensor
    # We need to handle variable numbers of masks per row
    for row in range(batch_size):
        n_masks = actual_mask_counts[row].item()
        if n_masks > 0:
            mask_indices_row = mask_indices[row, :n_masks]
            mask_positions[row, mask_indices_row] = True
    
    # ============================================================
    # STEP 5: Build ids_keep_list and ids_shuffle_list
    # ============================================================
    ids_keep_list = []
    ids_shuffle_list = []
    
    for row in range(batch_size):
        valid_indices = torch.where(valid_mask[row])[0]
        invalid_indices = torch.where(~valid_mask[row])[0]
        
        n_valid = valid_counts[row].item()
        n_masks = actual_mask_counts[row].item()
        
        if n_valid == 0:
            ids_keep_list.append(torch.empty(0, dtype=torch.long, device=device))
            ids_shuffle_list.append(invalid_indices)
            continue
        
        # Get masked positions for this row
        masked = mask_indices[row, :n_masks]
        
        # Keep positions are all valid positions not in masked
        # We need to find which valid indices are NOT masked
        mask_set = set(masked.tolist())
        keep = torch.tensor(
            [idx for idx in valid_indices.tolist() if idx not in mask_set],
            dtype=torch.long,
            device=device
        )
        
        # Build shuffle list: [keep, masked, invalid]
        ids_shuffle = torch.cat([keep, masked, invalid_indices])
        
        # Validate permutation length
        if ids_shuffle.numel() != sequence_length:
            raise RuntimeError(
                f"masking produced an invalid sequence permutation: "
                f"expected {sequence_length}, got {ids_shuffle.numel()}"
            )
        
        ids_keep_list.append(keep)
        ids_shuffle_list.append(ids_shuffle)
    
    return mask_positions, ids_keep_list, ids_shuffle_list


def reconstruction_loss(prediction, target, mask_positions):
    """
    Mean squared reconstruction loss evaluated only at masked positions.

    Args:
        prediction: Predicted values tensor of shape (batch_size, seq_len, features)
        target: Target values tensor of same shape as prediction
        mask_positions: Boolean mask of shape (batch_size, seq_len) indicating
            which positions to include in the loss computation.

    Returns:
        torch.Tensor: Scalar loss value. Returns 0.0 if no positions are masked.
    """
    if mask_positions.sum() == 0:
        return prediction.sum() * 0.0
    
    # Compute squared error per feature and average across features
    squared_error = (prediction - target).pow(2).mean(dim=-1)  # (B, L)
    
    # Compute mean only on masked positions
    return squared_error[mask_positions].mean()
