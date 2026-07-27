"""Fixed masking for the v2 MAE-style photometry model."""

import torch


def apply_mask_fixed_v2(valid_mask: torch.Tensor, num_mask: int = 20):
    """Select masked positions and return the unmasked indices for the encoder."""
    batch_size, sequence_length = valid_mask.shape
    device = valid_mask.device
    mask_positions = torch.zeros_like(valid_mask, dtype=torch.bool)
    ids_keep_list = []
    ids_shuffle_list = []


    #This comes with huge concerns, because we essentially need to make parallel on batch!!!!!
    for row in range(batch_size):
        valid_indices = torch.where(valid_mask[row])[0]
        number_valid = len(valid_indices)

        if number_valid == 0:
            ids_keep_list.append(torch.empty(0, dtype=torch.long, device=device))
            ids_shuffle_list.append(torch.arange(sequence_length, device=device))
            continue

        number_masked = min(num_mask, number_valid)
        shuffled_valid = valid_indices[torch.randperm(number_valid, device=device)]
        ids_keep = shuffled_valid[: number_valid - number_masked]
        ids_masked = shuffled_valid[number_valid - number_masked :]

        mask_positions[row, ids_masked] = True
        ids_shuffle_full = torch.arange(sequence_length, device=device)
        ids_shuffle_full[valid_indices] = shuffled_valid
        ids_keep_list.append(ids_keep)
        ids_shuffle_list.append(ids_shuffle_full)

    return mask_positions, ids_keep_list, ids_shuffle_list


def reconstruction_loss_v2(prediction, target, mask_positions):
    """Mean squared reconstruction loss evaluated only at masked positions."""
    if mask_positions.sum() == 0:
        return prediction.sum() * 0.0
    squared_error = (prediction - target).pow(2).mean(dim=-1)
    return squared_error[mask_positions].mean()

