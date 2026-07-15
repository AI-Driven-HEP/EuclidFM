"""
Dataset class for photometry data.
"""

import torch
from torch.utils.data import Dataset


class PhotometryDataset(Dataset):
    """
    Dataset for pre-transformed photometry data.
    
    Args:
        values: (N, L, 2) array of [flux, error] values
        mask: (N, L) boolean mask indicating valid measurements
        band_ids: (N, L) band IDs
        survey_ids: (N, L) survey IDs
        type_ids: (N, L) type IDs
        global_features: (N, 4) global features [sin_ra, cos_ra, dec_scaled, flux_transformed]
    """
    
    def __init__(self, values, mask, band_ids, survey_ids, type_ids, global_features):
        self.values = torch.tensor(values, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.band_ids = torch.tensor(band_ids, dtype=torch.long)
        self.survey_ids = torch.tensor(survey_ids, dtype=torch.long)
        self.type_ids = torch.tensor(type_ids, dtype=torch.long)
        self.global_features = torch.tensor(global_features, dtype=torch.float32)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return {
            "values": self.values[idx],
            "valid_mask": self.mask[idx],
            "band_ids": self.band_ids[idx],
            "survey_ids": self.survey_ids[idx],
            "type_ids": self.type_ids[idx],
            "global_features": self.global_features[idx],
        }
