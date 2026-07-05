from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Config:
    # Paths
    input_csv: Path = Path("euclid_random_100k_even_3fields_all_features.csv")
    cleaned_csv: Path = Path("cleaned_output.csv")
    output_npz: Path = Path("euclid_measurements_dataset.npz")
    model_save_path: Path = Path("photometry_transformer_pretrained.pt")
    best_model_path: Path = Path("photometry_transformer_best.pt")
    
    # Data
    seq_len: int = 94  # will be set dynamically
    batch_size: int = 8
    num_workers: int = 0
    num_mask: int = 20  # fixed masking: 20 positions per sample
    
    # Model
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 8
    ff_dim: int = 256
    dropout: float = 0.0
    use_cls_token: bool = True
    global_features: int = 4  # sin_ra, cos_ra, dec_scaled, arcsinh(flux)
    
    # Training
    lr: float = 3e-4
    epochs: int = 100
    weight_decay: float = 0.0
    seed: int = 42
    
    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()