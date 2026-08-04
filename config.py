"""Configuration for the MAE-style training pipeline."""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Config:
    input_csv: str = "euclid_random_100k_even_3fields_all_features.csv"
    cleaned_csv: str = "cleaned_output.csv"
    dataset_npz: str = "euclid_measurements_dataset.npz"
    sample_size: int = 10_000
    validation_fraction: float = 0.2
    seed: int = 42
    global_features: int = 4
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 8
    decoder_layers: int = 4
    ff_dim: int = 512
    dropout: float = 0.1
    use_cls_token: bool = True
    batch_size: int = 32
    num_workers: int = 0
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 100
    num_mask: int = 20
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 999
    checkpoint_frequency: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir: Path = Path("models")
    checkpoints_dir: Path = Path("checkpoints")
    evaluation_dir: Path = Path("evaluation_outputs")
    visualization_dir: Path = Path("visualizations")
    logs_dir: Path = Path("logs")

    def create_output_directories(self):
        for directory in (
            self.models_dir,
            self.checkpoints_dir,
            self.evaluation_dir,
            self.visualization_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def final_model_path(self):
        return self.models_dir / "photometry_transformer_final.pt"

    @property
    def best_model_path(self):
        return self.models_dir / "photometry_transformer_best.pt"

    @property
    def training_history_path(self):
        return self.logs_dir / "training_history.json"

