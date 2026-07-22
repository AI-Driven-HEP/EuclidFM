"""Configuration for the MAE-style v2 training pipeline."""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class V2Config:
    input_csv: str = "euclid_random_100k_even_3fields_all_features.csv"
    cleaned_csv: str = "cleaned_output.csv"
    dataset_npz: str = "euclid_measurements_dataset_v2.npz"
    sample_size: int = 10_000
    validation_fraction: float = 0.2
    seed: int = 42
    global_features: int = 4
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 8
    decoder_layers: int = 4
    ff_dim: int = 256
    dropout: float = 0.0
    use_cls_token: bool = True
    batch_size: int = 8
    num_workers: int = 0
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 100
    num_mask: int = 20
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 999
    checkpoint_frequency: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir: Path = Path("models_v2")
    checkpoints_dir: Path = Path("checkpoints_v2")
    evaluation_dir: Path = Path("evaluation_outputs_v2")
    visualization_dir: Path = Path("visualizations_v2")
    logs_dir: Path = Path("logs_v2")

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
        return self.models_dir / "photometry_transformer_v2_final.pt"

    @property
    def best_model_path(self):
        return self.models_dir / "photometry_transformer_v2_best.pt"

    @property
    def training_history_path(self):
        return self.logs_dir / "training_history_v2.json"

