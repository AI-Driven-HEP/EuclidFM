"""Configuration for supervised three-class fine-tuning."""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class FineTuneV2Config:
    # Inputs
    labelled_csv: str = "3-Class Sample.csv"
    reference_csv: str = "euclid_random_100k_even_3fields_all_features.csv"
    pretrained_checkpoint: str = "models_v2/photometry_transformer_v2_best.pt"

    # Labels: QSO=0, star=1, galaxy=2
    label_column: str = "label"
    object_id_column: str = "object_id"
    class_names: tuple = ("QSO", "Star", "Galaxy")
    num_classes: int = 3

    # Reproduce the pretrained v2 architecture exactly
    global_features: int = 4
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 8
    decoder_layers: int = 4
    ff_dim: int = 256
    dropout: float = 0.0
    use_cls_token: bool = True

    # Split and optimization
    seed: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 64
    num_workers: int = 0
    frozen_epochs: int = 5
    epochs: int = 50
    head_lr: float = 3e-4
    encoder_lr: float = 3e-5
    weight_decay: float = 1e-2
    label_smoothing: float = 0.05
    patience: int = 10
    gradient_clip_norm: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir: Path = Path("fine_tuning_outputs_v2")

    def validate(self):
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-8:
            raise ValueError("Train, validation and test fractions must sum to 1.")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def best_model_path(self):
        return self.output_dir / "euclidfm_v2_3class_best.pt"

