"""
Configuration for Photometry Transformer
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Data configuration."""
    
    input_csv: str = "euclid_random_100k_even_3fields_all_features.csv"
    cleaned_csv: str = "cleaned_output.csv"
    output_npz: str = "euclid_measurements_dataset.npz"
    n_samples: int = 10000
    test_size: float = 0.2
    seed: int = 42
    id_column: str = "OBJECT_ID"
    
    detection_columns: List[str] = field(default_factory=lambda: [
        "MER__RIGHT_ASCENSION",
        "MER__DECLINATION",
        "MER__FLUX_DETECTION_TOTAL"
    ])


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Core architecture
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 8
    ff_dim: int = 256
    dropout: float = 0.0
    
    # Embedding dimensions
    cont_dim: int = 64
    band_dim: int = 8
    survey_dim: int = 8
    type_dim: int = 4
    
    # Features
    use_cls_token: bool = True
    global_features: int = 4


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # DataLoader
    batch_size: int = 8
    num_workers: int = 0
    
    # Optimization
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 0.0  # V1: 0.0, V2: 0.01
    clip_grad_norm: float = 1.0
    
    # Masking
    mask_fixed: int = 20
    
    # Hardware
    device: str = "cuda"
    
    # Saving
    checkpoint_dir: str = "checkpoints/"
    model_save_path: str = "photometry_transformer_pretrained.pt"
    best_model_path: str = "photometry_transformer_best.pt"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    level: str = "INFO"
    save_dir: str = "logs/"
    save_frequency: int = 1


@dataclass
class Config:
    """Master configuration."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create config from dictionary."""
        return cls(
            data=DataConfig(**data.get('data', {})),
            model=ModelConfig(**data.get('model', {})),
            training=TrainingConfig(**data.get('training', {})),
            logging=LoggingConfig(**data.get('logging', {}))
        )


def get_default_config() -> Config:
    """Get default configuration (v1 baseline)."""
    return Config()


def get_config_v2() -> Config:
    """Get configuration for v2 (MAE-style)."""
    cfg = Config()
    cfg.training.weight_decay = 0.01  # Only difference from v1
    return cfg
