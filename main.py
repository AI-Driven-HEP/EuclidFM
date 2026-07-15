"""
Main entry point for training and evaluation.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from config import get_default_config
from src.utils.helpers import set_global_seed
from src.training.trainer import run_training


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Photometry Transformer")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--version", type=str, default="v1", 
                        choices=["v1", "v2"], help="Model version")
    args = parser.parse_args()
    
    # Load config
    if args.version == "v1":
        from config import get_default_config
        config = get_default_config()
    else:
        from config import get_config_v2
        config = get_config_v2()
    
    # Override with command line
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.lr = args.lr
    if args.device:
        config.training.device = args.device
    
    # Set seed
    set_global_seed(config.data.seed)
    
    logger.info(f"Using version: {args.version}")
    logger.info(f"Config: {config}")
    
    # Run training
    run_training(config)


if __name__ == "__main__":
    main()
