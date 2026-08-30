"""Command-line entry point for the MAE-style pipeline."""

import argparse

from config import Config
from scripts.run_training import main as run


def parse_args():
    parser = argparse.ArgumentParser(description="Train Photometry Transformer")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--device", type=str, help="Override device")
    parser.add_argument("--resume-best", action="store_true", 
                       help="Resume training from the best model checkpoint")
    
    # NEW: Global features control
    parser.add_argument("--use-global", action="store_true", 
                       help="Use global features (RA, DEC, flux) in the model")
    parser.add_argument("--no-global", action="store_true", 
                       help="Disable global features in the model")
    
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.device is not None:
        cfg.device = args.device
    if args.resume_best:
        cfg.resume_from_best = True
    
    # Handle global features control
    if args.use_global:
        cfg.use_global_features = True
    elif args.no_global:
        cfg.use_global_features = False
    # If neither is provided, default (True) is used

    return run(cfg)


if __name__ == "__main__":
    main()
