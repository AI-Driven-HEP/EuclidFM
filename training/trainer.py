"""Training, checkpointing, and metric history for the MAE-style model."""

import json
import logging
from datetime import datetime

import torch
from tqdm import tqdm

from models.masking import apply_mask_fixed, reconstruction_loss

logger = logging.getLogger(__name__)


class TrainingHistory:
    def __init__(self):
        self.train_losses = []
        self.validation_losses = []
        self.learning_rates = []
        self.epoch_times = []
        self.best_validation_loss = float("inf")
        self.best_epoch = -1

    def update(self, train_loss, validation_loss, learning_rate, seconds):
        self.train_losses.append(train_loss)
        self.validation_losses.append(validation_loss)
        self.learning_rates.append(learning_rate)
        self.epoch_times.append(seconds)
        if validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            self.best_epoch = len(self.train_losses) - 1

    def save(self, path):
        payload = vars(self).copy()
        payload["timestamp"] = datetime.now().isoformat()
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_epoch(model, dataloader, cfg, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    valid_batches = 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch in tqdm(
            dataloader,
            desc="train" if training else "validation",
            leave=False,
        ):
            batch = {key: value.to(cfg.device) for key, value in batch.items()}
            mask_positions, ids_keep, ids_shuffle = apply_mask_fixed(
                batch["valid_mask"], cfg.num_mask
            )
            prediction, _ = model(
                batch["values"],
                batch["band_ids"],
                batch["survey_ids"],
                batch["type_ids"],
                batch["global_features"],
                batch["valid_mask"],
                mask_positions,
                ids_keep,
                ids_shuffle,
            )
            loss = reconstruction_loss(
                prediction, batch["values"], mask_positions
            )
            if not torch.isfinite(loss):
                logger.warning("Skipping a batch with non-finite loss")
                continue

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.gradient_clip_norm
                )
                optimizer.step()

            total_loss += loss.item()
            valid_batches += 1

    return total_loss / max(valid_batches, 1)


def train(model, train_loader, validation_loader, cfg):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    history = TrainingHistory()
    latest_path = cfg.checkpoints_dir / "latest_checkpoint.pt"
    start_epoch = 0
    best_loss = float("inf")
    patience_counter = 0

    if latest_path.exists():
        try:
            checkpoint = torch.load(
                latest_path, map_location=cfg.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(latest_path, map_location=cfg.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint.get("best_validation_loss", best_loss)
        patience_counter = checkpoint.get("patience_counter", 0)
        logger.info("Resuming training at epoch %d", start_epoch + 1)

    for epoch in range(start_epoch, cfg.epochs):
        epoch_start = datetime.now()
        train_loss = _run_epoch(model, train_loader, cfg, optimizer)
        validation_loss = _run_epoch(model, validation_loader, cfg)
        scheduler.step(validation_loss)
        seconds = (datetime.now() - epoch_start).total_seconds()
        history.update(
            train_loss,
            validation_loss,
            optimizer.param_groups[0]["lr"],
            seconds,
        )

        improved = validation_loss < best_loss
        if improved:
            best_loss = validation_loss
            patience_counter = 0
        else:
            patience_counter += 1

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "best_validation_loss": best_loss,
            "patience_counter": patience_counter,
        }
        torch.save(checkpoint, latest_path)
        if (epoch + 1) % cfg.checkpoint_frequency == 0:
            torch.save(
                checkpoint,
                cfg.checkpoints_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt",
            )
        if improved:
            torch.save(checkpoint, cfg.best_model_path)

        logger.info(
            "epoch %d/%d — train %.6f, validation %.6f",
            epoch + 1,
            cfg.epochs,
            train_loss,
            validation_loss,
        )
        if patience_counter >= cfg.early_stopping_patience:
            logger.info("Stopping training early")
            break

    torch.save(model.state_dict(), cfg.final_model_path)
    history.save(cfg.training_history_path)
    return history

