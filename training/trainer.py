"""
Training loop - CORRECTED VERSION
"""
import logging
import torch
from tqdm import tqdm
from models.masking import reconstruction_loss

logger = logging.getLogger(__name__)


class Trainer:
    """Training class for photometry transformer."""
    
    def __init__(self, model, optimizer, device, best_model_path):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.best_model_path = best_model_path
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader):
        """Run one training epoch."""
        self.model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch in tqdm(train_loader, desc="Train", leave=False):
            values = batch["values"].to(self.device)
            band = batch["band_ids"].to(self.device)
            survey = batch["survey_ids"].to(self.device)
            type_id = batch["type_ids"].to(self.device)
            valid_mask = batch["valid_mask"].to(self.device)
            global_feat = batch["global_features"].to(self.device)

            # Forward pass with correct masking sequence
            pred, _, mask = self.model(values, band, survey, type_id, global_feat, valid_mask)
            loss = reconstruction_loss(pred, values, mask, valid_mask)

            if torch.isnan(loss):
                logger.warning("NaN loss detected, skipping batch")
                continue

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        return train_loss / max(train_batches, 1)
    
    def validate_epoch(self, val_loader):
        """Run one validation epoch."""
        self.model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                values = batch["values"].to(self.device)
                band = batch["band_ids"].to(self.device)
                survey = batch["survey_ids"].to(self.device)
                type_id = batch["type_ids"].to(self.device)
                valid_mask = batch["valid_mask"].to(self.device)
                global_feat = batch["global_features"].to(self.device)

                pred, _, mask = self.model(values, band, survey, type_id, global_feat, valid_mask)
                loss = reconstruction_loss(pred, values, mask, valid_mask)

                if not torch.isnan(loss):
                    val_loss += loss.item()
                    val_batches += 1

        return val_loss / max(val_batches, 1)
    
    def train(self, train_loader, val_loader, epochs, model_save_path=None):
        """Full training loop."""
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            logger.info(f"========== EPOCH {epoch+1}/{epochs} ==========")
            
            avg_train_loss = self.train_epoch(train_loader)
            train_losses.append(avg_train_loss)
            logger.info(f"Train Loss: {avg_train_loss:.6f}")
            
            avg_val_loss = self.validate_epoch(val_loader)
            val_losses.append(avg_val_loss)
            logger.info(f"Validation Loss: {avg_val_loss:.6f}")
            
            # Save best model
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': self.best_val_loss,
                }, self.best_model_path)
                logger.info(f"Best model saved to {self.best_model_path} (val_loss: {self.best_val_loss:.6f})")
        
        # Save final model
        if model_save_path:
            torch.save(self.model.state_dict(), model_save_path)
            logger.info(f"Final model saved to {model_save_path}")
        
        return train_losses, val_losses
