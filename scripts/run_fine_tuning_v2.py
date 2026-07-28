"""Fine-tune EuclidFM v2 for QSO/star/galaxy classification."""

import json
import logging
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from config_finetune_v2 import FineTuneV2Config
from data.preprocessing import (
    build_measurement_dataset_arcsinh,
    extract_global_features,
    find_measurements,
    fit_global_scaler,
    transform_global,
)
from models.classifier_v2 import EuclidFMClassifierV2


class LabelledPhotometryDataset(Dataset):
    """Tensor dataset that also keeps labels and object IDs."""

    def __init__(self, arrays, global_features, labels, object_ids):
        values, mask, band, survey, type_id = arrays[:5]
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.mask = torch.as_tensor(mask, dtype=torch.bool)
        self.band = torch.as_tensor(band, dtype=torch.long)
        self.survey = torch.as_tensor(survey, dtype=torch.long)
        self.type_id = torch.as_tensor(type_id, dtype=torch.long)
        self.global_features = torch.as_tensor(global_features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.object_ids = torch.as_tensor(object_ids, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "values": self.values[index],
            "valid_mask": self.mask[index],
            "band_ids": self.band[index],
            "survey_ids": self.survey[index],
            "type_ids": self.type_id[index],
            "global_features": self.global_features[index],
            "label": self.labels[index],
            "object_id": self.object_ids[index],
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalized_key(name):
    """Match IRSA names to prefixed Euclid export names."""
    key = str(name).strip().upper()
    for prefix in ("MER__", "PHZ__"):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def align_to_reference(labelled, reference_columns):
    """Rename fine-tuning columns to the names used during pretraining."""
    lookup = {}
    for column in labelled.columns:
        lookup.setdefault(normalized_key(column), []).append(column)

    rename = {}
    missing = []
    ambiguous = []
    for reference_name in reference_columns:
        if reference_name in labelled.columns:
            continue
        candidates = lookup.get(normalized_key(reference_name), [])
        if len(candidates) == 1:
            rename[candidates[0]] = reference_name
        elif len(candidates) > 1:
            ambiguous.append((reference_name, candidates))
        else:
            missing.append(reference_name)

    if ambiguous:
        raise ValueError(f"Ambiguous feature mappings: {ambiguous[:5]}")
    return labelled.rename(columns=rename), missing


def locate_column(frame, requested):
    """Find a column without depending on capitalization."""
    matches = [column for column in frame if column.lower() == requested.lower()]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely find column '{requested}'.")
    return matches[0]


def load_inputs(cfg):
    labelled = pd.read_csv(cfg.labelled_csv, low_memory=False)
    reference = pd.read_csv(cfg.reference_csv, low_memory=False)

    label_col = locate_column(labelled, cfg.label_column)
    id_col = locate_column(labelled, cfg.object_id_column)
    labelled[label_col] = pd.to_numeric(labelled[label_col], errors="raise").astype(int)
    labelled[id_col] = pd.to_numeric(labelled[id_col], errors="raise").astype("int64")

    if not labelled[label_col].isin(range(cfg.num_classes)).all():
        raise ValueError("Labels must be QSO=0, star=1 and galaxy=2.")
    if labelled[id_col].duplicated().any():
        raise ValueError("The labelled CSV contains duplicate object IDs.")

    # The reference file defines the exact pretrained measurement schema.
    measurements = find_measurements(reference)
    required = [item[key] for item in measurements for key in ("flux", "error")]
    required += [
        "MER__RIGHT_ASCENSION",
        "MER__DECLINATION",
        "MER__FLUX_DETECTION_TOTAL",
    ]
    labelled, missing = align_to_reference(labelled, required)
    missing_required = [name for name in missing if name not in labelled.columns]
    if missing_required:
        report = cfg.output_dir / "missing_fine_tuning_features.txt"
        report.write_text("\n".join(missing_required), encoding="utf-8")
        raise ValueError(
            f"{len(missing_required)} pretrained features are missing. "
            f"See {report}."
        )

    # Restore canonical names for downstream bookkeeping.
    labelled = labelled.rename(columns={label_col: "label", id_col: "object_id"})
    return labelled, measurements


def stratified_split(frame, cfg):
    """Create 70/15/15 splits while preserving each class ratio."""
    train, remainder = train_test_split(
        frame,
        test_size=1.0 - cfg.train_fraction,
        stratify=frame["label"],
        random_state=cfg.seed,
    )
    relative_test = cfg.test_fraction / (
        cfg.validation_fraction + cfg.test_fraction
    )
    validation, test = train_test_split(
        remainder,
        test_size=relative_test,
        stratify=remainder["label"],
        random_state=cfg.seed,
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def make_dataset(frame, measurements, scaler):
    arrays = build_measurement_dataset_arcsinh(frame, measurements)
    raw_global = extract_global_features(frame)
    global_features = transform_global(raw_global, scaler)
    return LabelledPhotometryDataset(
        arrays,
        global_features,
        frame["label"].to_numpy(),
        frame["object_id"].to_numpy(dtype="int64"),
    ), arrays[5], arrays[6], arrays[7]


def load_pretrained_encoder(model, path, device):
    """Load either a checkpoint dictionary or a plain model state dictionary."""
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    result = model.backbone.load_state_dict(state, strict=True)
    logging.info("Loaded pretrained encoder from %s (%s)", path, result)


def class_weights(labels, num_classes, device):
    counts = np.bincount(labels, minlength=num_classes)
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, device, optimizer=None, clip_norm=1.0):
    training = optimizer is not None
    model.train(training)
    losses, true_labels, predictions, object_ids = [], [], [], []

    with torch.set_grad_enabled(training):
        for batch in loader:
            labels = batch.pop("label").to(device)
            ids = batch.pop("object_id")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(
                batch["values"], batch["band_ids"], batch["survey_ids"],
                batch["type_ids"], batch["global_features"], batch["valid_mask"],
            )
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            losses.append(loss.item() * len(labels))
            true_labels.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            object_ids.extend(ids.tolist())

    return {
        "loss": sum(losses) / len(true_labels),
        "accuracy": accuracy_score(true_labels, predictions),
        "macro_f1": f1_score(true_labels, predictions, average="macro"),
        "y_true": true_labels,
        "y_pred": predictions,
        "object_id": object_ids,
    }


def optimizer_for(model, cfg, encoder_is_frozen):
    if encoder_is_frozen:
        return torch.optim.AdamW(
            model.classifier.parameters(),
            lr=cfg.head_lr,
            weight_decay=cfg.weight_decay,
        )
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": cfg.encoder_lr},
            {"params": model.classifier.parameters(), "lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )


def save_test_outputs(result, cfg):
    predictions = pd.DataFrame({
        "object_id": result["object_id"],
        "true_label": result["y_true"],
        "predicted_label": result["y_pred"],
    })
    predictions.to_csv(cfg.output_dir / "test_predictions.csv", index=False)

    report = classification_report(
        result["y_true"], result["y_pred"],
        target_names=cfg.class_names, output_dict=True, digits=4,
    )
    (cfg.output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    matrix = confusion_matrix(result["y_true"], result["y_pred"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=cfg.class_names, yticklabels=cfg.class_names,
    )
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "confusion_matrix.png", dpi=200)
    plt.close()


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
    )
    cfg = FineTuneV2Config()
    cfg.validate()
    set_seed(cfg.seed)

    frame, measurements = load_inputs(cfg)
    cfg.seq_len = len(measurements)
    train_frame, validation_frame, test_frame = stratified_split(frame, cfg)

    # Fit position scaling on training data only to prevent leakage.
    scaler = fit_global_scaler(extract_global_features(train_frame))
    train_set, band_vocab, survey_vocab, type_vocab = make_dataset(
        train_frame, measurements, scaler
    )
    validation_set, _, _, _ = make_dataset(validation_frame, measurements, scaler)
    test_set, _, _, _ = make_dataset(test_frame, measurements, scaler)

    split_table = pd.concat([
        train_frame.assign(split="train"),
        validation_frame.assign(split="validation"),
        test_frame.assign(split="test"),
    ])[["object_id", "label", "split"]]
    split_table.to_csv(cfg.output_dir / "data_splits.csv", index=False)

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.device.startswith("cuda"),
    )
    validation_loader = DataLoader(
        validation_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.device.startswith("cuda"),
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.device.startswith("cuda"),
    )

    model = EuclidFMClassifierV2(
        cfg, len(band_vocab), len(survey_vocab), len(type_vocab)
    ).to(cfg.device)
    load_pretrained_encoder(model, cfg.pretrained_checkpoint, cfg.device)
    model.freeze_encoder()

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights(train_frame["label"].to_numpy(), cfg.num_classes, cfg.device),
        label_smoothing=cfg.label_smoothing,
    )
    optimizer = optimizer_for(model, cfg, encoder_is_frozen=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    history, best_f1, waiting = [], -np.inf, 0
    for epoch in range(cfg.epochs):
        # First train only the new head, then carefully unfreeze the encoder.
        if epoch == cfg.frozen_epochs:
            model.unfreeze_encoder()
            optimizer = optimizer_for(model, cfg, encoder_is_frozen=False)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=3
            )
            logging.info("Encoder unfrozen at epoch %d", epoch + 1)

        train_result = run_epoch(
            model, train_loader, criterion, cfg.device, optimizer,
            cfg.gradient_clip_norm,
        )
        validation_result = run_epoch(
            model, validation_loader, criterion, cfg.device
        )
        scheduler.step(validation_result["macro_f1"])
        row = {
            "epoch": epoch + 1,
            "train_loss": train_result["loss"],
            "validation_loss": validation_result["loss"],
            "train_macro_f1": train_result["macro_f1"],
            "validation_macro_f1": validation_result["macro_f1"],
        }
        history.append(row)
        logging.info(
            "Epoch %02d — train loss %.4f — val macro-F1 %.4f",
            epoch + 1, row["train_loss"], row["validation_macro_f1"],
        )

        if validation_result["macro_f1"] > best_f1:
            best_f1 = validation_result["macro_f1"]
            waiting = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": vars(cfg),
                "measurements": measurements,
                "band_vocab": band_vocab,
                "survey_vocab": survey_vocab,
                "type_vocab": type_vocab,
                "global_scaler": scaler,
                "validation_macro_f1": best_f1,
            }, cfg.best_model_path)
        else:
            waiting += 1
            if waiting >= cfg.patience:
                logging.info("Early stopping after %d epochs", epoch + 1)
                break

    pd.DataFrame(history).to_csv(
        cfg.output_dir / "training_history.csv", index=False
    )
    best = torch.load(cfg.best_model_path, map_location=cfg.device)
    model.load_state_dict(best["model_state_dict"])
    test_result = run_epoch(model, test_loader, criterion, cfg.device)
    save_test_outputs(test_result, cfg)
    logging.info(
        "Test accuracy %.4f — test macro-F1 %.4f",
        test_result["accuracy"], test_result["macro_f1"],
    )


if __name__ == "__main__":
    main()

