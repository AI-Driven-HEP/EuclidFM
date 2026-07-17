"""Run the complete MAE-style v2 training and evaluation pipeline."""

import logging
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config_v2 import V2Config
from data import (
    PhotometryDataset,
    build_measurement_dataset_arcsinh,
    extract_global_features,
    find_measurements,
    fit_global_scaler,
    transform_global,
)
from models.transformer_v2 import PhotometryTransformerV2
from training.evaluator_v2 import evaluate_v2
from training.trainer_v2 import train_v2
from visualize_v2 import create_v2_plots

REMOVE_OBJECT_IDS = [
    -516833347281158785, -524325240264858394, -526763971264021159,
    -531358091287592116, -601550136483347596, -605798685503436276,
    -621407362500663238, -632954073493539985, -633889436492517261,
    -635092693485500394, -643401708482019153, 2664991879655406635,
    2675618383640896787, 2676463652656186488, 2676932786640872754,
    2679138096642491394, 2690071563641296650,
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_filter(input_csv, cleaned_csv):
    frame = pd.read_csv(input_csv)
    detection_flag = pd.to_numeric(
        frame["MER__DET_QUALITY_FLAG"], errors="coerce"
    )
    spurious_flag = pd.to_numeric(frame["MER__SPURIOUS_FLAG"], errors="coerce")
    keep = detection_flag.isin([0, 2, 512]) & (spurious_flag == 0)
    cleaned = frame.loc[
        keep & ~frame["OBJECT_ID"].isin(REMOVE_OBJECT_IDS)
    ].dropna(axis=1, how="all").copy()
    cleaned.to_csv(cleaned_csv, index=False)
    logging.info("v2 cleaning kept %d of %d objects", len(cleaned), len(frame))
    return cleaned


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )
    cfg = V2Config()
    cfg.create_output_directories()
    set_seed(cfg.seed)

    cleaned = load_and_filter(cfg.input_csv, cfg.cleaned_csv)
    measurements = find_measurements(cleaned)
    cfg.seq_len = len(measurements)
    sample_count = min(cfg.sample_size, len(cleaned))
    sampled = cleaned.sample(n=sample_count, random_state=cfg.seed)
    train_frame, validation_frame = train_test_split(
        sampled,
        test_size=cfg.validation_fraction,
        random_state=cfg.seed,
    )

    train_arrays = build_measurement_dataset_arcsinh(train_frame, measurements)
    validation_arrays = build_measurement_dataset_arcsinh(
        validation_frame, measurements
    )
    (
        train_values, train_mask, train_band, train_survey, train_type,
        band_vocab, survey_vocab, type_vocab,
    ) = train_arrays
    (
        validation_values, validation_mask, validation_band,
        validation_survey, validation_type, _, _, _,
    ) = validation_arrays

    train_global = extract_global_features(train_frame)
    validation_global = extract_global_features(validation_frame)
    global_scaler = fit_global_scaler(train_global)
    train_global = transform_global(train_global, global_scaler)
    validation_global = transform_global(validation_global, global_scaler)
    np.savez(
        cfg.dataset_npz,
        train_values=train_values,
        train_mask=train_mask,
        validation_values=validation_values,
        validation_mask=validation_mask,
    )

    train_dataset = PhotometryDataset(
        train_values, train_mask, train_band, train_survey, train_type,
        train_global,
    )
    validation_dataset = PhotometryDataset(
        validation_values, validation_mask, validation_band,
        validation_survey, validation_type, validation_global,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = PhotometryTransformerV2(
        cfg,
        n_band=len(band_vocab),
        n_survey=len(survey_vocab),
        n_type=len(type_vocab),
    ).to(cfg.device)
    history = train_v2(model, train_loader, validation_loader, cfg)
    results = evaluate_v2(model, validation_loader, cfg)
    create_v2_plots(history, results, cfg.visualization_dir, cfg.seed)
    logging.info("v2 pipeline complete: %s", cfg.final_model_path)


if __name__ == "__main__":
    main()

