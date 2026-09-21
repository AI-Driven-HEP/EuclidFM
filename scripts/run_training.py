"""
Run the complete MAE-style training and evaluation pipeline.

UPDATED:
  - Extracts per-object metadata (object_id, RA, DEC, total flux, PHZ class)
    from the cleaned DataFrame and passes it to `create_plots` so that
    per-object diagnostics (R² per object, t-SNE colouring by RA/DEC/flux/
    PHZ class) can be generated.
  - Passes `object_ids` into the PhotometryDataset for both train and val.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from config import Config
from data import (
    PhotometryDataset,
    build_measurement_dataset_arcsinh,
    extract_global_features,
    find_measurements,
    fit_global_scaler,
    transform_global,
)
from models.transformer import PhotometryTransformer
from training.evaluator import evaluate
from training.trainer import train
from utils.helpers import set_global_seed, setup_logging
from visualize import create_plots

# ==============================================================================
# CONFIGURATION FOR VISUALIZATION METADATA
# ------------------------------------------------------------------------------
# Adjust these column names to match your input CSV.
# ==============================================================================
RA_COL = "RIGHT_ASCENSION"              # or "RA", "ra"
DEC_COL = "DECLINATION"                 # or "DEC", "dec"
TOTAL_FLUX_COL = "FLUX_TOTAL"           # column holding total detection flux
CLASS_COL = "PHZ__PHZ_CLASSIFICATION"   # 0=quasar, 1=star, 2=galaxy

# If the class column contains strings, map them to integers:
CLASS_STR_TO_INT = {
    "quasar": 0, "qso": 0,
    "star":   1,
    "galaxy": 2, "gal": 2,
}

# List of object IDs to remove due to quality issues
REMOVE_OBJECT_IDS = [
    -516833347281158785,
    -524325240264858394,
    -526763971264021159,
    -531358091287592116,
    -601550136483347596,
    -605798685503436276,
    -621407362500663238,
    -632954073493539985,
    -633889436492517261,
    -635092693485500394,
    -643401708482019153,
    2664991879655406635,
    2675618383640896787,
    2676463652656186488,
    2676932786640872754,
    2679138096642491394,
    2690071563641296650,
]

logger = logging.getLogger(__name__)


def _project_path(path_value):
    """Convert relative paths to absolute paths relative to project root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_config_paths(cfg):
    """Keep config names unchanged while making relative paths repository-safe."""
    cfg.input_csv = str(_project_path(cfg.input_csv))
    cfg.cleaned_csv = str(_project_path(cfg.cleaned_csv))
    cfg.dataset_npz = str(_project_path(cfg.dataset_npz))
    cfg.models_dir = _project_path(cfg.models_dir)
    cfg.checkpoints_dir = _project_path(cfg.checkpoints_dir)
    cfg.evaluation_dir = _project_path(cfg.evaluation_dir)
    cfg.visualization_dir = _project_path(cfg.visualization_dir)
    cfg.logs_dir = _project_path(cfg.logs_dir)
    return cfg


def load_and_filter(input_csv, cleaned_csv):
    """
    Apply the repository's Euclid quality filters and save cleaned data.

    1. Read the input CSV.
    2. Apply quality filters (detection and spurious flags).
    3. Remove specific object IDs from the removal list.
    4. Save the cleaned data to a new CSV.
    """
    frame = pd.read_csv(input_csv)

    required = {"OBJECT_ID", "MER__DET_QUALITY_FLAG", "MER__SPURIOUS_FLAG"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Required input columns are missing: {missing}")

    detection_flag = pd.to_numeric(
        frame["MER__DET_QUALITY_FLAG"], errors="coerce"
    )
    spurious_flag = pd.to_numeric(frame["MER__SPURIOUS_FLAG"], errors="coerce")
    keep = detection_flag.isin([0, 2, 512]) & (spurious_flag == 0)

    cleaned = frame.loc[
        keep & ~frame["OBJECT_ID"].isin(REMOVE_OBJECT_IDS)
    ].dropna(axis=1, how="all").copy()

    cleaned_path = Path(cleaned_csv)
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(cleaned_path, index=False)
    logger.info("cleaning kept %d of %d objects", len(cleaned), len(frame))
    logger.debug(f"Columns in cleaned data: {cleaned.columns.tolist()}")

    return cleaned


def detect_object_id_column(df):
    """Detect the object ID column name in the dataframe."""
    possible_names = ['OBJECT_ID', 'object_id', 'ObjID', 'id', 'ID', 'obj_id']
    for name in possible_names:
        if name in df.columns:
            logger.info(f"Found object ID column: '{name}'")
            return name

    for col in df.columns:
        if 'id' in col.lower() or 'object' in col.lower():
            logger.warning(f"Using fallback object ID column: '{col}'")
            return col

    logger.error(f"Available columns: {df.columns.tolist()}")
    raise ValueError("Could not find object ID column in the data!")


def check_object_overlap(train_df, val_df, object_id_col):
    """Comprehensive overlap check between train and validation sets."""
    train_objects = set(train_df[object_id_col].unique())
    val_objects = set(val_df[object_id_col].unique())

    overlap = train_objects & val_objects
    only_train = train_objects - val_objects
    only_val = val_objects - train_objects

    report = {
        'total_train_objects': len(train_objects),
        'total_val_objects': len(val_objects),
        'overlap_count': len(overlap),
        'overlap_percent': len(overlap) / len(train_objects) * 100 if train_objects else 0,
        'only_train': len(only_train),
        'only_val': len(only_val),
        'overlap_examples': list(overlap)[:5] if overlap else []
    }

    logger.info("=" * 60)
    logger.info("📊 OBJECT OVERLAP REPORT")
    logger.info("=" * 60)
    logger.info(f"Total objects in Train:     {report['total_train_objects']}")
    logger.info(f"Total objects in Validation: {report['total_val_objects']}")
    logger.info(f"Overlap (in BOTH):          {report['overlap_count']} ({report['overlap_percent']:.1f}%)")
    logger.info(f"Only in Train:              {report['only_train']}")
    logger.info(f"Only in Validation:         {report['only_val']}")

    if report['overlap_count'] > 0:
        logger.warning("=" * 60)
        logger.warning("❌ OVERLAP DETECTED! Data leakage risk!")
        logger.warning("=" * 60)
        for obj in report['overlap_examples']:
            train_rows = len(train_df[train_df[object_id_col] == obj])
            val_rows = len(val_df[val_df[object_id_col] == obj])
            logger.warning(f"   {obj}: {train_rows} rows in Train, {val_rows} rows in Validation")
        logger.warning("=" * 60)
        raise RuntimeError(
            f"Object overlap detected! {report['overlap_count']} objects appear in both "
            f"train and validation sets. This indicates data leakage."
        )
    else:
        logger.info("=" * 60)
        logger.info("✅ NO OVERLAP! Data is cleanly separated by object.")
        logger.info("=" * 60)

    return report


def _extract_class_labels(frame, class_col):
    """
    Extract per-object PHZ class labels from the dataframe.

    Handles both numeric (0/1/2) and string ("star"/"galaxy"/"quasar") encodings.
    Returns None if the column is missing.
    """
    if class_col not in frame.columns:
        logger.warning(
            f"Column '{class_col}' not found — class-based plots will be skipped."
        )
        return None

    raw = frame[class_col]

    # Numeric encoding
    if pd.api.types.is_numeric_dtype(raw):
        return raw.to_numpy(dtype=int)

    # String encoding — map to integers
    logger.info(
        f"Column '{class_col}' contains strings — mapping "
        f"via CLASS_STR_TO_INT."
    )
    mapped = raw.astype(str).str.lower().str.strip().map(CLASS_STR_TO_INT)
    return mapped.fillna(-1).to_numpy(dtype=int)


def main(config: Config | None = None):
    """Train and evaluate using only the MAE-style implementation."""
    cfg = _resolve_config_paths(config or Config())
    setup_logging(log_file=cfg.logs_dir / "training.log")
    cfg.create_output_directories()
    set_global_seed(cfg.seed)

    if str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; is using CPU instead.")
        cfg.device = "cpu"

    # ============================================================
    # STEP 1: LOAD AND CLEAN DATA
    # ============================================================
    cleaned = load_and_filter(cfg.input_csv, cfg.cleaned_csv)
    if len(cleaned) < 2:
        raise ValueError("At least two cleaned objects are required for training.")

    object_id_col = detect_object_id_column(cleaned)
    logger.info(f"Using object ID column: '{object_id_col}'")

    measurements = find_measurements(cleaned)
    if not measurements:
        raise ValueError("No paired flux/flux-error measurements were found.")
    cfg.seq_len = len(measurements)
    logger.info(f"Found {len(measurements)} measurement pairs")

    # ============================================================
    # STEP 2: SAMPLE AND SPLIT DATA (OBJECT-AWARE)
    # ============================================================
    sample_count = min(cfg.sample_size, len(cleaned))
    sampled = cleaned.sample(n=sample_count, random_state=cfg.seed)
    logger.info(f"Sampled {sample_count} objects from {len(cleaned)} total")

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=cfg.validation_fraction,
        random_state=cfg.seed
    )
    train_idx, val_idx = next(gss.split(sampled, groups=sampled[object_id_col]))

    train_frame = sampled.iloc[train_idx]
    validation_frame = sampled.iloc[val_idx]

    logger.info(f"Train size: {len(train_frame)} rows ({len(train_frame[object_id_col].unique())} objects)")
    logger.info(f"Validation size: {len(validation_frame)} rows ({len(validation_frame[object_id_col].unique())} objects)")

    # ============================================================
    # STEP 3: SANITY CHECK — NO OBJECT OVERLAP
    # ============================================================
    check_object_overlap(train_frame, validation_frame, object_id_col)

    # ============================================================
    # STEP 4: BUILD MEASUREMENT DATASETS
    # ============================================================
    train_arrays = build_measurement_dataset_arcsinh(train_frame, measurements)
    validation_arrays = build_measurement_dataset_arcsinh(
        validation_frame, measurements
    )
    (
        train_values,
        train_mask,
        train_band,
        train_survey,
        train_type,
        band_vocab,
        survey_vocab,
        type_vocab,
    ) = train_arrays
    (
        validation_values,
        validation_mask,
        validation_band,
        validation_survey,
        validation_type,
        _,
        _,
        _,
    ) = validation_arrays

    # ---- NEW: per-row object ids (must match the row order of *_values) ----
    train_object_ids = train_frame[object_id_col].to_numpy()
    validation_object_ids = validation_frame[object_id_col].to_numpy()

    # ============================================================
    # STEP 5: EXTRACT AND NORMALIZE GLOBAL FEATURES
    # ============================================================
    train_global_raw = extract_global_features(train_frame)
    validation_global_raw = extract_global_features(validation_frame)
    global_scaler = fit_global_scaler(train_global_raw)
    train_global = transform_global(train_global_raw, global_scaler)
    validation_global = transform_global(validation_global_raw, global_scaler)

    # ---- NEW: extract per-object metadata for visualisation ----
    #      The cleaned DataFrame has one row per object, so the row order of
    #      validation_frame matches the row order of validation_values/mask/etc.
    val_ra_per_obj = (
        validation_frame[RA_COL].to_numpy(dtype=float)
        if RA_COL in validation_frame.columns else None
    )
    val_dec_per_obj = (
        validation_frame[DEC_COL].to_numpy(dtype=float)
        if DEC_COL in validation_frame.columns else None
    )
    val_flux_per_obj = (
        validation_frame[TOTAL_FLUX_COL].to_numpy(dtype=float)
        if TOTAL_FLUX_COL in validation_frame.columns else None
    )
    val_class_per_obj = _extract_class_labels(validation_frame, CLASS_COL)

    # Report class distribution, if available
    if val_class_per_obj is not None:
        unique, counts = np.unique(val_class_per_obj, return_counts=True)
        name_map = {0: "Quasar", 1: "Star", 2: "Galaxy", -1: "Unknown"}
        logger.info("=" * 60)
        logger.info("🌟 PHZ CLASS DISTRIBUTION (validation set)")
        logger.info("=" * 60)
        for code, n in zip(unique, counts):
            logger.info(f"  code={int(code):>2}  {name_map.get(int(code), 'N/A'):<8}  n = {n:,}")
        logger.info("=" * 60)

    # ============================================================
    # STEP 6: SAVE DATASET FOR FUTURE USE
    # ============================================================
    np.savez(
        cfg.dataset_npz,
        train_values=train_values,
        train_mask=train_mask,
        train_band=train_band,
        train_survey=train_survey,
        train_type=train_type,
        train_global=train_global,
        train_object_ids=train_object_ids,          # NEW
        validation_values=validation_values,
        validation_mask=validation_mask,
        validation_band=validation_band,
        validation_survey=validation_survey,
        validation_type=validation_type,
        validation_global=validation_global,
        validation_object_ids=validation_object_ids, # NEW
        # NEW: per-object metadata (all optional; NaN for missing)
        validation_ra=(val_ra_per_obj if val_ra_per_obj is not None
                       else np.full(len(validation_frame), np.nan)),
        validation_dec=(val_dec_per_obj if val_dec_per_obj is not None
                        else np.full(len(validation_frame), np.nan)),
        validation_total_flux=(val_flux_per_obj if val_flux_per_obj is not None
                               else np.full(len(validation_frame), np.nan)),
        validation_class=(val_class_per_obj if val_class_per_obj is not None
                          else np.full(len(validation_frame), -1, dtype=int)),
    )
    logger.info(f"Dataset saved to: {cfg.dataset_npz}")

    # ============================================================
    # STEP 7: CREATE PYTORCH DATASETS AND DATALOADERS
    # ============================================================
    train_dataset = PhotometryDataset(
        train_values,
        train_mask,
        train_band,
        train_survey,
        train_type,
        train_global,
        object_ids=train_object_ids,           # NEW
    )
    validation_dataset = PhotometryDataset(
        validation_values,
        validation_mask,
        validation_band,
        validation_survey,
        validation_type,
        validation_global,
        object_ids=validation_object_ids,      # NEW
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Validation loader: {len(validation_loader)} batches")

    # ============================================================
    # STEP 8: INITIALIZE MODEL
    # ============================================================
    model = PhotometryTransformer(
        cfg,
        n_band=len(band_vocab),
        n_survey=len(survey_vocab),
        n_type=len(type_vocab),
    ).to(cfg.device)
    logger.info(f"Model initialized on: {cfg.device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  - Use global features: {cfg.use_global_features}")

    # ============================================================
    # STEP 8.5: LOAD BEST MODEL IF REQUESTED
    # ============================================================
    if hasattr(cfg, 'resume_from_best') and cfg.resume_from_best:
        best_path = cfg.best_model_path
        if best_path.exists():
            logger.info(f"Loading best model from: {best_path}")
            checkpoint = torch.load(best_path, map_location=cfg.device)
            model.load_state_dict(checkpoint['model_state_dict'])

            best_val_loss = checkpoint.get('validation_loss', float('inf'))
            best_epoch = checkpoint.get('epoch', 0)
            logger.info(f"Loaded best model from epoch {best_epoch} with validation loss: {best_val_loss:.6f}")
        else:
            logger.warning(f"Best model not found at {best_path}. Training from scratch.")
            cfg.resume_from_best = False

    # ============================================================
    # STEP 9: TRAIN THE MODEL
    # ============================================================
    logger.info("Starting training...")
    history = train(model, train_loader, validation_loader, cfg)
    logger.info("Training complete!")

    # ============================================================
    # STEP 10: EVALUATE THE MODEL
    # ============================================================
    logger.info("Starting evaluation...")
    results = evaluate(model, validation_loader, cfg)
    logger.info("Evaluation complete!")

    # ============================================================
    # STEP 11: BUILD dataset_metadata FOR VISUALIZATION
    # ============================================================
    # The evaluate() function returns:
    #     results["sample_object"] : (N_samples,) → which object row
    #     results["object_ids"]    : (N_obj,)     → original object ids
    # We now expand per-object metadata to per-sample arrays using sample_object.

    sample_object = results["sample_object"]           # (N_samples,)
    object_ids_per_obj = results["object_ids"]         # (N_obj,)

    # Per-sample lookups (used by visualize.py for R² per-object aggregation)
    total_flux_sample = (
        val_flux_per_obj[sample_object]
        if val_flux_per_obj is not None else None
    )
    ra_sample = (
        val_ra_per_obj[sample_object]
        if val_ra_per_obj is not None else None
    )
    dec_sample = (
        val_dec_per_obj[sample_object]
        if val_dec_per_obj is not None else None
    )

    dataset_metadata = {
        # Per-sample (for R² per object)
        "object_ids":     sample_object,
        "total_flux":     total_flux_sample,
        "ra":             ra_sample,
        "dec":            dec_sample,

        # Per-object (for t-SNE colouring by RA/DEC/flux/class)
        "total_flux_obj": val_flux_per_obj,
        "ra_obj":         val_ra_per_obj,
        "dec_obj":        val_dec_per_obj,
        "class_labels_obj": val_class_per_obj,
    }

    # Diagnostic summary
    logger.info("=" * 60)
    logger.info("📊 dataset_metadata SHAPES")
    logger.info("=" * 60)
    for k, v in dataset_metadata.items():
        shape = None if v is None else np.asarray(v).shape
        logger.info(f"  {k:<20} {shape}")
    logger.info("=" * 60)

    # ============================================================
    # STEP 12: CREATE VISUALIZATION PLOTS
    # ============================================================
    logger.info("Creating visualization plots...")
    create_plots(
        history,
        results,
        cfg.visualization_dir,
        cfg.seed,
        dataset_metadata=dataset_metadata,
    )
    logger.info(f"Plots saved to: {cfg.visualization_dir}")

    # ============================================================
    # STEP 13: FINAL SUMMARY
    # ============================================================
    logger.info("=" * 60)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final model: {cfg.final_model_path}")
    logger.info(f"Best model: {cfg.best_model_path}")
    logger.info(f"History: {cfg.training_history_path}")
    logger.info("=" * 60)

    return model, history, results


if __name__ == "__main__":
    main()
