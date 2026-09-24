"""
Data preprocessing functions.
"""

import numpy as np
import pandas as pd


WAVELENGTHS = {
    "u": 3551, "g": 4686, "r": 6166, "i": 7480, "z": 8932,
    "y": 9600, "vis": 7150, "j": 12500, "h": 16500,
    "nir": 15000, "nir_stack": 15000,
}


def parse_measurement_column(colname: str):
    """
    Parse photometry columns like:
        MER__FLUX_<band>_<type>
        MER__FLUX_<band>_<survey>_<type>
        PHZ__FLUX_<band>_<survey>_<type>
        and FLUXERR equivalents
    
    Returns a dict with keys: colname, band, survey, type, is_err
    """
    if "__FLUX" not in colname and "__FLUXERR" not in colname:
        return None
    
    parts = colname.split("__", 1)
    if len(parts) < 2:
        return None
    
    right = parts[1]
    toks = right.split("_")
    
    if toks[0] not in ("FLUX", "FLUXERR"):
        return None
    
    is_err = toks[0] == "FLUXERR"
    toks = toks[1:]
    
    if not toks:
        return None
    
    band = toks[0]
    remaining = toks[1:]
    
    # Handle NIR_STACK
    if band.upper() == "NIR" and remaining and remaining[0].upper() == "STACK":
        band = "NIR_STACK"
        remaining = remaining[1:]
    
    band = band.lower()
    
    SURVEY_TOKS = {
        "EXT_DECAM", "EXT_HSC", "EXT_MEGACAM", 
        "EXT_PANSTARRS", "EXT_LSST", "EXT_JPCAM"
    }
    
    survey = "euclid"
    
    if len(remaining) >= 2:
        candidate = f"{remaining[0]}_{remaining[1]}".upper()
        if candidate in SURVEY_TOKS:
            survey = candidate.lower()
            remaining = remaining[2:]
    
    if len(remaining) >= 1:
        candidate = f"EXT_{remaining[0].upper()}"
        if candidate in SURVEY_TOKS:
            survey = candidate.lower()
            remaining = remaining[1:]
    
    if not remaining:
        return None
    
    if any(tok.upper() == "TO" for tok in remaining):
        return None
    
    type_name = "_".join(remaining).lower()
    
    return {
        "colname": colname,
        "band": band,
        "survey": survey,
        "type": type_name,
        "is_err": is_err,
    }


def find_measurements(df: pd.DataFrame) -> list:
    """
    Scan DataFrame columns and reconstruct paired (flux, fluxerr) measurements.
    Removes measurements that are entirely zero or NaN.
    Returns a list of measurement dicts sorted by wavelength.
    """
    parsed_cols = []
    
    for col in df.columns:
        # Skip detection columns
        if col in ("MER__FLUX_DETECTION_TOTAL", "MER__FLUXERR_DETECTION_TOTAL"):
            continue
        
        # Only process measurement columns
        if not (col.startswith("MER__FLUX_") or col.startswith("PHZ__FLUX_") or
                col.startswith("MER__FLUXERR_") or col.startswith("PHZ__FLUXERR_")):
            continue
        
        parsed = parse_measurement_column(col)
        if parsed is None:
            continue
        
        parsed_cols.append(parsed)
    
    # Group by (band, survey, type)
    measurements_by_key = {}
    for p in parsed_cols:
        key = (p["band"], p["survey"], p["type"])
        if key not in measurements_by_key:
            measurements_by_key[key] = {"flux": None, "error": None}
        
        if p["is_err"]:
            measurements_by_key[key]["error"] = p["colname"]
        else:
            measurements_by_key[key]["flux"] = p["colname"]
    
    # Build measurement list
    measurements = []
    for (band, survey, type_name), entry in measurements_by_key.items():
        flux_col = entry["flux"]
        err_col = entry["error"]
        
        if flux_col is None or err_col is None:
            continue
        
        flux_vals = df[flux_col].values.astype(np.float64)
        err_vals = df[err_col].values.astype(np.float64)
        
        flux_all_zero = np.all((flux_vals == 0) | np.isnan(flux_vals))
        err_all_zero = np.all((err_vals == 0) | np.isnan(err_vals))
        
        if flux_all_zero or err_all_zero:
            continue
        
        measurements.append({
            "flux": flux_col,
            "error": err_col,
            "band": band,
            "survey": survey,
            "type": type_name,
            "wavelength": WAVELENGTHS.get(band, np.nan),
        })
    
    # Sort by wavelength
    measurements.sort(key=lambda x: (np.isnan(x["wavelength"]), x["wavelength"]))
    
    return measurements


def build_survey_vocab(measurements: list) -> dict:
    """
    Build vocabulary for survey names, always putting 'euclid' as index 0.
    """
    surveys = sorted({m["survey"] for m in measurements})
    vocab = {"euclid": 0}
    idx = 1
    for s in surveys:
        if s != "euclid":
            vocab[s] = idx
            idx += 1
    return vocab


def apply_arcsinh(df: pd.DataFrame, flux_col: str, err_col: str):
    """
    Apply plain arcsinh transformation to flux and error columns.
    No scaling, no fitting needed - deterministic.
    """
    flux = df[flux_col].values.astype(np.float64)
    err = df[err_col].values.astype(np.float64)
    
    # Apply arcsinh
    flux_transformed = np.arcsinh(flux)
    err_transformed = np.arcsinh(err)
    
    # Handle invalid values
    valid_flux = ~(np.isnan(flux) | np.isinf(flux))
    valid_err = ~(np.isnan(err) | np.isinf(err))
    
    flux_transformed[~valid_flux] = 0.0
    err_transformed[~valid_err] = 0.0
    
    # Mask is valid only where BOTH original flux and error are valid
    valid_mask = valid_flux & valid_err
    
    return flux_transformed.astype(np.float32), err_transformed.astype(np.float32), valid_mask


def build_measurement_dataset_arcsinh(df: pd.DataFrame, measurements: list, true_class_label=False):
    """
    Build measurement arrays using plain arcsinh transformation.
    
    Returns:
        measurement_values, measurement_mask, band_ids, survey_ids, type_ids,
        band_vocab, survey_vocab, type_vocab
    """
    n_obj = len(df)
    n_meas = len(measurements)
    F = 2  # flux and error
    
    measurement_values = np.zeros((n_obj, n_meas, F), dtype=np.float32)
    measurement_mask = np.zeros((n_obj, n_meas), dtype=bool)
    band_ids = np.zeros((n_obj, n_meas), dtype=np.int64)
    survey_ids = np.zeros((n_obj, n_meas), dtype=np.int64)
    type_ids = np.zeros((n_obj, n_meas), dtype=np.int64)
    
    # Build vocabularies
    band_vocab = {b: i for i, b in enumerate(sorted({m["band"] for m in measurements}))}
    survey_vocab = build_survey_vocab(measurements)
    types = sorted({m["type"] for m in measurements})
    type_vocab = {t: i for i, t in enumerate(types)}
    
    for j, m in enumerate(measurements):
        flux_transformed, err_transformed, valid_mask = apply_arcsinh(
            df, m["flux"], m["error"]
        )
        
        measurement_values[:, j, 0] = flux_transformed
        measurement_values[:, j, 1] = err_transformed
        measurement_mask[:, j] = valid_mask
        
        band_ids[:, j] = band_vocab[m["band"]]
        survey_ids[:, j] = survey_vocab[m["survey"]]
        type_ids[:, j] = type_vocab[m["type"]]
    
    if true_class_label:
        labels = df["label"].to_numpy(dtype=np.int64)
        return (measurement_values, measurement_mask, band_ids, survey_ids, type_ids, labels,
                band_vocab, survey_vocab, type_vocab)
    else:
        return (measurement_values, measurement_mask, band_ids, survey_ids, type_ids,
                band_vocab, survey_vocab, type_vocab)


def extract_global_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract global features (RA, DEC, detection total flux) from dataframe.
    """
    n_obj = len(df)
    global_features = np.zeros((n_obj, 3), dtype=np.float32)
    
    for i, col in enumerate([
        "MER__RIGHT_ASCENSION", 
        "MER__DECLINATION", 
        "MER__FLUX_DETECTION_TOTAL"
    ]):
        if col in df.columns:
            vals = df[col].values.astype(np.float64)
        else:
            vals = np.zeros(n_obj)
        global_features[:, i] = vals
    
    return global_features


def fit_global_scaler(train_global: np.ndarray) -> dict:
    """
    Fit scaling parameters for DEC only.
    RA is transformed using sin/cos (no scaling needed).
    Flux uses arcsinh transformation directly.
    """
    dec = train_global[:, 1]
    
    # DEC: min-max scaling to [-1, 1]
    dec_min, dec_max = dec.min(), dec.max()
    eps = 1e-8
    
    if dec_max - dec_min < eps:
        dec_min, dec_max = -90.0, 90.0
    
    dec_range = (dec_max - dec_min) + eps
    
    return {
        'dec_min': dec_min,
        'dec_max': dec_max,
        'dec_range': dec_range,
    }


def transform_global(global_feat: np.ndarray, scaler: dict) -> np.ndarray:
    """
    Transform global features:
        - RA (0-360 deg) -> sin(ra), cos(ra)
        - DEC -> min-max scaled to [-1, 1]
        - Detection flux -> arcsinh(flux)
    
    Returns: [sin_ra, cos_ra, dec_scaled, flux_transformed]
    """
    # RA: convert to sin/cos (preserves periodicity)
    ra_rad = np.radians(global_feat[:, 0])
    ra_sin = np.sin(ra_rad)
    ra_cos = np.cos(ra_rad)
    
    # DEC: min-max scaling to [-1, 1]
    dec = global_feat[:, 1]
    dec_scaled = (dec - scaler['dec_min']) / scaler['dec_range'] * 2 - 1
    dec_scaled = np.clip(dec_scaled, -1.0, 1.0)
    
    # Detection flux: direct arcsinh transformation
    det_flux = global_feat[:, 2]
    flux_transformed = np.arcsinh(det_flux)
    
    return np.stack([ra_sin, ra_cos, dec_scaled, flux_transformed], axis=1).astype(np.float32)
