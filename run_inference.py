"""
Run to produce the data for inference.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data import (
    build_measurement_dataset_arcsinh,
    extract_global_features,
    find_measurements,
    fit_global_scaler,
    transform_global,
)


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

def load_and_filter(input_csv, cleaned_csv):
    """
    Apply the repository's Euclid quality filters and save cleaned data.
    
    This function:
    1. Reads the input CSV file
    2. Applies quality filters (detection and spurious flags)
    3. Removes specific object IDs from the removal list
    4. Saves the cleaned data to a new CSV file
    """
    frame = pd.read_csv(input_csv)
    
    # Check for required columns
    required = {"OBJECT_ID", "MER__DET_QUALITY_FLAG", "MER__SPURIOUS_FLAG"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Required input columns are missing: {missing}")

    # Apply quality filters
    detection_flag = pd.to_numeric(
        frame["MER__DET_QUALITY_FLAG"], errors="coerce"
    )
    spurious_flag = pd.to_numeric(frame["MER__SPURIOUS_FLAG"], errors="coerce")
    keep = detection_flag.isin([0, 2, 512]) & (spurious_flag == 0)
    
    # Remove specific object IDs and drop empty columns
    cleaned = frame.loc[
        keep & ~frame["OBJECT_ID"].isin(REMOVE_OBJECT_IDS)
    ].dropna(axis=1, how="all").copy()

    # Save cleaned data
    cleaned_path = Path(cleaned_csv)
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(cleaned_path, index=False)    
    return cleaned


def main():
    # Check that we got exactly 2 arguments after the script name
    if len(sys.argv) not in (3, 4):
        print("Usage: python script.py <input_csv> <cleaned_csv> [--labels]")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    cleaned_csv = sys.argv[2]
    with_labels = len(sys.argv) == 4 and sys.argv[3] in ("--labels", "-l", "true", "1")

    cleaned = load_and_filter(input_csv, cleaned_csv)

    # Find measurement columns (flux/flux_error pairs)
    measurements = find_measurements(cleaned)
    if not measurements:
        raise ValueError("No paired flux/flux-error measurements were found.")

    validation_frame = cleaned.sample(n=len(cleaned), random_state=42)
    
    validation_arrays = build_measurement_dataset_arcsinh(
        validation_frame, measurements, true_class_label=with_labels
    )

    if "z_vi" in validation_frame.columns:
        redshift = validation_frame["z_vi"].to_numpy(dtype=np.float64)
    else:
        redshift = None

    if with_labels:
        (
            validation_values,
            validation_mask,
            validation_band,
            validation_survey,
            validation_type,
            validation_labels,
            _,
            _,
            _,
        ) = validation_arrays
    else:
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
        validation_labels = None
    
    validation_global_raw = extract_global_features(validation_frame)
    global_scaler = fit_global_scaler(validation_global_raw)
    validation_global = transform_global(validation_global_raw, global_scaler)


    npz_path = Path(cleaned_csv).with_suffix("")
    npz_path = npz_path.parent / (npz_path.name + "_measurements_dataset.npz")

    np.savez(
        npz_path,
        validation_values=validation_values,
        validation_mask=validation_mask,
        validation_band=validation_band,
        validation_survey=validation_survey,
        validation_type=validation_type,
        validation_global=validation_global,
        validation_labels=validation_labels,
        redshift = redshift,
    )
  
if __name__ == "__main__":
    main()
