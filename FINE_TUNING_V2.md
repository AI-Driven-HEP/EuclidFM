# EuclidFM v2 three-class fine-tuning

This extension fine-tunes the pretrained MAE encoder to classify:

- `0`: QSO
- `1`: star
- `2`: galaxy

## Required files

Place these files in the repository root:

```text
3-Class Sample.csv
euclid_random_100k_even_3fields_all_features.csv
models_v2/photometry_transformer_v2_best.pt
```

The original 100k CSV is used only to reconstruct the exact measurement order
and categorical vocabularies seen during pretraining. It is not added to the
supervised train, validation or test sets.

## Run

```bash
pip install -r requirements.txt
python scripts/run_fine_tuning_v2.py
```

The script creates stratified 70/15/15 splits, trains the classification head
for five frozen-encoder epochs, unfreezes the encoder at a lower learning rate,
and selects the best checkpoint using validation macro-F1.

Outputs are written to `fine_tuning_outputs_v2/`:

- `euclidfm_v2_3class_best.pt`
- `data_splits.csv`
- `training_history.csv`
- `test_predictions.csv`
- `classification_report.json`
- `confusion_matrix.png`

The test set is evaluated only after model selection is complete.
