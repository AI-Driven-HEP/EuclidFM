# MAE-style Photometry Transformer v2

This version preserves the existing baseline v1 and adds the professional
MAE-style checkpoint as a separate implementation.

## Architectural difference

- **v1:** BERT-style masking. Mask tokens remain in the encoder input.
- **v2:** MAE-style masking. Masked measurements are removed before the encoder,
  then mask tokens are inserted and the original sequence order is restored for
  the Transformer decoder.

## New v2 files

- `config_v2.py`
- `models/masking_v2.py`
- `models/embedding_v2.py`
- `models/transformer_v2.py`
- `training/trainer_v2.py`
- `training/evaluator_v2.py`
- `visualize_v2.py`
- `scripts/run_training_v2.py`

No existing v1 Python file is replaced.

## Run v2

Place `euclid_random_100k_even_3fields_all_features.csv` in the repository root,
install the dependencies, and run:

```bash
pip install -r requirements.txt
python scripts/run_training_v2.py
```

The default v2 configuration matches the checkpoint: 10,000 sampled objects,
80/20 train-validation split, `d_model=64`, 4 attention heads, 8 encoder
layers, 4 decoder layers, batch size 8, learning rate `3e-4`, weight decay
`0.01`, 100 epochs, and 20 masked valid measurements per object.

Generated v2 checkpoints, models, evaluation tables, logs, and plots use
separate `_v2` directories and do not overwrite v1 outputs.

