# EuclidFM

EuclidFM is a masked autoencoder Transformer for Euclid Q1 catalogue photometry. It learns a 128-dimensional source embedding from up to 94 flux and uncertainty pairs. The model uses an eight-layer encoder and a four-layer reconstruction decoder.

## Pretrain

Install the dependencies in the repository root.

```bash
pip install -r requirements.txt
```

Download the original 100k working catalogue described in the [inference notebook](notebook/euclidfm_inference.ipynb). In a fresh checkout, save it at the repository root as `euclid_random_100k_even_3fields_all_features.csv`, then run:

```bash
python main.py --batch-size 64 --epochs 1000 --no-global
```

These are the settings for the long run that passed epoch 900. The quality cuts leave 90,258 sources. The seed-42 split assigns 72,206 sources to training and 18,052 to validation, grouped by object ID. Each training sample masks up to 20 valid measurements. The script writes the best validation checkpoint to `models/photometry_transformer_best.pt`.

## Use the pretrained encoder

The release checkpoint and run summary will be linked here. The checkpoint comes from the long run and records its best validation epoch, 656.

Open the [inference notebook](notebook/euclidfm_inference.ipynb) to load the checkpoint, extract CLS embeddings, and use them to train a quasar, star and galaxy classifier with either a frozen or a fine-tuned encoder. The notebook lists the labelled Platinum Sample and the working catalogue used for the examples.

