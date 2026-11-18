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





## Data and pretrained model

The datasets and pretrained model used in EuclidFM are available for download:

- **100k Working Catalogue:** [Download from Google Drive](https://drive.google.com/file/d/1khSYyTy-0g2pXlsBQGGnk8quafTcRJ3_/view?usp=sharing) — The original catalogue containing 100,000 sources from the three Euclid Deep Fields. After quality cuts, 90,258 sources are retained.

- **Platinum Sample:** [Download from Google Drive](https://drive.google.com/file/d/1E1sxxm7pDyZwqe7soBRaZhJ1Yiv7lqQX/view?usp=sharing) — A labelled sample of 1,452 spectroscopically identified quasars, stars, and galaxies for downstream classification.

- **Pretrained EuclidFM Checkpoint:** [Download from Google Drive](https://drive.google.com/file/d/1yCHxGV2ErNsJN2xRxb45pPzYpbFnBkSg/view?usp=sharing) — Model weights from the best validation epoch (656).

## Use the pretrained encoder

Download the checkpoint and save it as `models/photometry_transformer_best.pt`.

Open the [inference notebook](notebook/euclidfm_inference.ipynb) to load the checkpoint, extract 128-dimensional CLS embeddings, and train a quasar, star, and galaxy classifier using either a frozen or fine-tuned encoder.




