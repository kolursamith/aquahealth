# data/

Dataset images are **not** committed to this repository — only the directory
structure and instructions are.

```
data/
├── original/   # raw, unmodified dataset as collected (2,400 images, 8 classes)
├── train/      # stratified training split, produced by scripts/create_split.py
├── val/        # stratified validation split
└── test/       # stratified test split, held out until final evaluation
```

Each split directory is organized by class:

```
data/train/
├── Aeromoniasis/
├── Bacterial_Gill_Disease/
├── Columnaris/
├── Dropsy/
├── Fin_Rot/
├── Healthy/
├── Saprolegniasis/
└── White_Spot/
```

## Getting the dataset

Obtain the dataset from the shared team drive (see the hackathon kickoff
doc) and place the raw images under `data/original/<class_name>/`.

## Creating splits

```bash
python scripts/create_split.py
```

## Auditing

```bash
python scripts/audit_dataset.py
```

Checks image counts per class, corrupt files, and file extensions before
training.
