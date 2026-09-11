# Training Pipeline

```
data/train, data/val
    ↓
src/dataset.py        (Dataset, DataLoader)
    ↓
src/preprocessing.py  (CLAHE, resize, normalize)
    ↓
src/augmentation.py   (train-only: flip, rotation, brightness/contrast, crop/zoom)
    ↓
src/model.py           (EfficientNet-B0, 8-class head)
    ↓
src/train.py           (loss, backprop, optimizer, validation loop)
    ↓
models/final_model.pth (best checkpoint by validation metric)
    ↓
src/evaluate.py        (accuracy, precision, recall, macro-F1, confusion matrix)
    ↓
results/               (metrics.csv, confusion_matrix.png, training_curves.png)
```

## Running

```bash
python scripts/create_split.py
python scripts/audit_dataset.py
python -m src.train
python -m src.evaluate
```

All hyperparameters (batch size, epochs, learning rate, seed) live in
`src/config.py` — do not hardcode them elsewhere.
