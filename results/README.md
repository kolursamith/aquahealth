# results/

Experimental output only — written by `python -m src.evaluate`, never
entered by hand. Everything here is git-ignored except this file.

```
results/<run-name>/
├── metrics.json                      summary, per-class metrics, confusion (raw + normalised),
│                                     confidence analysis (ECE, reliability bins), checkpoint provenance
├── predictions.csv                   one row per image: path, target, predicted, confidence, correct, p_<class>…
├── misclassified.csv                 the incorrect rows of predictions.csv
├── confusion_matrix.csv
├── confusion_matrix_normalized.csv   row-normalised (per true class)
└── classification_report.txt
```

The held-out `data/test/` split is evaluated **once**, at the very end, and
is never used to choose a checkpoint; use `data/val/` for that.
