# data/raw/ — delivered datasets for the multi-dataset experiment

Nothing under this directory is copied or committed. Each entry is a symlink
into the delivery drop, created by

```bash
python scripts/link_raw_datasets.py --source /path/to/delivery/Dataset
```

(`--source` can also come from `$AQUAHEALTH_DATASET_SOURCE`). The keys and the
delivered folder they point at are defined in `src/multi_dataset.py::DATASET_SOURCES`:

| key | delivered folder (inside the drop) | layout |
|---|---|---|
| `current_freshwater/` | `train_split/`, `test_split` (or `test_split&validation`), `test.csv` — three links inside a real directory | `train_split/<class>/`, flat `test_split/` + `test.csv` |
| `kaptai/` | `Fresh Water Fish Dataset/` | `<class>/` |
| `roboflow/` | `Fish Disease.v1i.folder/` | `<train|valid|test>/<class>/` |
| `mendeley/` | `MatsyaDx-BD An image dataset of freshwater fish di/MatsyaDx-BD/` | `<condition>/<species>/<specimen>/` + `metadata.csv` |
| `paper_dataset/` | `SalmonScan … Aquaculture System/SalmonScan/` | `<FreshFish|InfectedFish>/` |

The delivered train/valid/test folders are recorded as `original_split`
provenance only; they are **not** a split for the new experiment. The
existing baseline's `data/original/aquahealth` link and `data/split_manifest.csv`
are untouched.
