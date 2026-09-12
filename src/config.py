"""Single source of truth for dataset, model, and training configuration."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# EXPECTED class list from the project brief — NOT yet verified against the
# real dataset. scripts/audit_dataset.py must confirm the actual class
# directories, counts and names before any training run; this list is then
# corrected to match the data, never the other way round. The classifier
# head (src/model.build_classifier) takes num_classes as an explicit argument
# and does not read this constant.
CLASS_NAMES = [
    "Aeromoniasis",
    "Bacterial_Gill_Disease",
    "Columnaris",
    "Dropsy",
    "Fin_Rot",
    "Healthy",
    "Saprolegniasis",
    "White_Spot",
]
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 224
SEED = 42

DATA_DIR = ROOT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
ORIGINAL_DIR = DATA_DIR / "original"

MODEL_NAME = "efficientnet_b0"
MODELS_DIR = ROOT_DIR / "models"
CHECKPOINT_PATH = MODELS_DIR / "final_model.pth"

RESULTS_DIR = ROOT_DIR / "results"

BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

RISK_THRESHOLD_MODERATE = 0.50
RISK_THRESHOLD_HIGH = 0.80
