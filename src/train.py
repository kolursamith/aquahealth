"""Dataset -> DataLoader -> Model -> Loss -> Backprop -> Optimizer -> Validation -> Best checkpoint.

Owner: Student 1
"""

from src.config import (
    CHECKPOINT_PATH,
    LEARNING_RATE,
    NUM_EPOCHS,
    SEED,
    TRAIN_DIR,
    VAL_DIR,
    WEIGHT_DECAY,
)
from src.utils import get_logger, set_seed

logger = get_logger(__name__)


def train() -> None:
    set_seed(SEED)
    raise NotImplementedError(
        "Build train/val DataLoaders, instantiate model, run the training loop, "
        f"and save the best checkpoint to {CHECKPOINT_PATH}"
    )


if __name__ == "__main__":
    train()
