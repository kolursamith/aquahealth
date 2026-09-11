"""Images -> Labels -> Dataset -> DataLoader.

Owner: Student 2
"""

from pathlib import Path

from torch.utils.data import DataLoader, Dataset

from src.config import BATCH_SIZE, CLASS_NAMES, NUM_WORKERS


class FishDiseaseDataset(Dataset):
    """Reads class-labeled images from a directory laid out as `<split>/<class_name>/*.jpg`."""

    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
        self.samples = self._index_samples()

    def _index_samples(self) -> list[tuple[Path, int]]:
        samples = []
        for class_name in CLASS_NAMES:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            for path in class_dir.glob("*.*"):
                samples.append((path, self.class_to_idx[class_name]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        raise NotImplementedError(
            "Load image, apply transform + preprocess, return (tensor, label)"
        )


def get_dataloader(root: Path, transform=None, shuffle: bool = False) -> DataLoader:
    dataset = FishDiseaseDataset(root, transform=transform)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
    )
