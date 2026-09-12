"""Layer 5 — dataset discovery, decoding and loading (`src.dataset`).

All data is synthetic (see conftest.py). Each class is rendered in a known
colour, so label correctness is checked against pixel content.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision.transforms import v2

from src.dataset import (
    IMAGE_EXTENSIONS,
    ImageDecodeError,
    ImageFolderDataset,
    build_dataloader,
    discover_classes,
    index_samples,
    is_image_file,
    load_image,
)
from tests.conftest import class_colour

CLASSES = ("alpha", "beta", "gamma")


def _mean_colour(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(round(v)) for v in tensor.float().mean(dim=(1, 2)).tolist())


# --- discovery ---


def test_discover_classes_is_sorted_and_ignores_hidden_and_files(make_image_folder):
    root = make_image_folder(("zeta", "alpha", "Mid"))
    (root / ".hidden").mkdir()
    (root / "notes.txt").write_text("not a class")
    assert discover_classes(root) == ["Mid", "alpha", "zeta"]


def test_discover_classes_rejects_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_classes(tmp_path / "nope")


def test_discover_classes_rejects_root_without_class_dirs(tmp_path):
    (tmp_path / "loose.jpg").write_bytes(b"")
    with pytest.raises(ValueError, match="no class directories"):
        discover_classes(tmp_path)


# --- indexing ---


@pytest.mark.parametrize(
    "name,expected",
    [
        ("a.jpg", True),
        ("a.JPEG", True),
        ("a.png", True),
        ("a.webp", True),
        ("a.bmp", True),
        ("a.txt", False),
        (".gitkeep", False),
        (".DS_Store", False),
        ("a.tif", False),
        (".hidden.jpg", False),
    ],
)
def test_is_image_file(tmp_path, name, expected):
    path = tmp_path / name
    path.write_bytes(b"")
    assert is_image_file(path) is expected


def test_index_ignores_non_image_and_hidden_files(make_image_folder):
    root = make_image_folder(CLASSES, per_class=2)
    (root / "alpha" / ".gitkeep").write_bytes(b"")
    (root / "alpha" / "README.md").write_text("x")
    (root / "beta" / ".DS_Store").write_bytes(b"\x00")
    samples = index_samples(root, list(CLASSES))
    assert len(samples) == 6
    assert all(s.path.suffix.lower() in IMAGE_EXTENSIONS for s in samples)


def test_index_order_is_deterministic(make_image_folder):
    root = make_image_folder(CLASSES)
    first = [s.path for s in index_samples(root, list(CLASSES))]
    second = [s.path for s in index_samples(root, list(CLASSES))]
    assert first == second == sorted(first)


def test_index_rejects_unknown_class_directory(make_image_folder):
    root = make_image_folder(CLASSES)
    with pytest.raises(ValueError, match="unexpected class directories.*gamma"):
        index_samples(root, ["alpha", "beta"])


def test_index_tolerates_declared_class_with_no_directory(make_image_folder):
    root = make_image_folder(("alpha", "beta"))
    samples = index_samples(root, ["alpha", "beta", "gamma"])
    assert {s.label for s in samples} == {0, 1}


# --- decoding ---


def test_load_image_returns_rgb_for_grayscale_and_rgba(tmp_path):
    grey = tmp_path / "grey.png"
    rgba = tmp_path / "rgba.png"
    Image.new("L", (8, 8), 77).save(grey)
    Image.new("RGBA", (8, 8), (1, 2, 3, 4)).save(rgba)
    assert load_image(grey).mode == "RGB"
    assert load_image(rgba).mode == "RGB"


def test_load_image_names_the_file_on_corrupt_input(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"definitely not a jpeg")
    with pytest.raises(ImageDecodeError, match="bad.jpg"):
        load_image(bad)


# --- dataset ---


def test_dataset_length_and_class_counts(make_image_folder):
    dataset = ImageFolderDataset(make_image_folder(CLASSES, per_class=4))
    assert len(dataset) == 12
    assert dataset.class_names == list(CLASSES)
    assert dataset.class_to_idx == {"alpha": 0, "beta": 1, "gamma": 2}
    assert dataset.class_counts() == {"alpha": 4, "beta": 4, "gamma": 4}


def test_dataset_rejects_empty_root(make_image_folder):
    root = make_image_folder(CLASSES, per_class=0)
    with pytest.raises(ValueError, match="no image files"):
        ImageFolderDataset(root)


def test_item_is_uint8_chw_tensor_with_int_label(make_image_folder):
    dataset = ImageFolderDataset(make_image_folder(CLASSES, size=(48, 40)))
    tensor, label = dataset[0]
    assert tensor.shape == (3, 40, 48)
    assert tensor.dtype == torch.uint8
    assert isinstance(label, int)


def test_every_item_content_matches_its_label(make_image_folder):
    dataset = ImageFolderDataset(make_image_folder(CLASSES, per_class=3))
    for index in range(len(dataset)):
        tensor, label = dataset[index]
        expected = class_colour(label)
        observed = _mean_colour(tensor)
        assert all(abs(o - e) <= 3 for o, e in zip(observed, expected)), (index, observed, expected)


def test_targets_match_iteration_labels(make_image_folder):
    dataset = ImageFolderDataset(make_image_folder(CLASSES, per_class=2))
    assert dataset.targets == [dataset[i][1] for i in range(len(dataset))]
    assert dataset.targets == [0, 0, 1, 1, 2, 2]


def test_transform_is_applied(make_image_folder):
    resize = v2.Compose([v2.PILToTensor(), v2.Resize((16, 24), antialias=True)])
    dataset = ImageFolderDataset(make_image_folder(CLASSES), transform=resize)
    tensor, _ = dataset[0]
    assert tensor.shape == (3, 16, 24)


def test_split_shares_label_mapping_via_class_names(make_image_folder):
    train = ImageFolderDataset(make_image_folder(CLASSES, name="train"))
    val = ImageFolderDataset(
        make_image_folder(("alpha", "gamma"), name="val"), class_names=train.class_names
    )
    assert val.class_to_idx == train.class_to_idx
    assert set(val.targets) == {0, 2}
    assert val.class_counts()["beta"] == 0


def test_split_with_unknown_class_is_rejected(make_image_folder):
    train = ImageFolderDataset(make_image_folder(CLASSES, name="train"))
    with pytest.raises(ValueError, match="unexpected class directories"):
        ImageFolderDataset(
            make_image_folder(("alpha", "delta"), name="val"), class_names=train.class_names
        )


def test_corrupt_file_fails_loudly_at_access_not_at_index(make_image_folder):
    root = make_image_folder(CLASSES, per_class=1)
    (root / "alpha" / "000_bad.jpg").write_bytes(b"garbage")
    dataset = ImageFolderDataset(root)
    assert len(dataset) == 4
    bad_index = next(i for i, s in enumerate(dataset.samples) if s.path.name == "000_bad.jpg")
    with pytest.raises(ImageDecodeError, match="000_bad.jpg"):
        dataset[bad_index]


# --- dataloader ---


def _uniform(make_image_folder, per_class=4):
    to_float = v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True)])
    return ImageFolderDataset(make_image_folder(CLASSES, per_class=per_class), transform=to_float)


def test_batches_have_expected_shapes_and_dtypes(make_image_folder):
    loader = build_dataloader(_uniform(make_image_folder), batch_size=5, num_workers=0)
    batches = list(loader)
    assert [b[0].shape[0] for b in batches] == [5, 5, 2]
    images, labels = batches[0]
    assert images.shape == (5, 3, 40, 48) and images.dtype == torch.float32
    assert labels.shape == (5,) and labels.dtype == torch.int64


def test_drop_last_discards_partial_batch(make_image_folder):
    loader = build_dataloader(
        _uniform(make_image_folder), batch_size=5, num_workers=0, drop_last=True
    )
    assert [b[0].shape[0] for b in list(loader)] == [5, 5]


def test_unshuffled_loader_preserves_index_order(make_image_folder):
    dataset = _uniform(make_image_folder)
    loader = build_dataloader(dataset, batch_size=4, shuffle=False, num_workers=0)
    labels = torch.cat([b[1] for b in loader]).tolist()
    assert labels == dataset.targets


def test_shuffle_is_reproducible_under_seed_and_varies_across_seeds(make_image_folder):
    dataset = _uniform(make_image_folder, per_class=8)

    def order(seed):
        loader = build_dataloader(dataset, batch_size=6, shuffle=True, seed=seed, num_workers=0)
        return torch.cat([b[1] for b in loader]).tolist()

    assert order(1) == order(1)
    assert order(1) != order(2)
    assert sorted(order(1)) == sorted(dataset.targets)


def test_worker_processes_yield_the_same_data_as_main_process(make_image_folder):
    dataset = _uniform(make_image_folder)
    single = build_dataloader(dataset, batch_size=4, num_workers=0)
    multi = build_dataloader(dataset, batch_size=4, num_workers=2)
    for (a_img, a_lbl), (b_img, b_lbl) in zip(single, multi, strict=True):
        assert torch.equal(a_img, b_img)
        assert torch.equal(a_lbl, b_lbl)


def test_mixed_sizes_without_resize_cannot_be_batched(make_image_folder):
    root = make_image_folder(("alpha",), per_class=1, size=(20, 20), name="mixed")
    Image.new("RGB", (30, 30), class_colour(0)).save(root / "alpha" / "001.png")
    loader = build_dataloader(ImageFolderDataset(root), batch_size=2, num_workers=0)
    with pytest.raises(RuntimeError, match="stack expects each tensor to be equal size"):
        next(iter(loader))


def test_default_loader_settings_come_from_config():
    import inspect

    from src.config import BATCH_SIZE, NUM_WORKERS

    defaults = inspect.signature(build_dataloader).parameters
    assert defaults["batch_size"].default == BATCH_SIZE
    assert defaults["num_workers"].default == NUM_WORKERS


def test_image_folder_root_path_is_recorded(make_image_folder):
    root = make_image_folder(CLASSES)
    assert ImageFolderDataset(root).root == Path(root)
