"""Layer 9 — staged fine-tuning machinery (`src.model` freeze API, `src.train` param
groups, `src.finetune` schedule). Fixture data only; nothing here is a disease result."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.dataset import ImageFolderDataset, build_dataloader
from src.finetune import (
    ALL_BLOCKS,
    DEFAULT_SCHEDULE,
    Stage,
    apply_stage,
    run_schedule,
    stage_train_config,
    validate_schedule,
)
from src.model import (
    backbone_block_count,
    build_classifier,
    set_trainable_blocks,
    summarize,
    trainable_block_count,
)
from src.preprocessing import PreprocessConfig, build_eval_transform
from src.train import (
    BEST_CHECKPOINT_NAME,
    LAST_CHECKPOINT_NAME,
    TrainConfig,
    build_optimizer,
    load_checkpoint,
    model_spec,
    set_train_mode,
    train_one_epoch,
)
from src.utils import set_seed
from tests.conftest import write_image_folder

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ("alpha", "beta", "gamma", "delta")
SMALL = PreprocessConfig(image_size=64, resize_size=72)
CPU = torch.device("cpu")
TOTAL_BLOCKS = 9


@pytest.fixture(scope="module")
def roots(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("ft")
    return {
        "train": write_image_folder(base / "train", CLASSES, per_class=6, size=(80, 60), seed=0),
        "val": write_image_folder(base / "val", CLASSES, per_class=2, size=(80, 60), seed=1),
    }


def _loader(root: Path, class_names=None, batch_size: int = 8, shuffle: bool = False):
    dataset = ImageFolderDataset(
        root, transform=build_eval_transform(SMALL), class_names=class_names
    )
    return dataset, build_dataloader(
        dataset, batch_size=batch_size, shuffle=shuffle, seed=0, num_workers=0
    )


def _model(seed: int = 0):
    set_seed(seed)
    return build_classifier(len(CLASSES))


def _block_params(model, index):
    return sum(p.numel() for p in model.features[index].parameters())


# --- freeze / unfreeze API ---


def test_backbone_has_nine_feature_blocks():
    assert backbone_block_count(build_classifier(4, pretrained=False)) == TOTAL_BLOCKS


@pytest.mark.parametrize("last_n", [0, 1, 3, 9])
def test_set_trainable_blocks_unfreezes_exactly_the_trailing_suffix(last_n):
    model = build_classifier(4, pretrained=False)
    set_trainable_blocks(model, last_n)
    flags = [all(p.requires_grad for p in block.parameters()) for block in model.features]
    assert flags == [False] * (TOTAL_BLOCKS - last_n) + [True] * last_n
    assert all(p.requires_grad for p in model.classifier.parameters())
    assert trainable_block_count(model) == last_n
    expected_trainable = sum(
        _block_params(model, i) for i in range(TOTAL_BLOCKS - last_n, TOTAL_BLOCKS)
    )
    expected_trainable += sum(p.numel() for p in model.classifier.parameters())
    assert summarize(model).trainable_parameters == expected_trainable


@pytest.mark.parametrize("bad", [-1, 10])
def test_set_trainable_blocks_rejects_out_of_range(bad):
    with pytest.raises(ValueError, match="last_n_blocks"):
        set_trainable_blocks(build_classifier(4, pretrained=False), bad)


def test_set_trainable_blocks_is_reversible_in_both_directions():
    model = build_classifier(4, pretrained=False)
    set_trainable_blocks(model, 9)
    set_trainable_blocks(model, 2)
    assert trainable_block_count(model) == 2
    set_trainable_blocks(model, 0)
    assert trainable_block_count(model) == 0
    set_trainable_blocks(model, 9)
    assert trainable_block_count(model) == 9


def test_trainable_block_count_rejects_non_suffix_pattern():
    model = build_classifier(4, pretrained=False)
    set_trainable_blocks(model, 0)
    for p in model.features[3].parameters():
        p.requires_grad_(True)
    with pytest.raises(ValueError, match="trailing suffix"):
        trainable_block_count(model)


def test_set_trainable_blocks_re_enables_a_manually_frozen_head():
    model = build_classifier(4, pretrained=False)
    for p in model.classifier.parameters():
        p.requires_grad_(False)
    set_trainable_blocks(model, 0)
    assert all(p.requires_grad for p in model.classifier.parameters())


# --- model mode per block ---


def test_set_train_mode_is_block_granular():
    model = build_classifier(4, pretrained=False)
    set_trainable_blocks(model, 3)
    set_train_mode(model)
    assert [block.training for block in model.features] == [False] * 6 + [True] * 3
    assert model.classifier.training


def test_frozen_blocks_keep_batchnorm_stats_while_unfrozen_blocks_update(roots):
    _, loader = _loader(roots["train"])
    model = _model()
    set_trainable_blocks(model, 3)
    running = {k: v.clone() for k, v in model.state_dict().items() if "running_" in k}
    train_one_epoch(model, loader, build_optimizer(model, TrainConfig()), CPU, epoch=1)
    after = model.state_dict()
    frozen_keys = [k for k in running if int(k.split(".")[1]) < 6]
    unfrozen_keys = [k for k in running if int(k.split(".")[1]) >= 6]
    assert frozen_keys and unfrozen_keys
    assert all(torch.equal(running[k], after[k]) for k in frozen_keys)
    assert any(not torch.equal(running[k], after[k]) for k in unfrozen_keys)


# --- optimiser parameter groups ---


def test_optimizer_groups_head_and_backbone_with_their_own_lrs():
    model = build_classifier(4, pretrained=False)
    set_trainable_blocks(model, 2)
    config = TrainConfig(learning_rate=1e-3, backbone_learning_rate=1e-5, weight_decay=0.01)
    optimizer = build_optimizer(model, config)
    groups = {g["name"]: g for g in optimizer.param_groups}
    assert set(groups) == {"head", "backbone"}
    assert groups["head"]["lr"] == 1e-3 and groups["backbone"]["lr"] == 1e-5
    assert all(g["weight_decay"] == 0.01 for g in optimizer.param_groups)
    assert sum(p.numel() for p in groups["head"]["params"]) == 1280 * 4 + 4
    assert sum(p.numel() for p in groups["backbone"]["params"]) == _block_params(
        model, 7
    ) + _block_params(model, 8)


def test_optimizer_has_no_backbone_group_when_backbone_is_frozen():
    model = build_classifier(4, pretrained=False, freeze_backbone=True)
    optimizer = build_optimizer(model, TrainConfig(backbone_learning_rate=1e-5))
    assert [g["name"] for g in optimizer.param_groups] == ["head"]


def test_backbone_lr_defaults_to_head_lr():
    model = build_classifier(4, pretrained=False)
    optimizer = build_optimizer(model, TrainConfig(learning_rate=2e-4))
    assert {g["lr"] for g in optimizer.param_groups} == {2e-4}


def test_train_config_rejects_non_positive_backbone_lr():
    with pytest.raises(ValueError, match="backbone_learning_rate"):
        TrainConfig(backbone_learning_rate=0.0)


def test_smaller_backbone_lr_moves_backbone_weights_less(roots):
    _, loader = _loader(roots["train"])
    deltas = {}
    for backbone_lr in (1e-3, 1e-5):
        model = _model()
        set_trainable_blocks(model, 1)
        before = model.features[8][0].weight.clone()
        config = TrainConfig(learning_rate=1e-3, backbone_learning_rate=backbone_lr)
        train_one_epoch(model, loader, build_optimizer(model, config), CPU, epoch=1)
        deltas[backbone_lr] = (model.features[8][0].weight - before).abs().mean().item()
    assert deltas[1e-3] > 10 * deltas[1e-5]


# --- gradient routing under partial unfreezing ---


def test_gradients_and_updates_follow_the_freeze_pattern(roots):
    _, loader = _loader(roots["train"])
    model = _model()
    set_trainable_blocks(model, 3)
    before = {k: v.clone() for k, v in model.features.state_dict().items()}
    train_one_epoch(
        model, loader, build_optimizer(model, TrainConfig(learning_rate=1e-3)), CPU, epoch=1
    )

    for index, block in enumerate(model.features):
        params = list(block.named_parameters())
        if index < 6:
            assert all(p.grad is None for _, p in params), f"frozen block {index} got gradients"
            assert all(
                torch.equal(before[f"{index}.{name}"], p) for name, p in params
            ), f"frozen block {index} changed"
        else:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in params)
            assert any(not torch.equal(before[f"{index}.{name}"], p) for name, p in params)
    assert all(p.grad is not None for p in model.classifier.parameters())


# --- checkpoint compatibility ---


def test_checkpoint_records_and_restores_partial_freezing(roots, tmp_path):
    _, loader = _loader(roots["train"])
    model = _model()
    set_trainable_blocks(model, 2)
    from src.train import save_checkpoint

    spec = model_spec(model)
    assert spec["trainable_blocks"] == 2 and spec["freeze_backbone"] is False
    path = save_checkpoint(
        tmp_path / "p.pt",
        model=model,
        optimizer=build_optimizer(model, TrainConfig()),
        epoch=1,
        history=[],
        class_names=list(CLASSES),
        preprocess=SMALL,
        train_config=TrainConfig(),
        stage="partial",
    )
    checkpoint = load_checkpoint(path)
    assert checkpoint.stage == "partial"
    rebuilt = checkpoint.build_model()
    assert trainable_block_count(rebuilt) == 2
    assert summarize(rebuilt).trainable_parameters == summarize(model).trainable_parameters


# --- schedule ---


@pytest.mark.parametrize(
    "kwargs",
    [{"name": ""}, {"epochs": 0}, {"trainable_blocks": -2}],
)
def test_stage_rejects_invalid_values(kwargs):
    base = {"name": "s", "epochs": 1, "trainable_blocks": 0, "learning_rate": 1e-3}
    with pytest.raises(ValueError):
        Stage(**{**base, **kwargs})


def test_default_schedule_is_head_then_partial_then_full():
    validate_schedule(DEFAULT_SCHEDULE)
    assert [s.name for s in DEFAULT_SCHEDULE] == ["head", "partial", "full"]
    assert DEFAULT_SCHEDULE[0].trainable_blocks == 0
    assert 0 < DEFAULT_SCHEDULE[1].trainable_blocks < TOTAL_BLOCKS
    assert DEFAULT_SCHEDULE[2].trainable_blocks == ALL_BLOCKS
    assert DEFAULT_SCHEDULE[1].backbone_learning_rate < DEFAULT_SCHEDULE[1].learning_rate
    assert DEFAULT_SCHEDULE[2].backbone_learning_rate < DEFAULT_SCHEDULE[2].learning_rate


def test_validate_schedule_rejects_empty_and_duplicate_names():
    with pytest.raises(ValueError, match="at least one"):
        validate_schedule([])
    with pytest.raises(ValueError, match="unique"):
        validate_schedule([Stage("a", 1, 0, 1e-3), Stage("a", 1, 0, 1e-3)])


def test_apply_stage_resolves_all_blocks_and_stage_config_carries_lrs():
    model = build_classifier(4, pretrained=False)
    assert apply_stage(model, Stage("full", 1, ALL_BLOCKS, 1e-4, 1e-5)) == TOTAL_BLOCKS
    assert trainable_block_count(model) == TOTAL_BLOCKS
    config = stage_train_config(
        Stage("partial", 2, 3, 3e-4, 3e-5), seed=7, selection_metric="accuracy"
    )
    assert (config.epochs, config.learning_rate, config.backbone_learning_rate) == (2, 3e-4, 3e-5)
    assert config.seed == 7 and config.selection_metric == "accuracy"


TINY_SCHEDULE = (
    Stage("head", epochs=2, trainable_blocks=0, learning_rate=1e-3),
    Stage("partial", epochs=1, trainable_blocks=2, learning_rate=3e-4, backbone_learning_rate=3e-5),
    Stage(
        "full",
        epochs=1,
        trainable_blocks=ALL_BLOCKS,
        learning_rate=1e-4,
        backbone_learning_rate=1e-5,
    ),
)


def test_run_schedule_chains_stages_and_carries_best_weights_forward(roots, tmp_path):
    train_dataset, train_loader = _loader(roots["train"], shuffle=True)
    _, val_loader = _loader(roots["val"], train_dataset.class_names)
    model = _model()
    schedule = run_schedule(
        model,
        train_loader,
        TINY_SCHEDULE,
        class_names=train_dataset.class_names,
        preprocess=SMALL,
        device=CPU,
        checkpoint_dir=tmp_path / "ft",
        val_loader=val_loader,
        seed=0,
    )

    assert [s.stage.name for s in schedule.stages] == ["head", "partial", "full"]
    assert [s.trainable_blocks for s in schedule.stages] == [0, 2, TOTAL_BLOCKS]
    assert [len(s.result.history) for s in schedule.stages] == [2, 1, 1]
    for stage_result in schedule.stages:
        directory = stage_result.checkpoint_dir
        assert (directory / LAST_CHECKPOINT_NAME).exists() and (
            directory / BEST_CHECKPOINT_NAME
        ).exists()
        last = load_checkpoint(directory / LAST_CHECKPOINT_NAME)
        assert last.stage == stage_result.stage.name
        assert last.model_spec["trainable_blocks"] == stage_result.trainable_blocks
    assert schedule.final_checkpoint == tmp_path / "ft" / "full" / BEST_CHECKPOINT_NAME
    assert trainable_block_count(model) == TOTAL_BLOCKS


def test_stage_starts_from_previous_stage_best_checkpoint(roots, tmp_path):
    """The weights entering stage 2 must equal stage 1's best.pt, not its last.pt."""
    train_dataset, train_loader = _loader(roots["train"], shuffle=True)
    _, val_loader = _loader(roots["val"], train_dataset.class_names)

    captured = {}
    from src import finetune as finetune_module

    original_fit = finetune_module.fit

    def spying_fit(model, *args, stage=None, **kwargs):
        captured[stage] = {k: v.clone() for k, v in model.state_dict().items()}
        return original_fit(model, *args, stage=stage, **kwargs)

    finetune_module.fit = spying_fit
    try:
        run_schedule(
            _model(),
            train_loader,
            TINY_SCHEDULE[:2],
            class_names=train_dataset.class_names,
            preprocess=SMALL,
            device=CPU,
            checkpoint_dir=tmp_path / "ft",
            val_loader=val_loader,
            seed=0,
        )
    finally:
        finetune_module.fit = original_fit

    head_best = load_checkpoint(tmp_path / "ft" / "head" / BEST_CHECKPOINT_NAME).model_state
    for key, value in head_best.items():
        assert torch.equal(value, captured["partial"][key]), key


def test_run_schedule_resumes_from_a_stage_checkpoint(roots, tmp_path):
    train_dataset, train_loader = _loader(roots["train"], shuffle=True)
    _, val_loader = _loader(roots["val"], train_dataset.class_names)
    common = dict(
        class_names=train_dataset.class_names, preprocess=SMALL, device=CPU, val_loader=val_loader
    )
    full = run_schedule(
        _model(), train_loader, TINY_SCHEDULE, checkpoint_dir=tmp_path / "a", seed=0, **common
    )
    partial_only = run_schedule(
        _model(),
        _loader(roots["train"], shuffle=True)[1],
        TINY_SCHEDULE[:1],
        checkpoint_dir=tmp_path / "b",
        seed=0,
        **common,
    )
    assert partial_only.stages[-1].stage.name == "head"

    checkpoint = load_checkpoint(tmp_path / "b" / "head" / LAST_CHECKPOINT_NAME)
    resumed = run_schedule(
        checkpoint.build_model(),
        _loader(roots["train"], shuffle=True)[1],
        TINY_SCHEDULE,
        checkpoint_dir=tmp_path / "b",
        seed=0,
        resume_from=checkpoint,
        **common,
    )
    assert [s.stage.name for s in resumed.stages] == ["head", "partial", "full"]
    assert [len(s.result.history) for s in resumed.stages] == [2, 1, 1]
    assert [s.result.history[-1].validation.loss for s in resumed.stages] == [
        s.result.history[-1].validation.loss for s in full.stages
    ]


def test_run_schedule_rejects_checkpoint_from_unknown_stage(roots, tmp_path):
    train_dataset, train_loader = _loader(roots["train"])
    from src.train import save_checkpoint

    model = _model()
    path = save_checkpoint(
        tmp_path / "x.pt",
        model=model,
        optimizer=build_optimizer(model, TrainConfig()),
        epoch=1,
        history=[],
        class_names=train_dataset.class_names,
        preprocess=SMALL,
        train_config=TrainConfig(),
        stage="mystery",
    )
    with pytest.raises(ValueError, match="not in the schedule"):
        run_schedule(
            model,
            train_loader,
            TINY_SCHEDULE,
            class_names=train_dataset.class_names,
            preprocess=SMALL,
            device=CPU,
            checkpoint_dir=tmp_path / "ft",
            resume_from=load_checkpoint(path),
        )


def test_cli_runs_a_tiny_schedule(roots, tmp_path):
    ck = tmp_path / "cli"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.finetune",
            "--train-root",
            str(roots["train"]),
            "--val-root",
            str(roots["val"]),
            "--checkpoint-dir",
            str(ck),
            "--head-epochs",
            "1",
            "--partial-epochs",
            "1",
            "--full-epochs",
            "1",
            "--partial-blocks",
            "2",
            "--image-size",
            "64",
            "--batch-size",
            "8",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    for stage in ("head", "partial", "full"):
        assert f"stage {stage}:" in result.stderr
        assert (ck / stage / BEST_CHECKPOINT_NAME).exists()
    assert "schedule complete" in result.stderr
    assert load_checkpoint(ck / "full" / LAST_CHECKPOINT_NAME).model_spec["trainable_blocks"] == 9
