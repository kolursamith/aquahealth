"""Development / final-test partition (Phase 8) and K-fold assignment (Phase 9)
for the multi-dataset experiment, built on the clean manifest.

The assignment rule is the one src/manifest.py::group_aware_stratified_split
already uses for the baseline, generalised from three fixed splits to N parts:
within every stratum, groups are shuffled with the seed, handed out largest
first, each to the part that is furthest below its largest-remainder quota.
The unit of assignment is the clean manifest's `group_id` (exact/near
duplicates and, where the dataset states it, the same specimen), so no group
can straddle two parts. The stratum is (unified class, majority source dataset
of the group), so every class and every contributing source is represented in
each part where its group counts allow.

    data/audit/split_v3/development.csv, final_test.csv    SPLIT_COLUMNS
    data/audit/split_v3/split_manifest.sha256              digest of both files

(split_v3 / cv_v3 = dataset configuration v3, four sources, MatsyaDx-BD excluded;
the v2 partition is archived under data/audit/archive/v2_mendeley/.)

`final_test.csv` is frozen once written: the builder refuses to overwrite it.
"""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable

from src.config import SEED
from src.dataset_cleaning import CleanRow
from src.manifest import CANONICAL_CLASSES, file_sha256, largest_remainder_quota

SPLIT_DIR_NAME = "split_v3"  # dataset configuration v3 (four sources)
DEVELOPMENT = "development"
FINAL_TEST = "final_test"
DEFAULT_TEST_RATIO = (
    0.20  # the only ratio named for the new experiment (team method note, "for example")
)


@dataclass
class SplitRow:
    image_id: str
    source_dataset: str
    filepath: str
    original_path: str
    original_filename: str
    original_class: str
    unified_class: str
    label: int  # canonical index, src/manifest.py::CANONICAL_CLASSES
    group_id: str
    specimen_id: str
    species: str
    stratum: str  # "<unified class>|<majority source of the group>"
    split: str  # development | final_test


SPLIT_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(SplitRow))


# --- assignment ------------------------------------------------------------------------


def assign_groups(
    items: list[tuple[str, str, str]],
    *,
    ratios: tuple[float, ...],
    seed: int = SEED,
) -> dict[str, int]:
    """items = (item_id, stratum, group_id) -> {item_id: part index}.

    Per stratum: groups shuffled with `seed`, sorted largest first, each handed
    to the part with the largest relative deficit (ties: lowest index) — the
    baseline's rule (src/manifest.py) for len(ratios) parts.
    """
    if abs(sum(ratios) - 1.0) > 1e-9 or min(ratios) < 0:
        raise ValueError("ratios must be non-negative and sum to 1")
    n = len(ratios)
    by_stratum: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    group_stratum: dict[str, str] = {}
    for item_id, stratum, group_id in items:
        if group_stratum.setdefault(group_id, stratum) != stratum:
            raise ValueError(f"group {group_id} appears in two strata")
        by_stratum[stratum][group_id].append(item_id)
    rng = random.Random(seed)
    assignment: dict[str, int] = {}
    for stratum in sorted(by_stratum):
        groups = [sorted(ids) for _, ids in sorted(by_stratum[stratum].items())]
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        total = sum(len(g) for g in groups)
        quota = largest_remainder_quota(total, ratios)
        filled = [0] * n
        for group in groups:
            deficit = [(quota[i] - filled[i]) / max(quota[i], 1) for i in range(n)]
            target = max(range(n), key=lambda i: (deficit[i], -i))
            filled[target] += len(group)
            for item_id in group:
                assignment[item_id] = target
    return assignment


def strata(rows: Iterable[CleanRow]) -> dict[str, str]:
    """image_id -> stratum: the image's class plus the majority source of its group."""
    rows = list(rows)
    source_votes: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        source_votes[r.group_id][r.source_dataset] += 1
    majority = {
        g: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for g, c in source_votes.items()
    }
    return {r.image_id: f"{r.unified_class}|{majority[r.group_id]}" for r in rows}


def build_dev_test_split(
    rows: list[CleanRow], *, test_ratio: float = DEFAULT_TEST_RATIO, seed: int = SEED
) -> list[SplitRow]:
    """Two-way group-aware stratified partition of the *included* clean rows."""
    if not 0 < test_ratio < 1:
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")
    included = [r for r in rows if r.included]
    if not included:
        raise ValueError("clean manifest has no included rows")
    stratum_of = strata(included)
    parts = assign_groups(
        [(r.image_id, stratum_of[r.image_id], r.group_id) for r in included],
        ratios=(1 - test_ratio, test_ratio),
        seed=seed,
    )
    names = (DEVELOPMENT, FINAL_TEST)
    out = [
        SplitRow(
            image_id=r.image_id,
            source_dataset=r.source_dataset,
            filepath=r.filepath,
            original_path=r.original_path,
            original_filename=r.original_filename,
            original_class=r.original_class,
            unified_class=r.unified_class,
            label=CANONICAL_CLASSES.index(r.unified_class),
            group_id=r.group_id,
            specimen_id=r.specimen_id,
            species=r.species,
            stratum=stratum_of[r.image_id],
            split=names[parts[r.image_id]],
        )
        for r in included
    ]
    out.sort(key=lambda s: (s.split, s.label, s.filepath))
    validate_split(out, rows)
    return out


def validate_split(split: list[SplitRow], clean: list[CleanRow] | None = None) -> None:
    """Invariants: unique ids, both parts non-empty, every class in both parts,
    no group in two parts, and (with the clean manifest) exactly the included rows."""
    ids = [s.image_id for s in split]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate image_id in split")
    by_part: dict[str, set[str]] = defaultdict(set)
    group_parts: dict[str, set[str]] = defaultdict(set)
    for s in split:
        if s.split not in (DEVELOPMENT, FINAL_TEST):
            raise ValueError(f"unknown split {s.split!r}")
        by_part[s.split].add(s.unified_class)
        group_parts[s.group_id].add(s.split)
        if CANONICAL_CLASSES[s.label] != s.unified_class:
            raise ValueError(f"{s.image_id}: label {s.label} != {s.unified_class}")
    if set(by_part) != {DEVELOPMENT, FINAL_TEST}:
        raise ValueError("both partitions must be non-empty")
    if by_part[DEVELOPMENT] != by_part[FINAL_TEST]:
        raise ValueError("every class must appear in both partitions")
    crossing = [g for g, p in group_parts.items() if len(p) > 1]
    if crossing:
        raise ValueError(f"groups straddle development/final_test: {crossing[:5]}")
    if clean is not None:
        expected = {r.image_id for r in clean if r.included}
        if set(ids) != expected:
            raise ValueError("split rows are not exactly the included clean rows")
        excluded = {r.image_id for r in clean if not r.included}
        if excluded & set(ids):
            raise ValueError("an excluded clean row entered the split")


# --- I/O -----------------------------------------------------------------------------------


def write_split(split: list[SplitRow], directory: Path) -> str:
    """development.csv + final_test.csv + one .sha256 file over both; returns the digest line."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in (DEVELOPMENT, FINAL_TEST):
        with (directory / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(SPLIT_COLUMNS))
            writer.writeheader()
            for s in split:
                if s.split == name:
                    writer.writerow(asdict(s))
    lines = [f"{file_sha256(directory / f'{n}.csv')}  {n}.csv" for n in (DEVELOPMENT, FINAL_TEST)]
    (directory / "split_manifest.sha256").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def read_split(directory: Path) -> list[SplitRow]:
    directory = Path(directory)
    out: list[SplitRow] = []
    for name in (DEVELOPMENT, FINAL_TEST):
        with (directory / f"{name}.csv").open(newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != SPLIT_COLUMNS:
                raise ValueError(f"{name}.csv columns {reader.fieldnames} != {list(SPLIT_COLUMNS)}")
            for raw in reader:
                typed: dict[str, Any] = dict(raw)
                typed["label"] = int(raw["label"])
                if typed["split"] != name:
                    raise ValueError(f"{name}.csv contains a row marked {typed['split']!r}")
                out.append(SplitRow(**typed))
    return out


def verify_split_digest(directory: Path) -> None:
    """Raise if development.csv / final_test.csv no longer match split_manifest.sha256."""
    directory = Path(directory)
    for line in (directory / "split_manifest.sha256").read_text().splitlines():
        digest, name = line.split()
        actual = file_sha256(directory / name)
        if actual != digest:
            raise ValueError(f"{name} has been modified: {actual[:12]} != recorded {digest[:12]}")


def read_development(directory: Path) -> list[SplitRow]:
    """The development rows only — the reader every later phase must use so the
    frozen test file is never opened by training code."""
    directory = Path(directory)
    verify_split_digest(directory)
    with (directory / f"{DEVELOPMENT}.csv").open(newline="") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            typed: dict[str, Any] = dict(raw)
            typed["label"] = int(raw["label"])
            rows.append(SplitRow(**typed))
    if any(r.split != DEVELOPMENT for r in rows):
        raise ValueError("development.csv contains non-development rows")
    return rows


# --- report --------------------------------------------------------------------------------


def split_summary(
    split: list[SplitRow], clean: list[CleanRow], *, test_ratio: float, seed: int, clean_digest: str
) -> dict[str, Any]:
    dev = [s for s in split if s.split == DEVELOPMENT]
    test = [s for s in split if s.split == FINAL_TEST]
    excluded = Counter(r.exclusion_reason for r in clean if not r.included)

    def per(key: str, rows: list[SplitRow]) -> dict[str, int]:
        return dict(sorted(Counter(getattr(s, key) for s in rows).items()))

    groups_dev = {s.group_id for s in dev}
    groups_test = {s.group_id for s in test}
    return {
        "strategy": "group-aware stratified assignment (src/split_v2.py::assign_groups, the "
        "baseline rule of src/manifest.py generalised): unit = clean-manifest group_id "
        "(exact/near duplicates ∪ same specimen), stratum = (unified class, majority source "
        "of the group), largest-remainder quotas, largest groups first to the most deficient part",
        "seed": seed,
        "test_ratio": test_ratio,
        "ratio_provenance": "not fixed by any project document; the team method note gives "
        "80/20 development/test as an example (see data/audit/split_policy_proposal.md)",
        "clean_manifest_sha256": clean_digest,
        "clean_total": len(clean),
        "clean_included": len(split),
        "excluded_from_clean_manifest": dict(excluded),
        "development": {
            "images": len(dev),
            "groups": len(groups_dev),
            "per_class": per("unified_class", dev),
            "per_source": per("source_dataset", dev),
        },
        "final_test": {
            "images": len(test),
            "groups": len(groups_test),
            "per_class": per("unified_class", test),
            "per_source": per("source_dataset", test),
            "share": round(len(test) / len(split), 4),
        },
        "per_class_share_in_test": {
            c: round(per("unified_class", test).get(c, 0) / n, 3)
            for c, n in per("unified_class", split).items()
        },
        "group_constraints": {
            "groups_straddling_partitions": len(groups_dev & groups_test),
            "largest_group_in_test": max((Counter(s.group_id for s in test).values()), default=0),
            "specimen_groups_in_test": len({s.group_id for s in test if s.specimen_id}),
        },
    }


# --- K-fold cross-validation on the development partition (Phase 9) ---------------------

CV_DIR_NAME = "cv_v3"  # dataset configuration v3 (four sources)
DEFAULT_FOLDS = 10


@dataclass
class FoldRow:
    image_id: str
    source_dataset: str
    filepath: str
    original_path: str
    original_class: str
    unified_class: str
    label: int
    group_id: str
    specimen_id: str
    stratum: str
    fold: int  # 1..K: the fold in which this image is the VALIDATION sample


FOLD_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(FoldRow))


def assign_folds(
    development: list[SplitRow], *, n_folds: int = DEFAULT_FOLDS, seed: int = SEED
) -> list[FoldRow]:
    """Every development image gets exactly one validation fold; whole groups go
    to one fold; strata (class | majority source) are spread evenly over folds."""
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if any(s.split != DEVELOPMENT for s in development):
        raise ValueError("assign_folds accepts development rows only")
    parts = assign_groups(
        [(s.image_id, s.stratum, s.group_id) for s in development],
        ratios=tuple(1.0 / n_folds for _ in range(n_folds)),
        seed=seed,
    )
    rows = [
        FoldRow(
            image_id=s.image_id,
            source_dataset=s.source_dataset,
            filepath=s.filepath,
            original_path=s.original_path,
            original_class=s.original_class,
            unified_class=s.unified_class,
            label=s.label,
            group_id=s.group_id,
            specimen_id=s.specimen_id,
            stratum=s.stratum,
            fold=parts[s.image_id] + 1,
        )
        for s in development
    ]
    rows.sort(key=lambda r: (r.fold, r.label, r.filepath))
    validate_folds(rows, n_folds)
    return rows


def validate_folds(rows: list[FoldRow], n_folds: int) -> None:
    ids = [r.image_id for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate image_id in folds")
    folds = {r.fold for r in rows}
    if folds != set(range(1, n_folds + 1)):
        raise ValueError(f"folds present {sorted(folds)} != 1..{n_folds}")
    group_folds: dict[str, set[int]] = defaultdict(set)
    for r in rows:
        group_folds[r.group_id].add(r.fold)
        if CANONICAL_CLASSES[r.label] != r.unified_class:
            raise ValueError(f"{r.image_id}: label mismatch")
    crossing = [g for g, f in group_folds.items() if len(f) > 1]
    if crossing:
        raise ValueError(f"groups straddle folds: {crossing[:5]}")


def fold_members(rows: list[FoldRow], fold: int) -> tuple[list[FoldRow], list[FoldRow]]:
    """(train, validation) for one fold: validation = rows of that fold, train = the rest."""
    validation = [r for r in rows if r.fold == fold]
    train = [r for r in rows if r.fold != fold]
    return train, validation


def write_folds(rows: list[FoldRow], directory: Path, n_folds: int) -> str:
    """folds.csv (image -> fold) plus fold_XX_train.csv / fold_XX_validation.csv,
    all digest-listed in folds.sha256; returns the digest text."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = ["folds.csv"]
    with (directory / "folds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FOLD_COLUMNS))
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)
    for fold in range(1, n_folds + 1):
        train, validation = fold_members(rows, fold)
        for part, members in (("train", train), ("validation", validation)):
            name = f"fold_{fold:02d}_{part}.csv"
            with (directory / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(FOLD_COLUMNS))
                writer.writeheader()
                writer.writerows(asdict(r) for r in members)
            names.append(name)
    lines = [f"{file_sha256(directory / n)}  {n}" for n in names]
    (directory / "folds.sha256").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def read_folds(directory: Path) -> list[FoldRow]:
    directory = Path(directory)
    for line in (directory / "folds.sha256").read_text().splitlines():
        digest, name = line.split()
        if file_sha256(directory / name) != digest:
            raise ValueError(f"{name} has been modified since the folds were written")
    with (directory / "folds.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FOLD_COLUMNS:
            raise ValueError("folds.csv has unexpected columns")
        rows = []
        for raw in reader:
            typed: dict[str, Any] = dict(raw)
            typed["label"] = int(raw["label"])
            typed["fold"] = int(raw["fold"])
            rows.append(FoldRow(**typed))
    return rows


def cv_report_rows(rows: list[FoldRow], n_folds: int, seed: int) -> list[dict[str, Any]]:
    """One row per fold for data/audit/cv_report.csv."""
    out = []
    for fold in range(1, n_folds + 1):
        train, validation = fold_members(rows, fold)
        classes = Counter(r.unified_class for r in validation)
        sources = Counter(r.source_dataset for r in validation)
        groups = {r.group_id for r in validation}
        out.append(
            {
                "fold": fold,
                "train_count": len(train),
                "validation_count": len(validation),
                "class_distribution": "; ".join(f"{c}={n}" for c, n in sorted(classes.items())),
                "source_distribution": "; ".join(f"{s}={n}" for s, n in sorted(sources.items())),
                "group_information": (
                    f"groups={len(groups)}; specimen_groups="
                    f"{len({r.group_id for r in validation if r.specimen_id})}; "
                    f"largest_group={max(Counter(r.group_id for r in validation).values())}; "
                    f"groups_shared_with_train={len(groups & {r.group_id for r in train})}"
                ),
                "seed": seed,
            }
        )
    return out
