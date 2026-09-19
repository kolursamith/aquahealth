"""Clean manifest for the multi-dataset experiment: which images of the master
manifest enter the clean training corpus, and why the others do not.

Input is `data/audit/master_dataset.csv` (src/multi_dataset.py: provenance,
label mapping, SHA-256, dHash, duplicate groups). Output is one row per master
image with `included` and, when excluded, an `exclusion_reason`:

    CORRUPT               undecodable file
    LABEL_UNRESOLVED      mapping_status UNRESOLVED (no unified class yet)
    LABEL_EXCLUDED        mapping_status EXCLUDED
    LABEL_CONFLICT_GROUP  member of a duplicate group whose mapped members carry
                          different unified labels (project policy: the whole
                          group leaves the corpus; no label is picked)
    EXACT_DUPLICATE       byte-identical to `representative_image_id`
    NEAR_DUPLICATE        only with collapse_near_duplicates=True: same dHash as
                          the representative (NOT the project's default policy)

The project's established policy (scripts/build_split_manifest.py) is applied
unchanged: exact duplicates collapse to the first eligible path, near-duplicate
groups are kept together as one `group_id` so a later split can keep them in
one partition, and label-conflicting groups are excluded. A `group_id` also
joins photographs of the same specimen where the dataset states a specimen id
(MatsyaDx-BD), because same-fish leakage is a duplicate-leakage case too.

No raw file is touched; this module only writes a CSV.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable

from src.multi_dataset import ImageRecord

MAPPED_STATUSES = ("EXACT_MATCH", "SUPPORTED_MAPPING")
EXCLUSION_REASONS = (
    "CORRUPT",
    "LABEL_UNRESOLVED",
    "LABEL_EXCLUDED",
    "LABEL_CONFLICT_GROUP",
    "EXACT_DUPLICATE",
    "NEAR_DUPLICATE",
)
CLEAN_MANIFEST_NAME = "clean_manifest.csv"


@dataclass
class CleanRow:
    image_id: str
    source_dataset: str
    filepath: str
    original_path: str
    original_filename: str
    original_class: str
    original_split: str
    unified_class: str
    mapping_status: str
    species: str
    specimen_id: str
    width: int
    height: int
    extension: str
    sha256: str
    dhash: str
    exact_dup_group: str
    near_dup_group: str
    group_id: str  # leakage group: near-duplicate cluster ∪ same specimen
    group_size: int
    included: bool
    exclusion_reason: str
    representative_image_id: str  # the kept copy, for excluded duplicates


CLEAN_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(CleanRow))


# --- leakage groups (union-find) ---------------------------------------------------


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic root: the lexically smaller key
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def leakage_groups(records: Iterable[ImageRecord]) -> dict[str, str]:
    """image_id -> group_id. Images share a group when they share a near-duplicate
    cluster (which contains every exact-duplicate cluster) or, where the dataset
    states one, a specimen id. Singletons get their own image_id as group_id."""
    uf = _UnionFind()
    for r in records:
        uf.find(r.image_id)
        if r.near_dup_group:
            uf.union(r.image_id, f"near:{r.near_dup_group}")
        if r.specimen_id:
            uf.union(r.image_id, f"specimen:{r.source_dataset}:{r.specimen_id}")
    groups: dict[str, str] = {}
    members: dict[str, list[str]] = defaultdict(list)
    for r in records:
        members[uf.find(r.image_id)].append(r.image_id)
    for ids in members.values():
        gid = min(ids)  # stable, human-traceable: the smallest member id
        for i in ids:
            groups[i] = gid
    return groups


# --- clean manifest --------------------------------------------------------------------


def build_clean_manifest(
    records: list[ImageRecord], *, collapse_near_duplicates: bool = False
) -> list[CleanRow]:
    """Apply the established exclusion policy as rows; never touches files."""
    records = sorted(records, key=lambda r: r.filepath)
    group_of = leakage_groups(records)
    group_size = Counter(group_of.values())

    # label-conflict groups: near-duplicate clusters whose *mapped* members disagree
    by_near: dict[str, set[str]] = defaultdict(set)
    for r in records:
        if r.status == "ok" and r.near_dup_group and r.mapping_status in MAPPED_STATUSES:
            by_near[r.near_dup_group].add(r.unified_class)
    conflict_groups = {g for g, classes in by_near.items() if len(classes) > 1}

    def label_reason(r: ImageRecord) -> str:
        if r.status != "ok":
            return "CORRUPT"
        if r.mapping_status == "UNRESOLVED":
            return "LABEL_UNRESOLVED"
        if r.mapping_status == "EXCLUDED":
            return "LABEL_EXCLUDED"
        if r.mapping_status not in MAPPED_STATUSES:
            raise ValueError(f"{r.image_id}: unknown mapping_status {r.mapping_status!r}")
        if r.near_dup_group in conflict_groups:
            return "LABEL_CONFLICT_GROUP"
        return ""

    # representatives: first eligible path per SHA-256 (exact) and, optionally, per dHash
    exact_rep: dict[str, str] = {}
    near_rep: dict[str, str] = {}
    for r in records:  # sorted by filepath = the established "first path" rule
        if label_reason(r):
            continue
        exact_rep.setdefault(r.sha256, r.image_id)
        if r.near_dup_group:
            near_rep.setdefault(r.near_dup_group, r.image_id)

    rows: list[CleanRow] = []
    for r in records:
        reason = label_reason(r)
        representative = ""
        if not reason and exact_rep[r.sha256] != r.image_id:
            reason, representative = "EXACT_DUPLICATE", exact_rep[r.sha256]
        elif (
            not reason
            and collapse_near_duplicates
            and r.near_dup_group
            and near_rep[r.near_dup_group] != r.image_id
        ):
            reason, representative = "NEAR_DUPLICATE", near_rep[r.near_dup_group]
        rows.append(
            CleanRow(
                image_id=r.image_id,
                source_dataset=r.source_dataset,
                filepath=r.filepath,
                original_path=r.original_path,
                original_filename=r.original_filename,
                original_class=r.original_class,
                original_split=r.original_split,
                unified_class=r.unified_class,
                mapping_status=r.mapping_status,
                species=r.species,
                specimen_id=r.specimen_id,
                width=r.width,
                height=r.height,
                extension=r.extension,
                sha256=r.sha256,
                dhash=r.dhash,
                exact_dup_group=r.exact_dup_group,
                near_dup_group=r.near_dup_group,
                group_id=group_of[r.image_id],
                group_size=group_size[group_of[r.image_id]],
                included=not reason,
                exclusion_reason=reason,
                representative_image_id=representative,
            )
        )
    validate_clean_manifest(rows)
    return rows


def validate_clean_manifest(rows: list[CleanRow]) -> None:
    """Invariants a later split relies on. Raises on violation."""
    ids = {r.image_id for r in rows}
    if len(ids) != len(rows):
        raise ValueError("duplicate image_id in clean manifest")
    included = [r for r in rows if r.included]
    if any(r.exclusion_reason for r in included):
        raise ValueError("included row carries an exclusion reason")
    if any(not r.exclusion_reason for r in rows if not r.included):
        raise ValueError("excluded row without a reason")
    if any(r.exclusion_reason not in EXCLUSION_REASONS for r in rows if not r.included):
        raise ValueError("unknown exclusion reason")
    if any(not r.unified_class for r in included):
        raise ValueError("included row without a unified class")
    seen_sha: set[str] = set()
    for r in included:
        if r.sha256 in seen_sha:
            raise ValueError(f"two included rows share SHA-256 {r.sha256[:12]}")
        seen_sha.add(r.sha256)
    for r in rows:
        if r.representative_image_id and (
            r.representative_image_id not in ids or r.representative_image_id == r.image_id
        ):
            raise ValueError(f"{r.image_id}: bad representative {r.representative_image_id}")
    classes_per_group: dict[str, set[str]] = defaultdict(set)
    for r in included:
        classes_per_group[r.group_id].add(r.unified_class)
    mixed = [g for g, c in classes_per_group.items() if len(c) > 1]
    if mixed:
        raise ValueError(f"included leakage groups with several labels: {mixed[:5]}")


# --- I/O ------------------------------------------------------------------------------


def write_clean_manifest(rows: Iterable[CleanRow], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CLEAN_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_clean_manifest(path: Path) -> list[CleanRow]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CLEAN_COLUMNS:
            raise ValueError(f"{path} columns {reader.fieldnames} != {list(CLEAN_COLUMNS)}")
        rows = []
        for raw in reader:
            typed: dict[str, Any] = dict(raw)
            for key in ("width", "height", "group_size"):
                typed[key] = int(raw[key])
            typed["included"] = raw["included"] == "True"
            rows.append(CleanRow(**typed))
        return rows


# --- summary ----------------------------------------------------------------------------


def dedup_summary(
    records: list[ImageRecord], rows: list[CleanRow], *, collapse_near_duplicates: bool
) -> dict[str, Any]:
    """Machine-readable before/after numbers for the de-duplication report."""
    ok = [r for r in records if r.status == "ok"]
    by_sha: dict[str, set[str]] = defaultdict(set)
    by_near: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in ok:
        by_sha[r.sha256].add(r.source_dataset)
        if r.near_dup_group:
            by_near[r.near_dup_group].append(r)
    exact_groups = {r.exact_dup_group for r in ok if r.exact_dup_group}
    exact_cross = sum(1 for sha, s in by_sha.items() if len(s) > 1)
    near_groups = {g for g, ms in by_near.items() if len({m.sha256 for m in ms}) > 1}
    near_cross = sum(1 for g in near_groups if len({m.source_dataset for m in by_near[g]}) > 1)
    conflict = {
        g
        for g, ms in by_near.items()
        if len({m.unified_class for m in ms if m.mapping_status in MAPPED_STATUSES}) > 1
    }
    row_of = {r.image_id: r for r in rows}
    included = [r for r in rows if r.included]
    per_dataset_before = Counter(r.source_dataset for r in records)
    per_dataset_after = Counter(r.source_dataset for r in included)
    per_class_after = Counter(r.unified_class for r in included)
    per_class_before = Counter(
        r.unified_class for r in records if r.mapping_status in MAPPED_STATUSES
    )
    reasons = Counter(r.exclusion_reason for r in rows if not r.included)
    reasons_by_dataset = {
        d: dict(
            Counter(r.exclusion_reason for r in rows if not r.included and r.source_dataset == d)
        )
        for d in per_dataset_before
    }
    group_sizes = Counter(r.group_id for r in included)
    return {
        "policy": {
            "exact_duplicates": "collapse to the first eligible path (sorted filepath)",
            "near_duplicates": (
                "collapse to the first eligible path"
                if collapse_near_duplicates
                else "kept, joined under one group_id (established project policy)"
            ),
            "label_conflicts": "whole group excluded; no label chosen",
            "same_specimen": "joined under one group_id where the dataset states a specimen id",
        },
        "raw": {
            "total": len(records),
            "per_dataset": dict(per_dataset_before),
            "corrupt": len(records) - len(ok),
            "mapped_label_per_class": dict(per_class_before),
        },
        "duplicates": {
            "exact_groups": len(exact_groups),
            "exact_groups_cross_dataset": exact_cross,
            "near_groups_distinct_bytes": len(near_groups),
            "near_groups_cross_dataset": near_cross,
            "conflicting_label_groups": len(conflict),
            "conflicting_label_group_images": sum(len(by_near[g]) for g in conflict),
        },
        "excluded": {
            "total": len(rows) - len(included),
            "by_reason": dict(reasons),
            "by_dataset_and_reason": reasons_by_dataset,
        },
        "clean": {
            "total": len(included),
            "per_dataset": dict(per_dataset_after),
            "per_class": dict(per_class_after),
            "per_dataset_and_class": {
                d: dict(Counter(r.unified_class for r in included if r.source_dataset == d))
                for d in per_dataset_after
            },
            "leakage_groups": len(group_sizes),
            "leakage_groups_multi_image": sum(1 for n in group_sizes.values() if n > 1),
            "largest_leakage_group": max(group_sizes.values(), default=0),
        },
        "unresolved": {
            "images_awaiting_label_decision": sum(
                1 for r in rows if r.exclusion_reason == "LABEL_UNRESOLVED"
            ),
            "per_dataset_and_class": dict(
                Counter(
                    f"{r.source_dataset}/{r.original_class}"
                    for r in rows
                    if r.exclusion_reason in ("LABEL_UNRESOLVED", "LABEL_EXCLUDED")
                )
            ),
            "conflict_group_members": [
                {
                    "image_id": r.image_id,
                    "source_dataset": r.source_dataset,
                    "original_path": r.original_path,
                    "unified_class": r.unified_class or r.original_class,
                    "near_dup_group": r.near_dup_group,
                }
                for r in records
                if r.near_dup_group in conflict
                and row_of[r.image_id].exclusion_reason == "LABEL_CONFLICT_GROUP"
            ],
        },
    }
