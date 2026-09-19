#!/usr/bin/env python
"""Phases 1-5 of the multi-dataset pipeline: inventory, provenance manifest,
label mapping, hashing and cross-dataset duplicate analysis.

    python scripts/build_master_dataset.py            # data/raw -> data/audit

Writes (never into data/raw/):

    data/audit/dataset_inventory.csv / .md   one row per (dataset, split, class)
    data/audit/non_image_files.csv           every non-image file seen (nothing is skipped silently)
    data/audit/label_mapping.csv             src/label_harmonization.LABEL_MAPPING
    data/audit/master_dataset.csv            one row per image: provenance, label mapping,
                                             dimensions, SHA-256, dHash, duplicate groups
    data/audit/duplicate_report.csv / .md    every duplicate pair with provenance

Hashing reuses src.manifest.file_sha256 / perceptual_dhash and the grouping
helper scripts/build_split_manifest.assign_duplicate_groups, so "exact" means
byte-identical (SHA-256) and "near" means identical 8x8 dHash — the same
definitions as results/data_audit.csv. No file is deleted, moved or relabelled;
the established duplicate policy (exact duplicates collapse to the first path,
label-conflicting near-duplicate groups are excluded, other near-duplicate
groups stay together) is recorded per image as `dedup_action` for the split
phase to act on.

No train/validation/test assignment is made here.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from build_split_manifest import assign_duplicate_groups  # noqa: E402

from src.config import DATA_DIR, ROOT_DIR  # noqa: E402
from src.label_harmonization import (  # noqa: E402
    LABEL_MAPPING,
    lookup,
    validate_mapping_table,
    write_label_mapping_csv,
)
from src.multi_dataset import (  # noqa: E402
    AUDIT_DIR_NAME,
    DATASET_SOURCES,
    MASTER_MANIFEST_NAME,
    ImageRecord,
    NonImageFile,
    discover_all,
    inventory_rows,
    probe,
    raw_root,
    read_master_manifest,
    reuse_probe,
    write_inventory_csv,
    write_master_manifest,
)
from src.utils import get_logger  # noqa: E402

logger = get_logger("master_dataset")

MAPPED = ("EXACT_MATCH", "SUPPORTED_MAPPING")
DUPLICATE_COLUMNS = (
    "image_a",
    "image_b",
    "source_dataset_a",
    "source_dataset_b",
    "original_path_a",
    "original_path_b",
    "original_class_a",
    "original_class_b",
    "unified_class_a",
    "unified_class_b",
    "hash_type",
    "hash_value_or_comparison",
    "similarity_or_distance",
    "duplicate_status",
    "cross_dataset",
    "label_relation",
    "recommended_action",
    "reason",
)
INFORMATIONAL_MAX_DISTANCE = 8

REASON_CONFLICT = "different unified labels (policy: label-conflict groups leave the split)"
REASON_UNRESOLVED = "a label is unresolved; the match is evidence for the mapping decision"
REASON_EXACT_KEEP = "byte-identical (policy: exact duplicates collapse to the first path)"
REASON_NEAR_KEEP = (
    "identical dHash, different bytes (policy: near-duplicate groups stay in one split)"
)


# --- label mapping ---------------------------------------------------------------


def apply_label_mapping(records: list[ImageRecord]) -> None:
    """Fill unified_class / mapping_status from the mapping table; every observed
    (dataset, class) must have a row — an unmapped class is an error, not a guess."""
    validate_mapping_table()
    missing = sorted(
        {
            (r.source_dataset, r.original_class)
            for r in records
            if lookup(r.source_dataset, r.original_class) is None
        }
    )
    if missing:
        raise ValueError(
            "classes without a row in src/label_harmonization.LABEL_MAPPING: "
            + "; ".join(f"{s}/{c}" for s, c in missing)
        )
    for r in records:
        mapping = lookup(r.source_dataset, r.original_class)
        assert mapping is not None
        r.unified_class = mapping.unified_class
        r.mapping_status = mapping.mapping_status


# --- duplicate analysis -----------------------------------------------------------------


def flag_conflicts_and_actions(records: list[ImageRecord]) -> int:
    """Apply the established policy as annotations (dedup_action), never as file
    operations. Returns the number of near-duplicate groups with conflicting
    unified labels among their mapped members."""
    by_group: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        if r.status == "ok" and r.near_dup_group:
            by_group[r.near_dup_group].append(r)
    conflicts = {
        g
        for g, members in by_group.items()
        if len({m.unified_class for m in members if m.mapping_status in MAPPED}) > 1
    }
    representative: dict[str, str] = {}
    for r in sorted(records, key=lambda r: r.filepath):
        if r.status == "ok":
            representative.setdefault(r.sha256, r.filepath)
    for r in records:
        if r.status != "ok":
            r.dedup_action = "CORRUPT"
        elif r.near_dup_group in conflicts:
            r.dedup_action = (
                "EXCLUDE_LABEL_CONFLICT" if r.mapping_status in MAPPED else "REVIEW_LABEL_CONFLICT"
            )
        elif r.exact_dup_group and representative[r.sha256] != r.filepath:
            r.dedup_action = "DROP_EXACT_DUPLICATE"
        elif r.near_dup_group and len({m.sha256 for m in by_group[r.near_dup_group]}) > 1:
            r.dedup_action = "KEEP_GROUPED"
        else:
            r.dedup_action = "KEEP"
    return len(conflicts)


def label_relation(a: ImageRecord, b: ImageRecord) -> str:
    if a.mapping_status in MAPPED and b.mapping_status in MAPPED:
        return "SAME_LABEL" if a.unified_class == b.unified_class else "LABEL_CONFLICT"
    return "UNRESOLVED_LABEL"


def duplicate_pairs(records: list[ImageRecord]) -> list[dict[str, Any]]:
    """One row per pair inside an exact group (SHA-256) or a near group (dHash,
    distinct bytes). Pairs are ordered by filepath so image_a is the copy the
    established policy would keep."""
    rows: list[dict[str, Any]] = []
    ok = [r for r in records if r.status == "ok"]
    by_sha: dict[str, list[ImageRecord]] = defaultdict(list)
    by_near: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in ok:
        if r.exact_dup_group:
            by_sha[r.sha256].append(r)
        if r.near_dup_group:
            by_near[r.near_dup_group].append(r)

    def row(
        a: ImageRecord,
        b: ImageRecord,
        hash_type: str,
        value: str,
        distance: int,
        status: str,
        action: str,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "image_a": a.image_id,
            "image_b": b.image_id,
            "source_dataset_a": a.source_dataset,
            "source_dataset_b": b.source_dataset,
            "original_path_a": a.original_path,
            "original_path_b": b.original_path,
            "original_class_a": a.original_class,
            "original_class_b": b.original_class,
            "unified_class_a": a.unified_class,
            "unified_class_b": b.unified_class,
            "hash_type": hash_type,
            "hash_value_or_comparison": value,
            "similarity_or_distance": distance,
            "duplicate_status": status,
            "cross_dataset": a.source_dataset != b.source_dataset,
            "label_relation": label_relation(a, b),
            "recommended_action": action,
            "reason": reason,
        }

    for sha, members in sorted(by_sha.items()):
        members.sort(key=lambda r: r.filepath)
        for a, b in combinations(members, 2):
            relation = label_relation(a, b)
            if relation == "LABEL_CONFLICT":
                action, reason = (
                    "EXCLUDE_BOTH",
                    "byte-identical, " + REASON_CONFLICT,
                )
            elif relation == "UNRESOLVED_LABEL":
                action, reason = (
                    "REVIEW",
                    "byte-identical, " + REASON_UNRESOLVED,
                )
            else:
                action, reason = (
                    "DROP_B_KEEP_A",
                    REASON_EXACT_KEEP,
                )
            rows.append(row(a, b, "sha256", sha, 0, "EXACT_DUPLICATE", action, reason))

    for group, members in sorted(by_near.items()):
        distinct = {m.sha256 for m in members}
        if len(distinct) < 2:
            continue
        members.sort(key=lambda r: r.filepath)
        for a, b in combinations(members, 2):
            if a.sha256 == b.sha256:
                continue  # already listed as an exact pair
            relation = label_relation(a, b)
            if relation == "LABEL_CONFLICT":
                action, reason = (
                    "EXCLUDE_BOTH",
                    "identical dHash, " + REASON_CONFLICT,
                )
            elif relation == "UNRESOLVED_LABEL":
                action, reason = (
                    "REVIEW",
                    "identical dHash, " + REASON_UNRESOLVED,
                )
            else:
                action, reason = (
                    "KEEP_SAME_SPLIT",
                    REASON_NEAR_KEEP,
                )
            rows.append(row(a, b, "dhash", a.dhash, 0, "NEAR_DUPLICATE", action, reason))
    return rows


def hamming_histogram(records: list[ImageRecord], max_distance: int) -> dict[str, Counter]:
    """Informational only: how many pairs sit at each small dHash distance, for
    all pairs and for cross-dataset pairs. No threshold is applied or implied."""
    ok = [r for r in records if r.status == "ok" and r.dhash]
    hashes = np.array([int(r.dhash, 16) for r in ok], dtype=np.uint64)
    sources = np.array([r.source_dataset for r in ok])
    table = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)
    all_pairs: Counter = Counter()
    cross: Counter = Counter()
    block = 512
    for start in range(0, len(hashes), block):
        rows = hashes[start : start + block]
        x = rows[:, None] ^ hashes[None, :]
        d = (
            table[x & np.uint64(0xFFFF)]
            + table[(x >> np.uint64(16)) & np.uint64(0xFFFF)]
            + table[(x >> np.uint64(32)) & np.uint64(0xFFFF)]
            + table[(x >> np.uint64(48)) & np.uint64(0xFFFF)]
        ).astype(np.int16)
        # keep i < j only
        i_idx = np.arange(start, start + len(rows))[:, None]
        j_idx = np.arange(len(hashes))[None, :]
        mask = (j_idx > i_idx) & (d <= max_distance)
        di = d[mask]
        for value, count in zip(*np.unique(di, return_counts=True)):
            all_pairs[int(value)] += int(count)
        cross_mask = mask & (sources[start : start + len(rows)][:, None] != sources[None, :])
        for value, count in zip(*np.unique(d[cross_mask], return_counts=True)):
            cross[int(value)] += int(count)
    return {"all": all_pairs, "cross_dataset": cross}


# --- reports --------------------------------------------------------------------------


def write_csv(rows: list[dict[str, Any]], columns: tuple[str, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def write_inventory_report(
    path: Path,
    records: list[ImageRecord],
    inventory: list[dict[str, Any]],
    others: list[NonImageFile],
    raw_dir: Path,
) -> None:
    lines = [
        "# Multi-dataset inventory (Phase 1-2)",
        "",
        "Generated by `scripts/build_master_dataset.py` from the raw datasets referenced under "
        f"`{raw_dir}` (symlinks into the delivery drop; nothing under them is modified). "
        "Counts are of files actually discovered; provenance per image is in "
        "`master_dataset.csv`.",
        "",
        "## Datasets",
        "",
        "| key | name | resolved path | images | corrupt | classes | splits | documentation |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for source in DATASET_SOURCES:
        mine = [r for r in records if r.source_dataset == source.key]
        local = raw_dir / source.key
        if local.is_dir() and not local.is_symlink():  # a directory of links (current_freshwater)
            resolved = "; ".join(
                f"{p.name} -> {p.resolve()}" for p in sorted(local.iterdir()) if p.is_symlink()
            ) or str(local)
        else:
            resolved = str(local.resolve() if local.exists() else local)
        classes = sorted({r.original_class for r in mine})
        splits = sorted({r.original_split for r in mine if r.original_split})
        lines.append(
            f"| {source.key} | {source.name} | `{resolved}` | {len(mine)} | "
            f"{sum(1 for r in mine if r.status == 'corrupt')} | {len(classes)} | "
            f"{', '.join(splits) or 'none (class folders)'} | "
            f"{', '.join(source.documentation) or 'none'} |"
        )
    lines += [
        "",
        "## Per class",
        "",
        "| dataset | split | original class | images | extensions | species | specimens | "
        "dimensions (top 4) | corrupt |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in inventory:
        lines.append(
            f"| {row['dataset']} | {row['split_or_structure']} | {row['original_class']} | "
            f"{row['image_count']} | {row['extensions']} | {row['species'] or '—'} | "
            f"{row['specimens'] or '—'} | {row['dimensions']} | {row['corrupt']} |"
        )
    lines += ["", "## Species information", ""]
    for source in DATASET_SOURCES:
        mine = [r for r in records if r.source_dataset == source.key]
        species = Counter(r.species for r in mine if r.species)
        if species:
            lines.append(
                f"- {source.key}: "
                + ", ".join(f"{s} ({n})" for s, n in species.most_common())
                + " — from the dataset's own folder/metadata"
            )
        else:
            lines.append(
                f"- {source.key}: not stated by the dataset files "
                "(not established by the available evidence)"
            )
    lines += [
        "",
        "## Non-image files seen (recorded, not silently skipped)",
        "",
        "| dataset | path | kind | bytes |",
        "|---|---|---|---|",
    ]
    for o in others:
        lines.append(f"| {o.source_dataset} | `{o.original_path}` | {o.kind} | {o.size} |")
    lines += ["", "## Notes per dataset", ""]
    for source in DATASET_SOURCES:
        lines.append(f"- **{source.key}** — {source.notes}")
    path.write_text("\n".join(lines) + "\n")


def write_duplicate_report(
    path: Path,
    records: list[ImageRecord],
    pairs: list[dict[str, Any]],
    exact_groups: int,
    near_groups: int,
    conflict_groups: int,
    histogram: dict[str, Counter],
) -> None:
    ok = [r for r in records if r.status == "ok"]
    keys = [s.key for s in DATASET_SOURCES]
    exact_pairs = [p for p in pairs if p["duplicate_status"] == "EXACT_DUPLICATE"]
    near_pairs = [p for p in pairs if p["duplicate_status"] == "NEAR_DUPLICATE"]
    matrix: dict[str, Counter] = {"EXACT_DUPLICATE": Counter(), "NEAR_DUPLICATE": Counter()}
    for p in pairs:
        a, b = sorted((p["source_dataset_a"], p["source_dataset_b"]))
        matrix[p["duplicate_status"]][(a, b)] += 1
    actions = Counter(r.dedup_action for r in records)
    lines = [
        "# Duplicate analysis (Phase 5) — exact SHA-256 and perceptual dHash, across all datasets",
        "",
        "Definitions are the repository's established ones (`scripts/build_split_manifest.py`, "
        "`results/data_audit.csv`): **exact** = identical SHA-256; **near** = identical 8x8 dHash "
        "with different bytes. No other similarity threshold is applied. Nothing was deleted; "
        "`dedup_action` in `master_dataset.csv` records what the established policy *would* do.",
        "",
        "## Totals",
        "",
        f"- images inspected: {len(records)} "
        f"({len(ok)} decodable, {len(records) - len(ok)} corrupt)",
        f"- exact-duplicate groups (SHA-256): {exact_groups} — {len(exact_pairs)} pairs, "
        f"{sum(1 for p in exact_pairs if p['cross_dataset'])} of them cross-dataset",
        f"- near-duplicate groups (dHash, distinct bytes): {near_groups} — "
        f"{len(near_pairs)} pairs, "
        f"{sum(1 for p in near_pairs if p['cross_dataset'])} of them cross-dataset",
        "- near-duplicate groups whose mapped members carry conflicting unified labels: "
        f"{conflict_groups}",
        f"- pairs involving an unresolved/excluded label (evidence for Phase 3, no action taken): "
        f"{sum(1 for p in pairs if p['label_relation'] == 'UNRESOLVED_LABEL')}",
        "",
        "## Pairs per dataset pair",
        "",
        "| dataset A | dataset B | exact pairs | near pairs |",
        "|---|---|---|---|",
    ]
    for i, a in enumerate(keys):
        for b in keys[i:]:
            e = matrix["EXACT_DUPLICATE"][(a, b)] if a <= b else matrix["EXACT_DUPLICATE"][(b, a)]
            n = matrix["NEAR_DUPLICATE"][(a, b)] if a <= b else matrix["NEAR_DUPLICATE"][(b, a)]
            lines.append(f"| {a} | {b} | {e} | {n} |")
    lines += [
        "",
        "## Recorded `dedup_action` per image (annotation only)",
        "",
        "| action | images |",
        "|---|---|",
    ]
    for action, n in actions.most_common():
        lines.append(f"| {action} | {n} |")
    lines += [
        "",
        "## Near-duplicate evidence for unresolved classes",
        "",
        "Unresolved/excluded original classes whose images are byte- or dHash-identical to images "
        "of a mapped class. This is evidence for the Phase-3 decision, not a mapping.",
        "",
        "| unresolved (dataset / class) | matches mapped class | pairs |",
        "|---|---|---|",
    ]
    evidence: Counter = Counter()
    for p in pairs:
        if p["label_relation"] != "UNRESOLVED_LABEL":
            continue
        for side, other in (("a", "b"), ("b", "a")):
            if not p[f"unified_class_{side}"] and p[f"unified_class_{other}"]:
                evidence[
                    (
                        f"{p[f'source_dataset_{side}']} / {p[f'original_class_{side}']}",
                        p[f"unified_class_{other}"],
                    )
                ] += 1
    for (unresolved, mapped), n in sorted(evidence.items()):
        lines.append(f"| {unresolved} | {mapped} | {n} |")
    if not evidence:
        lines.append("| (none) | | |")
    lines += [
        "",
        "## Informational: pairs at small dHash Hamming distance "
        f"(≤ {INFORMATIONAL_MAX_DISTANCE} bits)",
        "",
        "Distance 0 is the established near-duplicate definition above. Distances > 0 are listed "
        "only so a reviewer can see how many borderline pairs exist; **no threshold above 0 is "
        "established in this project and none is applied.**",
        "",
        "| distance | all pairs | cross-dataset pairs |",
        "|---|---|---|",
    ]
    for d in range(INFORMATIONAL_MAX_DISTANCE + 1):
        lines.append(
            f"| {d} | {histogram['all'].get(d, 0)} | {histogram['cross_dataset'].get(d, 0)} |"
        )
    lines += [
        "",
        "## Policy applied (as annotation)",
        "",
        "- exact duplicates: keep the first by sorted `filepath` (so a `current_freshwater` copy "
        "wins over any other dataset's copy, then `kaptai`, `mendeley`, `paper_dataset`, "
        "`roboflow`), drop the rest — `DROP_EXACT_DUPLICATE`",
        "- near-duplicate groups whose mapped members disagree on the unified label: "
        "`EXCLUDE_LABEL_CONFLICT` (unmapped members of such a group: `REVIEW_LABEL_CONFLICT`)",
        "- other near-duplicate groups: `KEEP_GROUPED` (must land in one split)",
        "- everything else: `KEEP`; undecodable files: `CORRUPT`",
        "",
        "Files themselves were not touched. Whether a cross-dataset copy should be kept from the "
        "*other* dataset instead (e.g. to favour the higher-resolution original) is a decision "
        "the existing policy does not make and is left open.",
    ]
    path.write_text("\n".join(lines) + "\n")


# --- main ------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--audit-dir", type=Path, default=None, help="default: <data-dir>/audit")
    parser.add_argument("--repo-root", type=Path, default=ROOT_DIR)
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="re-hash every file even if the previous manifest has it",
    )
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args(argv)

    raw_dir = raw_root(args.data_dir)
    audit_dir = args.audit_dir or (Path(args.data_dir) / AUDIT_DIR_NAME)
    audit_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = audit_dir / MASTER_MANIFEST_NAME

    # Phase 1-2: discovery with provenance
    records, others = discover_all(raw_dir, args.repo_root)
    logger.info("discovered %d images and %d non-image files", len(records), len(others))
    for kind, n in Counter(o.kind for o in others).items():
        logger.info("  non-image %s: %d", kind, n)

    # Phase 3: label mapping (table -> csv; applied as annotation)
    write_label_mapping_csv(audit_dir / "label_mapping.csv", LABEL_MAPPING)
    apply_label_mapping(records)
    logger.info("mapping statuses: %s", dict(Counter(r.mapping_status for r in records)))

    # Phase 4-5: probe (dimensions + hashes), restartable from the previous manifest
    previous: dict[str, ImageRecord] = {}
    if manifest_path.exists() and not args.no_reuse:
        previous = {r.filepath: r for r in read_master_manifest(manifest_path)}
        logger.info("reusing probe results from %s (%d rows)", manifest_path, len(previous))
    started = time.perf_counter()
    reused = 0
    for index, record in enumerate(records, 1):
        if reuse_probe(record, previous):
            reused += 1
        else:
            probe(record, args.repo_root)
        if index % args.progress_every == 0:
            logger.info("probed %d/%d (%.0fs)", index, len(records), time.perf_counter() - started)
    corrupt = [r for r in records if r.status == "corrupt"]
    logger.info(
        "probed %d images (%d reused) in %.0fs; corrupt: %d",
        len(records),
        reused,
        time.perf_counter() - started,
        len(corrupt),
    )
    for r in corrupt:
        logger.warning("corrupt: %s", r.filepath)

    # ImageRecord carries the same status/sha256/dhash/*_dup_group attributes as
    # FileRecord, so the existing grouping helper is reused as-is.
    exact_groups, near_groups = assign_duplicate_groups(records)  # type: ignore[arg-type]
    conflict_groups = flag_conflicts_and_actions(records)
    pairs = duplicate_pairs(records)
    histogram = hamming_histogram(records, INFORMATIONAL_MAX_DISTANCE)
    logger.info(
        "exact groups %d, near groups %d, conflicting groups %d, pairs %d",
        exact_groups,
        near_groups,
        conflict_groups,
        len(pairs),
    )

    write_master_manifest(records, manifest_path)
    inventory = inventory_rows(records, raw_dir)
    write_inventory_csv(inventory, audit_dir / "dataset_inventory.csv")
    write_csv(
        [asdict(o) for o in others],
        ("source_dataset", "original_path", "kind", "size"),
        audit_dir / "non_image_files.csv",
    )
    write_inventory_report(audit_dir / "dataset_inventory.md", records, inventory, others, raw_dir)
    write_csv(pairs, DUPLICATE_COLUMNS, audit_dir / "duplicate_report.csv")
    write_duplicate_report(
        audit_dir / "duplicate_report.md",
        records,
        pairs,
        exact_groups,
        near_groups,
        conflict_groups,
        histogram,
    )
    logger.info("wrote %s", audit_dir)
    print((audit_dir / "duplicate_report.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
