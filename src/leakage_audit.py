"""Pre-split leakage audit of the clean manifest (teacher workflow step
"Data Leakage Checks": target leakage, duplicate leakage, suspicious features).

Everything here is counting over manifest columns; no model is trained. The
"shortcut predictability" numbers are in-sample upper bounds: the accuracy a
rule "predict the majority class of this feature value" would reach on the
corpus itself. They quantify how much a non-disease feature (image size, file
type, source dataset) already says about the label; they are not a claim that a
CNN uses that feature. Findings are documented, never silently fixed.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable

from src.dataset_cleaning import CleanRow

# Words in a filename that would reveal the label without looking at pixels.
# Built from the unified class names and the delivered folder names, plus the
# generic health words the deliveries use.
LABEL_TOKENS: dict[str, tuple[str, ...]] = {
    "Bacterial Red Disease": ("bacterial red", "red disease", "redspot", "red spot"),
    "Aeromoniasis": ("aeromoniasis",),
    "Bacterial Gill Disease": ("gill",),
    "EUS Disease": ("eus", "ulcerative"),
    "Saprolegniasis": ("saprolegniasis", "fungal"),
    "Parasitic Disease": ("parasit", "argulus"),
    "White Tail Disease": ("white tail", "tail"),
    "Healthy Fish": ("healthy", "fresh"),
}
GENERIC_HEALTH_TOKENS = ("disease", "infected", "diseased", "sick")
AUGMENTATION_TOKENS = ("aug", "_rot", "flip", "mirror")


def _norm(text: str) -> str:
    return re.sub(r"[_\-\.]+", " ", text.lower())


def filename_label_tokens(row: CleanRow) -> list[str]:
    """Label-revealing tokens present in the file *name* (not the folder path)."""
    name = _norm(row.original_filename)
    found = [t for t in LABEL_TOKENS.get(row.unified_class, ()) if t in name]
    found += [t for t in GENERIC_HEALTH_TOKENS if t in name]
    if _norm(row.original_class) and _norm(row.original_class) in name:
        found.append("<original class name>")
    return found


def target_leakage(rows: list[CleanRow]) -> dict[str, Any]:
    """Where the label is written down outside the pixels."""
    per_dataset: dict[str, dict[str, Any]] = {}
    for dataset in sorted({r.source_dataset for r in rows}):
        mine = [r for r in rows if r.source_dataset == dataset]
        with_tokens = [r for r in mine if filename_label_tokens(r)]
        examples = [r.original_filename for r in with_tokens[:3]]
        per_dataset[dataset] = {
            "images": len(mine),
            "filename_reveals_label": len(with_tokens),
            "filename_reveals_label_pct": round(100 * len(with_tokens) / max(len(mine), 1), 1),
            "label_in_folder_path": all(r.original_class in r.original_path for r in mine),
            "examples": examples,
        }
    return {
        "channels": {
            "folder_path": "label source by construction (folder = class); never read as a feature",
            "filename": "see per_dataset; only the decoded image reaches the model "
            "(src/dataset.py, src/manifest.py load pixels only)",
            "metadata_csv": (
                "mendeley/metadata.csv carries health_condition (= label), fish_category, "
                "specimen_id; used for provenance and grouping only"
                if "mendeley" in per_dataset
                else "no active dataset ships a metadata table (MatsyaDx-BD excluded in v3)"
            ),
            "test_csv": "current_freshwater/test.csv carries the label of the flat test_split; "
            "used only to assign original_class",
        },
        "per_dataset": per_dataset,
        "verdict": "labels exist in paths/filenames/metadata but no pipeline component feeds "
        "them to a model; keep it that way (no filename- or path-derived features)",
    }


def duplicate_leakage(rows: list[CleanRow]) -> dict[str, Any]:
    """Can a future split separate copies of the same picture / the same fish?
    Only if it ignores `group_id`; here we measure what `group_id` has to hold."""
    included = [r for r in rows if r.included]
    groups: dict[str, list[CleanRow]] = defaultdict(list)
    for r in included:
        groups[r.group_id].append(r)
    multi = {g: m for g, m in groups.items() if len(m) > 1}
    cross_dataset = {g for g, m in multi.items() if len({x.source_dataset for x in m}) > 1}
    cross_delivered_split = {
        g
        for g, m in multi.items()
        if len({(x.source_dataset, x.original_split) for x in m if x.original_split}) > 1
    }
    specimen_groups = {g for g, m in multi.items() if any(x.specimen_id for x in m)}
    sizes = Counter(len(m) for m in multi.values())
    return {
        "included_images": len(included),
        "leakage_groups": len(groups),
        "multi_image_groups": len(multi),
        "images_in_multi_image_groups": sum(len(m) for m in multi.values()),
        "groups_spanning_datasets": len(cross_dataset),
        "groups_spanning_delivered_splits": len(cross_delivered_split),
        "specimen_groups": len(specimen_groups),
        "largest_group": max((len(m) for m in multi.values()), default=1),
        "group_size_histogram": {str(k): v for k, v in sorted(sizes.items())},
        "included_group_with_mixed_labels": sum(
            1 for m in groups.values() if len({x.unified_class for x in m}) > 1
        ),
        "excluded_exact_copies": sum(1 for r in rows if r.exclusion_reason == "EXACT_DUPLICATE"),
        "verdict": "a split that assigns whole group_id values to one partition cannot leak "
        "exact, near-duplicate or same-specimen images; a random per-image split would "
        "(see groups_spanning_delivered_splits: the vendors' own splits already do)",
    }


def shortcut_predictability(
    rows: Iterable[CleanRow], feature: Callable[[CleanRow], str]
) -> dict[str, Any]:
    """In-sample upper bound of predicting the class from one categorical feature."""
    rows = list(rows)
    by_value: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_value[feature(r)][r.unified_class] += 1
    correct = sum(max(c.values()) for c in by_value.values())
    majority = max(Counter(r.unified_class for r in rows).values(), default=0)
    return {
        "distinct_values": len(by_value),
        "majority_class_baseline": round(majority / max(len(rows), 1), 4),
        "feature_only_accuracy_upper_bound": round(correct / max(len(rows), 1), 4),
    }


def _size_bucket(r: CleanRow) -> str:
    return f"{r.width}x{r.height}"


def suspicious_features(rows: list[CleanRow]) -> dict[str, Any]:
    included = [r for r in rows if r.included]
    features: dict[str, Callable[[CleanRow], str]] = {
        "source_dataset": lambda r: r.source_dataset,
        "resolution": _size_bucket,
        "extension": lambda r: r.extension,
        "resolution+extension": lambda r: f"{_size_bucket(r)}{r.extension}",
        "source+resolution": lambda r: f"{r.source_dataset}:{_size_bucket(r)}",
    }
    predictability = {name: shortcut_predictability(included, fn) for name, fn in features.items()}
    res_by_class: dict[str, dict[str, int]] = {}
    for cls in sorted({r.unified_class for r in included}):
        c = Counter(_size_bucket(r) for r in included if r.unified_class == cls)
        res_by_class[cls] = dict(c.most_common(4))
    source_by_class = {
        cls: dict(Counter(r.source_dataset for r in included if r.unified_class == cls))
        for cls in sorted({r.unified_class for r in included})
    }
    pre_augmented = Counter(
        r.source_dataset
        for r in included
        if any(t in _norm(r.original_filename) for t in AUGMENTATION_TOKENS)
    )
    single_source_classes = [cls for cls, srcs in source_by_class.items() if len(srcs) == 1]
    return {
        "predictability": predictability,
        "resolution_by_class_top4": res_by_class,
        "source_by_class": source_by_class,
        "classes_from_a_single_source": single_source_classes,
        "pre_augmented_filenames_by_dataset": dict(pre_augmented),
        "findings": [
            "resolution is class-correlated (e.g. one source delivers a single fixed size); "
            "the preprocessing resizes every image to 224x224 but JPEG quality/sharpness "
            "differences survive resizing — cannot be removed, must be reported and "
            "controlled by per-source evaluation",
            "source dataset is class-correlated because the sources cover different class "
            "subsets; source_dataset is metadata only and must never be an input feature; "
            "a split should stratify by (class, source) where possible",
            "the current dataset ships pre-augmented variants (filenames containing 'aug'); "
            "they are kept as near-duplicate groups so they cannot straddle partitions",
        ],
        "action": "documented only; nothing modified. Removing images to equalise "
        "resolution or source mix would be a design decision for approval.",
    }


def leakage_report(rows: list[CleanRow]) -> dict[str, Any]:
    return {
        "target_leakage": target_leakage(rows),
        "duplicate_leakage": duplicate_leakage(rows),
        "suspicious_features": suspicious_features(rows),
    }


# --- structured findings (data/audit/leakage_report.csv) --------------------------------

FINDING_COLUMNS = (
    "check",
    "finding",
    "evidence",
    "affected_images",
    "affected_classes",
    "severity",
    "action",
    "status",
)
NO_SEVERITY = "n/a (the project defines no severity convention)"
NOT_ESTABLISHED = "Cannot be established from available metadata."


def exif_summary(rows: Iterable[CleanRow], repo_root: Any) -> dict[str, dict[str, Any]]:
    """Per dataset: how many included files carry EXIF, an orientation tag, or a
    camera make/model. Reads headers only (Pillow is lazy); pixels are untouched."""
    from pathlib import Path

    from PIL import Image

    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        if not r.included:
            continue
        d = out.setdefault(
            r.source_dataset,
            {
                "images": 0,
                "with_exif": 0,
                "with_orientation": 0,
                "rotated_by_tag": 0,  # orientation tag != 1: pixels stored rotated/flipped
                "rotated_by_class": Counter(),
                "camera_models": Counter(),
            },
        )
        d["images"] += 1
        try:
            with Image.open(Path(repo_root) / r.filepath) as image:
                exif = image.getexif()
        except OSError:
            continue
        if len(exif):
            d["with_exif"] += 1
            orientation = exif.get(0x0112)
            if orientation is not None:
                d["with_orientation"] += 1
                if orientation != 1:
                    d["rotated_by_tag"] += 1
                    d["rotated_by_class"][r.unified_class] += 1
            model = " ".join(str(exif.get(t, "")).strip() for t in (0x010F, 0x0110)).strip()
            if model:
                d["camera_models"][model] += 1
    for d in out.values():
        d["camera_models"] = dict(d["camera_models"].most_common(5))
        d["rotated_by_class"] = dict(d["rotated_by_class"])
    return out


def findings_rows(
    rows: list[CleanRow], report: dict[str, Any], exif: dict[str, dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """One row per leakage check, in the teacher's order. Numbers come from `report`
    (leakage_report) and the manifest; wording states what is and is not established."""
    included = [r for r in rows if r.included]
    d, t, s = report["duplicate_leakage"], report["target_leakage"], report["suspicious_features"]
    by_group: dict[str, list[CleanRow]] = defaultdict(list)
    for r in included:
        by_group[r.group_id].append(r)
    within = [
        g for g, m in by_group.items() if len(m) > 1 and len({x.source_dataset for x in m}) == 1
    ]
    across = [g for g, m in by_group.items() if len({x.source_dataset for x in m}) > 1]
    classes_of = lambda gs: sorted({by_group[g][0].unified_class for g in gs})  # noqa: E731
    datasets = sorted({r.source_dataset for r in rows})
    specimen_datasets = sorted({r.source_dataset for r in rows if r.specimen_id})
    res = s["predictability"]

    def row(
        check: str, finding: str, evidence: str, images: Any, classes: Any, action: str, status: str
    ) -> dict[str, Any]:
        return {
            "check": check,
            "finding": finding,
            "evidence": evidence,
            "affected_images": images,
            "affected_classes": "; ".join(classes) if isinstance(classes, list) else classes,
            "severity": NO_SEVERITY,
            "action": action,
            "status": status,
        }

    out = [
        row(
            "1 duplicate leakage (within a dataset)",
            f"{d['excluded_exact_copies']} byte-identical copies excluded from the clean corpus; "
            f"{len(within)} near-duplicate/same-picture groups remain inside single datasets",
            "clean_manifest.csv: exclusion_reason=EXACT_DUPLICATE; group_id joins identical dHash "
            "(src/manifest.py::perceptual_dhash, exact-hash policy of build_split_manifest.py)",
            sum(len(by_group[g]) for g in within),
            classes_of(within),
            "exact copies excluded (representative kept); near-duplicates kept but a split/fold "
            "must "
            "assign whole group_id values",
            "resolved by manifest; enforced only if the split honours group_id",
        ),
        row(
            "2 cross-dataset duplicate leakage",
            f"{len(across)} included groups contain images from more than one dataset "
            f"(e.g. roboflow re-exports of current_freshwater; kaptai copies; SalmonScan photos "
            f"inside current_freshwater)",
            "duplicate_report.csv cross_dataset=True; clean_manifest.csv group_id",
            sum(len(by_group[g]) for g in across),
            classes_of(across),
            "same as 1: whole group_id per partition; 47 images in label-conflicting groups "
            "already "
            "excluded (LABEL_CONFLICT_GROUP)",
            "resolved by manifest; the cross-dataset label disagreements are evidence of label "
            "noise in the sources — human review recommended",
        ),
        row(
            "3 same-specimen / same-fish leakage",
            f"specimen identifiers exist only for {specimen_datasets or 'no dataset'}; "
            f"{d['specimen_groups']} specimen-bearing groups joined by group_id. For "
            f"{[x for x in datasets if x not in specimen_datasets]}: {NOT_ESTABLISHED}",
            "master_dataset.csv specimen_id (only a dataset that ships one, e.g. a metadata.csv); "
            "the active deliveries carry no fish identity"
            if not specimen_datasets
            else "master_dataset.csv specimen_id (from the dataset's metadata.csv); other "
            "deliveries carry no fish identity",
            sum(len(m) for m in by_group.values() if any(x.specimen_id for x in m)),
            classes_of([g for g, m in by_group.items() if any(x.specimen_id for x in m)]),
            "keep specimens inside one partition via group_id; for datasets without ids only "
            "exact/perceptual screening is possible (documented limitation)",
            "partially testable: resolved where ids exist; " + NOT_ESTABLISHED + " elsewhere. "
            f"dHash chaining joins near-duplicates into groups of up to {d['largest_group']} "
            "images (conservative but coarse)",
        ),
        row(
            "4 target leakage",
            "the label is written in folder paths (all datasets), in file names "
            + ", ".join(
                f"{k}: {v['filename_reveals_label_pct']}%" for k, v in t["per_dataset"].items()
            )
            + (
                ", and in mendeley/metadata.csv (health_condition)"
                if "mendeley" in t["per_dataset"]
                else "; no active dataset ships a label-bearing metadata table"
            ),
            "leakage_report.json target_leakage.per_dataset; src/dataset.py & src/manifest.py "
            "decode pixels only; no path/filename/metadata feature exists in the pipeline",
            sum(v["filename_reveals_label"] for v in t["per_dataset"].values()),
            "all",
            "no change needed; keep filename/path/metadata out of every model input (guarded by "
            "tests on the dataset classes)",
            "not a leak in the current pipeline; would become one if any loader used names",
        ),
        row(
            "5 suspicious metadata / features",
            (
                "mendeley metadata columns location, capture_condition, image_format, "
                "preprocessing, notes are constant (single value each) — no discriminative "
                "metadata; "
                if "mendeley" in datasets
                else "no metadata table in the active datasets; "
            )
            + (
                "EXIF present: "
                + ", ".join(
                    f"{k}: {v['with_exif']}/{v['images']} (orientation tag "
                    f"{v['with_orientation']}, "
                    f"of which stored rotated (tag != 1) {v['rotated_by_tag']} "
                    f"{v['rotated_by_class'] or ''}; camera {v['camera_models'] or 'none'})"
                    for k, v in exif.items()
                )
                if exif
                else "EXIF not scanned"
            ),
            "dataset metadata column values (Phase 1 inventory, where a dataset ships any); "
            "Pillow getexif() header scan of every clean image",
            sum(v["with_exif"] for v in exif.values()) if exif else 0,
            "all",
            "EXIF is not read by the pipeline (src/dataset.py::load_image = Pillow decode + "
            "convert('RGB'); no exif_transpose), so images whose orientation tag != 1 enter the "
            "model as stored, i.e. rotated relative to their intended view. Decision needed: "
            "apply PIL.ImageOps.exif_transpose in load_image (changes the existing loader; the "
            "baseline dataset has no EXIF so its results are unaffected) or leave as is. Camera "
            "identity per source reaches the model only through pixel statistics.",
            "no metadata reaches the model; the orientation issue is a preprocessing decision for "
            "human review (not silently changed); camera differences fold into finding 7",
        ),
        row(
            "6 class <-> image-resolution shortcut",
            f"resolution alone predicts the class with an in-sample upper bound of "
            f"{res['resolution']['feature_only_accuracy_upper_bound']} vs majority baseline "
            f"{res['resolution']['majority_class_baseline']}; see resolution_by_class_top4 "
            "(classes differ in their dominant resolutions, e.g. Aeromoniasis is mostly 128x128)",
            "leakage_report.json suspicious_features.predictability / resolution_by_class_top4",
            len(included),
            sorted({r.unified_class for r in included}),
            "cannot be removed by the pipeline (all images are resized to 224x224, but sharpness/"
            "compression traces survive); mitigation is methodological: per-source evaluation and "
            "(class, source)-stratified partitions; a resolution-only control classifier would "
            "quantify it and is proposed for a later phase",
            "documented; requires human decision (accept, or equalise sources by resampling)",
        ),
        row(
            "7 other dataset / source shortcuts",
            "source_dataset predicts the class at "
            f"{res['source_dataset']['feature_only_accuracy_upper_bound']} "
            f"(baseline {res['source_dataset']['majority_class_baseline']}); classes present in a "
            "single "
            f"source: {s['classes_from_a_single_source'] or 'none'}; file extension: "
            f"{res['extension']['feature_only_accuracy_upper_bound']}; image mode: "
            + ", ".join(f"{k}: {v}" for k, v in Counter(r.extension for r in included).items())
            + f"; pre-augmented files in current_freshwater: "
            f"{s['pre_augmented_filenames_by_dataset'].get('current_freshwater', 0)}; "
            f"{d['groups_spanning_delivered_splits']} groups straddle the vendors' own "
            "train/valid/test folders",
            "leakage_report.json suspicious_features; clean_manifest.csv extension/original_split",
            len(included),
            sorted({r.unified_class for r in included}),
            "source_dataset stays metadata; stratify partitions by (class, source); report "
            "per-source "
            "metrics; ignore the vendors' train/valid/test folders entirely (they leak)",
            "documented; per-source evaluation to be built into the CV phase",
        ),
    ]
    return out
