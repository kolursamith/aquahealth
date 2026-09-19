"""Disease-label harmonisation for the multi-dataset experiment (Phase 3).

The unified taxonomy starts from the project's agreed eight classes
(`src.manifest.CANONICAL_CLASSES`). Every (source dataset, original class) pair
observed in the delivered data gets an explicit mapping row with a status:

    EXACT_MATCH        the original label string is one the project already maps
                       (a key of SOURCE_FOLDER_TO_CLASS) or equals a canonical name
    SUPPORTED_MAPPING  a different string for the same disease term, supported by
                       the dataset's own documentation/metadata
    UNRESOLVED         cannot be mapped from the available evidence; human decision
    EXCLUDED           deliberately left out of the unified taxonomy

Only EXACT_MATCH and SUPPORTED_MAPPING rows carry a `unified_class`. No image is
moved or relabelled by this module; the master manifest merely records the
mapping so a later phase can act on it after human review.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

from src.manifest import CANONICAL_CLASSES, SOURCE_FOLDER_TO_CLASS

MAPPING_STATUSES = ("EXACT_MATCH", "SUPPORTED_MAPPING", "UNRESOLVED", "EXCLUDED")
UNIFIED_CLASSES: tuple[str, ...] = CANONICAL_CLASSES


@dataclass(frozen=True)
class LabelMapping:
    source_dataset: str
    original_class: str
    unified_class: str  # "" unless EXACT_MATCH / SUPPORTED_MAPPING
    mapping_status: str
    reason: str
    evidence: str

    def __post_init__(self) -> None:
        if self.mapping_status not in MAPPING_STATUSES:
            raise ValueError(f"unknown mapping_status {self.mapping_status!r}")
        mapped = self.mapping_status in ("EXACT_MATCH", "SUPPORTED_MAPPING")
        if mapped and self.unified_class not in UNIFIED_CLASSES:
            raise ValueError(
                f"{self.source_dataset}/{self.original_class}: {self.mapping_status} "
                f"needs a unified class from {UNIFIED_CLASSES}, got {self.unified_class!r}"
            )
        if not mapped and self.unified_class:
            raise ValueError(
                f"{self.source_dataset}/{self.original_class}: {self.mapping_status} "
                "must not carry a unified class"
            )


REVIEW_STATUS = {
    "EXACT_MATCH": "accepted",
    "SUPPORTED_MAPPING": "accepted",
    "UNRESOLVED": "ambiguous (requires review)",
    "EXCLUDED": "excluded",
}
LABEL_MAPPING_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(LabelMapping)) + (
    "review_status",
)

_PROJECT_MAP = "src/manifest.py::SOURCE_FOLDER_TO_CLASS (project's agreed mapping)"


def _exact_from_project(source: str, folder: str) -> LabelMapping:
    return LabelMapping(
        source,
        folder,
        SOURCE_FOLDER_TO_CLASS[folder],
        "EXACT_MATCH",
        "identical folder name to the current dataset's source folder, which the "
        "project already maps",
        _PROJECT_MAP,
    )


# The delivered classes, exactly as their folder / metadata strings read.
LABEL_MAPPING: tuple[LabelMapping, ...] = (
    # --- current_freshwater: the project's own mapping, unchanged ---
    *(_exact_from_project("current_freshwater", folder) for folder in SOURCE_FOLDER_TO_CLASS),
    # --- roboflow: the same seven folder strings as the current dataset (no EUS) ---
    *(
        _exact_from_project("roboflow", folder)
        for folder in SOURCE_FOLDER_TO_CLASS
        if folder != "EUS"
    ),
    # --- kaptai ("Fresh Water Fish Dataset"): no README, no metadata ---
    LabelMapping(
        "kaptai",
        "EUS",
        "EUS Disease",
        "EXACT_MATCH",
        "folder name is the same string 'EUS' the project already maps",
        _PROJECT_MAP + "; folder name only — the dataset ships no documentation",
    ),
    LabelMapping(
        "kaptai",
        "Healthy Fish",
        "Healthy Fish",
        "EXACT_MATCH",
        "folder name equals the canonical class name",
        _PROJECT_MAP,
    ),
    LabelMapping(
        "kaptai",
        "Argulus",
        "",
        "UNRESOLVED",
        "Argulus is a specific parasite (fish louse); the project's 'Parasitic Disease' "
        "class has no documented definition of which parasites it covers, so the "
        "mapping is plausible but not evidenced",
        "no dataset README/metadata; no project document defines 'Parasitic Disease'",
    ),
    LabelMapping(
        "kaptai",
        "Redspot",
        "",
        "UNRESOLVED",
        "'Redspot' could denote Bacterial Red Disease or Epizootic Ulcerative Syndrome "
        "(also called red-spot disease); the folder even contains files named 'EUS (n).jpg'",
        "folder listing of kaptai/Redspot (files 'EUS  (2).jpg' ... 'EUS  (14).jpg'); "
        "no documentation",
    ),
    LabelMapping(
        "kaptai",
        "THE BACTERIAL GILL ROT",
        "",
        "UNRESOLVED",
        "similar wording to 'Bacterial Gill Disease' but no documentation states the "
        "equivalence; the folder also contains a histology figure panel",
        "folder name only; contact-sheet inspection during the Phase-1 audit",
    ),
    LabelMapping(
        "kaptai",
        "Tail And Fin Rot",
        "",
        "UNRESOLVED",
        "no canonical class for fin rot ('White Tail Disease' is a distinct viral "
        "condition); needs a decision: new unified class or exclusion",
        "src/manifest.py::CANONICAL_CLASSES has no fin-rot class",
    ),
    LabelMapping(
        "kaptai",
        "Broken antennae and rostrum",
        "",
        "EXCLUDED",
        "describes mechanical damage to crustacean anatomy (antennae, rostrum), not a "
        "fish disease in the unified taxonomy; the folder mixes a prawn photo with "
        "unrelated fish images",
        "folder name; contact-sheet inspection during the Phase-1 audit",
    ),
    # --- mendeley (MatsyaDx-BD): labels from metadata.csv `health_condition` ---
    LabelMapping(
        "mendeley",
        "Bacterial Gill Diseases",
        "Bacterial Gill Disease",
        "SUPPORTED_MAPPING",
        "same disease term, plural form",
        "MatsyaDx-BD/metadata.csv column health_condition; species Rui/Silver carp/"
        "Grass carp (South-Asian freshwater aquaculture, same domain as the current set)",
    ),
    LabelMapping(
        "mendeley",
        "Bacterial Red Diseases",
        "Bacterial Red Disease",
        "SUPPORTED_MAPPING",
        "same disease term, plural form",
        "MatsyaDx-BD/metadata.csv column health_condition",
    ),
    LabelMapping(
        "mendeley",
        "Epizootic Ulcerative Syndrome (EUS)",
        "EUS Disease",
        "SUPPORTED_MAPPING",
        "label spells out the EUS abbreviation the project's class name uses",
        "MatsyaDx-BD/metadata.csv column health_condition",
    ),
    LabelMapping(
        "mendeley",
        "Healthy Fish",
        "Healthy Fish",
        "EXACT_MATCH",
        "label equals the canonical class name",
        "MatsyaDx-BD/metadata.csv column health_condition",
    ),
    # --- paper_dataset (SalmonScan): two health-status classes, salmon ---
    LabelMapping(
        "paper_dataset",
        "FreshFish",
        "",
        "UNRESOLVED",
        "'fresh' (healthy) salmon; merging a different species and rearing domain into "
        "'Healthy Fish' is a design decision, not an evidenced label equivalence",
        "folder name only; team note describes SalmonScan as an external-domain dataset",
    ),
    LabelMapping(
        "paper_dataset",
        "InfectedFish",
        "",
        "UNRESOLVED",
        "'infected' names no disease; it cannot be assigned to any of the seven " "disease classes",
        "folder name only; no per-image disease label in the delivery",
    ),
)


def lookup(source_dataset: str, original_class: str) -> LabelMapping | None:
    for mapping in LABEL_MAPPING:
        if mapping.source_dataset == source_dataset and mapping.original_class == original_class:
            return mapping
    return None


def validate_mapping_table(mappings: Iterable[LabelMapping] = LABEL_MAPPING) -> None:
    seen: set[tuple[str, str]] = set()
    for m in mappings:
        key = (m.source_dataset, m.original_class)
        if key in seen:
            raise ValueError(f"duplicate mapping row for {key}")
        seen.add(key)


def write_label_mapping_csv(path: Path, mappings: Iterable[LabelMapping] = LABEL_MAPPING) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LABEL_MAPPING_COLUMNS))
        writer.writeheader()
        for m in mappings:
            writer.writerow({**asdict(m), "review_status": REVIEW_STATUS[m.mapping_status]})


def read_label_mapping_csv(path: Path) -> list[LabelMapping]:
    with Path(path).open(newline="") as handle:
        return [
            LabelMapping(**{k: v for k, v in row.items() if k != "review_status"})
            for row in csv.DictReader(handle)
        ]
