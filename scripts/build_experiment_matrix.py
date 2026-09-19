#!/usr/bin/env python
"""Phase 12 — generate (or refresh the statuses of) the 100-experiment matrix.

    python scripts/build_experiment_matrix.py            # write the matrix (all PENDING)
    python scripts/build_experiment_matrix.py --refresh  # statuses <- experiments/*/status.json

Nothing is trained here.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, ROOT_DIR  # noqa: E402
from src.experiment_matrix import (  # noqa: E402
    build_matrix,
    experiments_root,
    matrix_path,
    read_matrix,
    refresh_statuses,
    write_matrix,
)
from src.utils import get_logger  # noqa: E402

logger = get_logger("experiment_matrix")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--repo-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    path = matrix_path(args.repo_root)
    if args.refresh:
        rows = refresh_statuses(read_matrix(path), experiments_root(args.repo_root))
    else:
        if path.exists():
            logger.error("%s exists; use --refresh to update statuses", path)
            return 2
        rows = build_matrix(args.data_dir, args.repo_root)
    write_matrix(rows, path)
    logger.info(
        "%s: %d rows, statuses %s", path, len(rows), dict(Counter(r["status"] for r in rows))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
