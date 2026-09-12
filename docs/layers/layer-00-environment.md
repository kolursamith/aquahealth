# Layer 0 — Python Environment

## 1. Objective

Create a reproducible Python development environment for AquaHealth AI, and a
verification mechanism that proves the environment matches what the project
declares. No model, no dataset, no training code.

## 2. Architecture

```
.python-version          declared interpreter (single source of truth)
requirements/base.txt    runtime dependencies   (empty at Layer 0)
requirements/dev.txt     quality-gate toolchain (pytest, ruff, mypy, black)
        │
        ▼
src/environment.py       stdlib-only inspection logic (testable, typed)
        │
        ▼
scripts/verify_environment.py   thin CLI → human text or --json; exit 0/1
```

The logic lives in `src/environment.py` rather than in the script so it can be
unit-tested with injected inputs. The module imports **only** the standard
library, because it must run correctly in an environment where nothing is
installed — which is precisely the condition it diagnoses.

The verifier derives what *should* be installed by parsing `requirements/`,
rather than holding its own hardcoded list. Adding a dependency in a later
layer therefore extends the environment check automatically, with no second
place to update.

## 3. Files

**Created**
- `requirements/base.txt` — runtime dependencies; intentionally empty at Layer 0
- `requirements/dev.txt` — pinned toolchain
- `src/environment.py` — environment inspection logic
- `tests/test_environment.py` — 32 tests
- `docs/layers/layer-00-environment.md` — this document

**Modified**
- `scripts/verify_environment.py` — rewritten as a CLI over `src/environment.py`
- `.github/workflows/ci.yml` — uses `requirements/dev.txt`; runs the env check; dropped `--cov`
- `.github/workflows/python-checks.yml` — uses `requirements/dev.txt`; mypy now covers `scripts/`
- `README.md` — real setup procedure + per-layer build status table
- `docs/team/student_{1..4}.md` — updated install command
- `tests/test_preprocessing.py`, `tests/test_prediction.py` — skip guards (see §9)
- `src/preprocessing.py`, `src/dataset.py`, `src/train.py`, `src/predict.py`, `scripts/create_split.py`, `src/utils.py` — lint/format only, no behavior change

**Deleted**
- `requirements.txt`, `requirements-dev.txt`

## 4. Dependencies

No runtime dependency is installed at this layer. Development-only:

| Package | Version | Reason | Used by |
|---|---|---|---|
| pytest | 9.1.1 | test runner for every layer's acceptance gate | all tests |
| ruff | 0.16.7 | linter | `ruff check .` |
| mypy | 2.3.1 | static type checker | `mypy src app scripts` |
| black | 26.5.1 | formatter | `black --check .` |

Also present from venv seeding: pip 26.2.1, setuptools 84.0.0, wheel 0.48.0,
packaging 26.3.

Removed: `timm` (Layer 3 will use TorchVision's EfficientNet-B0, so timm is not
needed) and `pytest-cov` (coverage is not enforced yet; it will be reintroduced
if and when a coverage gate is actually added).

## 5. Implementation Decisions

**D1 — Python 3.11.15, not the installed 3.14.7.** The only interpreter on PATH
was Homebrew Python 3.14.7. Both were verified viable: a `uv pip compile`
dry-run resolved the entire planned dependency set (torch, torchvision, numpy,
pillow, opencv-python-headless, scikit-learn, pandas, matplotlib, tqdm,
streamlit, albumentations) successfully for both 3.11 and 3.14, with identical
versions except numpy (2.4.6 vs 2.5.3). 3.11 was chosen because it matches the
committed `.python-version` and the CI pin, so local and CI stay identical.
Installed via `uv python install 3.11`.

**D2 — Layered requirements files.** The previous `requirements.txt` declared
torch, timm, albumentations, streamlit and opencv together. Installing it would
have silently pre-installed the dependencies of Layers 1–11 before those layers
were built or validated, defeating the gate system and contradicting the
instruction to determine the actual environment *before* selecting a PyTorch
installation. Dependencies are now added by the layer that needs them.

**D3 — No GPU/CUDA/MPS claim at this layer.** The machine is an Apple M1 Pro
(16-core GPU, Metal 3) with no NVIDIA hardware and no `nvidia-smi`, so CUDA is
not applicable. Whether PyTorch can use the Metal (MPS) backend cannot be
established without importing torch, so the verifier reports it as *not
determined in Layer 0* instead of guessing. Layer 1 owns that verification.

**D4 — The verifier parses `requirements/` instead of hardcoding names.** Uses
`importlib.metadata` to check distribution presence without importing packages,
which avoids a distribution-name → import-name mapping table
(`opencv-python-headless` → `cv2`, `pillow` → `PIL`) that would rot.

**D5 — Unsupported requirement lines raise.** Editable installs, direct URLs
and pip options raise `ValueError` rather than being skipped, so the check can
never silently under-report what the project declares.

**D6 — `.python-version` pinned to minor (`3.11`), not patch.** The local patch
is 3.11.15; `actions/setup-python` with `3.11` resolves to the latest 3.11.x.
Pinning the patch would make the two disagree on every CPython release.

## 6. Tests

`tests/test_environment.py` — 32 tests:

- requirement-name parsing across 8 forms (bare, `==`, `>=`, `!=`, extras, markers, spaced)
- requirements parsing: comments, inline comments, blank lines, raw-line fidelity
- `-r` and `--requirement` include following
- include-cycle detection raises
- unsupported option (`-e .`) raises
- distribution status with an injected lookup (installed + missing) and against real metadata
- `.python-version` parsing, including 3 malformed inputs that must raise
- pin matching (match and mismatch)
- CPU brand from an injected runner, and fallback when the command raises
- python report agrees with the running interpreter
- report `ok`/`not ok` for missing dist, mismatched interpreter, satisfied env
- de-duplication across requirement files
- **CLI integration (subprocess):** exit 0 on this environment, valid `--json`, exit 1 on a missing declared distribution

Dependency injection is used for `lookup` and `runner` so the tests assert
behavior rather than whatever happens to be installed on the machine.

## 7. Commands Executed

```bash
uv python install 3.11                       # → 3.11.15
uv venv --python 3.11 --seed .venv
.venv/bin/python -m pip install --upgrade pytest ruff mypy black
.venv/bin/python -m pytest -q -rs
.venv/bin/ruff check .
.venv/bin/black --check .
.venv/bin/mypy src app scripts
.venv/bin/python scripts/verify_environment.py
/opt/homebrew/bin/python3.14 scripts/verify_environment.py    # negative test
```

## 8. Results

Measured environment:

| Property | Value |
|---|---|
| OS | macOS 14.8.1 (Darwin 23.6.0, build 23J30) |
| Architecture | arm64 |
| CPU | Apple M1 Pro, 10 logical cores |
| Memory | 16.0 GiB |
| GPU | Apple M1 Pro, 16 cores, Metal 3 |
| NVIDIA / CUDA | not present / not applicable |
| Python | 3.11.15 (CPython), in `.venv` |
| pip | 26.2.1 |

Gate results:

| Check | Command | Result |
|---|---|---|
| Tests | `pytest -q -rs` | **40 passed, 2 skipped** |
| Lint | `ruff check .` | **All checks passed** |
| Format | `black --check .` | **29 files unchanged** |
| Types | `mypy src app scripts` | **no issues in 23 source files** |
| Smoke (positive) | `verify_environment.py` in venv | **exit 0**, all declared distributions OK |
| Smoke (negative) | same script under Python 3.14.7 | **exit 1**, pin mismatch + 4 missing distributions detected |

## 9. Known Limitations

1. **No dependency lock file.** Runtime dependencies are pinned exactly in
   `requirements/dev.txt`, but there is no full transitive lock. Deferred until
   the dependency set stabilizes (around Layer 6).
2. **Two test modules skip.** `tests/test_preprocessing.py` and
   `tests/test_prediction.py` require numpy/OpenCV, which belong to Layers 5–6.
   They are guarded with `pytest.importorskip` and report an explicit reason.
   *Regression risk:* a genuinely broken install would surface as a skip rather
   than a failure. Mitigated by `verify_environment.py` running in CI, which
   fails on any missing declared distribution. The guards must be removed when
   Layers 5 and 6 land.
3. **`src/preprocessing.py` CLAHE change is unexecuted.** The ambiguous variable
   `l` was renamed to `lightness`/`green_red`/`blue_yellow` to clear lint. The
   test covering it currently skips (no OpenCV), so the change is textual and
   unverified until Layer 6 runs it.
4. **`src/model.py` references `timm`,** which is no longer a declared
   dependency. It is Layer 3 scaffolding, is imported by no test and no Layer 0
   code path, and will be replaced by TorchVision in Layer 3.
5. **CI is unverified on GitHub.** The workflow changes were validated by
   running the identical commands locally; no CI run has executed them yet.
6. **`verify_environment.py` does not check version specifiers** — it verifies a
   distribution is installed, not that its version satisfies the pin. pip
   enforces that at install time.
7. **Single-machine validation.** All results above are from one macOS arm64
   machine. The environment has not been validated on Linux or Windows.

## 10. Acceptance Gate

**PASS**

| Criterion | Evidence |
|---|---|
| Python executes | 3.11.15 in `.venv`, verified |
| Virtual environment works | `sys.prefix != sys.base_prefix`, pip 26.2.1 operational |
| Environment verification runs | exit 0 positive, exit 1 negative |
| Required base dependencies import | all 4 declared distributions present; no runtime dependency declared at this layer by design |
| No unexplained dependency conflicts | clean resolution; `uv pip compile` dry-run confirms the forward dependency set resolves on 3.11 |

## 11. Next Layer Prerequisites

Layer 1 (PyTorch) can begin. It must:

- add `torch` to `requirements/base.txt` with a pinned version and a reason
- select the installation appropriate for macOS arm64 — this machine has no
  CUDA, so the default PyPI wheel applies; do not assume a CUDA build
- verify import, version, tensor creation, a tensor operation, and autograd
  backward on CPU
- determine MPS availability empirically via `torch.backends.mps.is_available()`
  and record the result — Layer 0 deliberately did not
- extend `scripts/verify_environment.py` to report torch-level accelerator facts
  now that torch is importable
