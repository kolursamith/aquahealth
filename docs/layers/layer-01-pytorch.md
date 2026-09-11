# Layer 1 — PyTorch

## 1. Objective

Install PyTorch as the project's first runtime dependency, prove that its core
primitives (tensors, arithmetic, autograd, seeding, serialization) work on this
machine, and establish — empirically, not by assumption — which compute
backends are available. No model, no dataset, no training.

## 2. Architecture

```
requirements/base.txt        torch==2.14.0 (pinned, with reason)
        │
        ▼
src/device.py                the ONE place that decides which device to use
  ├── available_backends()   cuda / mps / cpu availability, at call time
  ├── resolve_device()       None → best available; explicit name → honour or raise
  └── torch_report()         runtime facts as a frozen dataclass
        │
        ▼
scripts/verify_environment.py   adds a "PyTorch" section (text + --json)
                                 when torch is installed; still runs when it is not
```

`src/environment.py` remains stdlib-only. Torch-level facts are produced by
`src/device.py` and stitched into the verifier via a lazy import, so the
verifier keeps working in an environment where torch is absent — the very
situation it exists to diagnose.

### Device policy

`resolve_device(preferred=None)`:

| Call | Behaviour |
|---|---|
| `resolve_device()` | first available of `cuda` → `mps` → `cpu` |
| `resolve_device("cpu")` | always `cpu` |
| `resolve_device("mps")` / `("cuda")` / `("cuda:1")` | returned only if that backend is available, else `RuntimeError` |
| `resolve_device("xla")` | `ValueError` — not a backend this project supports |

An explicit request never silently falls back. Silently landing on the CPU
when a GPU was requested turns a 50× slowdown into a "working" run; a loud
failure is the correct behaviour. Later layers must obtain their device from
this function rather than querying `torch.cuda` / `torch.backends.mps`
directly.

## 3. Files

**Created**
- `src/device.py` — device policy + `TorchReport`
- `tests/test_torch.py` — 23 tests (runtime primitives, 6 of them parametrized over every available device)
- `tests/test_device.py` — 14 tests (policy, with backends injected via monkeypatch)
- `docs/layers/layer-01-pytorch.md` — this document

**Modified**
- `requirements/base.txt` — declares `torch==2.14.0`
- `scripts/verify_environment.py` — PyTorch section; `--json` gains a `torch` key
- `src/environment.py` — docstrings only (pointed at `src.device` instead of "Layer 1")
- `src/utils.py` — `set_seed` bug fix (see §9)
- `.github/workflows/ci.yml`, `.github/workflows/python-checks.yml` — install from the CPU torch index
- `README.md` — build-status table

## 4. Dependencies

| Package | Version | Reason | Used by |
|---|---|---|---|
| torch | 2.14.0 | deep-learning runtime | `src/device.py`, `src/utils.py`, all later layers |

Transitively installed by torch: filelock 3.32.6, fsspec 2026.7.0, Jinja2 3.1.6,
MarkupSafe 3.0.3, mpmath 1.3.0, networkx 3.6.1, sympy 1.14.0, typing_extensions
4.16.0. None are imported directly by project code.

**numpy is not installed at this layer** (see §10, limitation 1).

## 5. Implementation Decisions

**D1 — torch 2.14.0, the current PyPI release.** Before pinning, a
`pip install --dry-run` confirmed that `torch==2.14.0` resolves together with
the latest `torchvision==0.29.0`, so Layer 2 will not need to re-pin torch.

**D2 — Default PyPI wheel locally; CPU-index wheel in CI.** On macOS arm64 the
PyPI wheel is the only option and carries the MPS backend. On Linux the PyPI
wheel bundles CUDA and pulls ~10 `nvidia-*` packages totalling several GB,
which a GPU-less GitHub runner would download for nothing. CI therefore
installs with `--extra-index-url https://download.pytorch.org/whl/cpu`. This
was validated by a dry-run resolution targeting `manylinux_2_28_x86_64` /
Python 3.11: pip selects `torch 2.14.0+cpu` from `download.pytorch.org` and
**zero** `nvidia-*` packages (25 packages total). The `+cpu` local label is
ignored by the `==2.14.0` specifier per PEP 440, so `requirements/base.txt`
needs no CI-specific variant. The extra index cannot live in the requirements
file itself: the Layer 0 parser rejects option lines by design (D5 there).

**D3 — MPS availability determined empirically.** `torch.backends.mps.is_built()`
and `.is_available()` both return `True` on this machine; a 1000×1000
`randn` and a matmul were executed on `mps` and compared against CPU results
before any test was written. CUDA: `torch.version.cuda is None` (not built into
the macOS wheel), `torch.cuda.is_available() is False`.

**D4 — Policy tests inject backends.** `tests/test_device.py` monkeypatches
`available_backends` so the cuda-first / mps-fallback / cpu-fallback branches
and the "requested cuda on a cuda-less machine raises" branch are all exercised
on this machine, which has no CUDA. Tests that need real hardware are
parametrized over whatever `available_backends()` reports, so they run on
`cpu` and `mps` here and would additionally run on `cuda` elsewhere.

**D5 — `torch.manual_seed` seeds every backend.** `set_seed` calls
`torch.manual_seed` only; per the torch docs it seeds CPU, all CUDA devices and
MPS. `test_manual_seed_reproduces_on_device` confirms this empirically on MPS.

## 6. Tests

`tests/test_torch.py` — 23 tests (6 parametrized × 2 devices here):

- declared pin in `requirements/base.txt` equals `torch.__version__` (local label stripped)
- cpu backend always reported available
- tensor creation: shape, default float32, explicit int64
- matmul correctness vs a hand-computed result — every device
- `exp`/`tanh`/`sigmoid`/`relu` on device match CPU within 1e-5 — every device
- `sum`/`mean`/`max`/`argmax` — every device
- autograd: gradient of Σx² is 2x
- autograd through `nn.Linear`: grads exist, correct shapes, finite — every device
- `torch.no_grad` disables graph recording
- an SGD step changes parameters
- `torch.manual_seed` reproduces CPU tensors; `set_seed` reproduces; reproduces on device
- `state_dict` save → `torch.load(weights_only=True)` → identical tensors
- device → CPU round trip is lossless — every device
- **CLI (subprocess):** verifier prints a PyTorch section with the running version; `--json` `torch` key matches live `torch.*` calls

`tests/test_device.py` — 14 tests:

- priority order is `("cuda", "mps", "cpu")`
- `available_backends` keys match priority; values match torch
- auto: cuda first; mps when no cuda; cpu when nothing else
- auto selection on this machine yields a device that can compute
- explicit `cpu` always works; explicit available backend returned; `cuda:1` keeps its index
- explicit unavailable backend raises `RuntimeError` (no fallback)
- unsupported type raises `ValueError`; gibberish rejected by torch
- `torch_report` mirrors live torch facts; honours `preferred`

## 7. Commands Executed

```bash
.venv/bin/python -m pip index versions torch            # → 2.14.0 latest
.venv/bin/python -m pip install --dry-run --report - torch==2.14.0 torchvision   # pairing check
.venv/bin/python -m pip install torch==2.14.0
.venv/bin/python -m pip install --dry-run --report r.json \
    --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 \
    --python-version 3.11 --only-binary=:all: --target ./t \
    --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/dev.txt   # CI check
.venv/bin/python -m pytest -q -rs
.venv/bin/ruff check .
.venv/bin/black --check .
.venv/bin/mypy src app scripts
.venv/bin/python scripts/verify_environment.py
.venv/bin/python scripts/verify_environment.py --json
/opt/homebrew/bin/python3.14 scripts/verify_environment.py   # negative: no torch
```

## 8. Results

Measured torch runtime on this machine (Apple M1 Pro, macOS 14.8.1, Python 3.11.15):

| Property | Value |
|---|---|
| torch | 2.14.0 (git 08187d9e0f, release build) |
| CUDA built / available | no / no |
| MPS built / available | **yes / yes** |
| CPU intra-op threads | 8 |
| Auto-selected device | **mps** |
| CPU vs MPS matmul | identical |

Gate results:

| Check | Command | Result |
|---|---|---|
| Tests | `pytest -q -rs` | **77 passed, 2 skipped** (was 40 + 2) |
| Device parametrization | `pytest -v tests/test_torch.py` | 6 tests × `[cpu]` + 6 × `[mps]`, all PASSED |
| Lint | `ruff check .` | All checks passed |
| Format | `black --check .` | 32 files unchanged |
| Types | `mypy src app scripts` | no issues in 24 source files |
| Smoke (positive) | `verify_environment.py` | exit 0; PyTorch section present; `selected device  mps` |
| Smoke (JSON) | `verify_environment.py --json` | `torch` key present and consistent with live calls |
| Smoke (negative) | same script under Python 3.14 (no torch) | runs; prints `cuda / mps  not determined (requires torch)`; reports torch missing |
| Regression (Layer 0) | all 40 Layer 0 tests | pass unchanged |

## 9. Bugs Found and Fixed

**B1 — `src/utils.py` hard-imported numpy, which is not installed.**
Discovered when `tests/test_torch.py` imported `set_seed` and collection
failed with `ModuleNotFoundError: numpy`. Root cause: skeleton code from the
initial commit treated numpy as mandatory and torch as optional — the inverse
of what the project declares at this layer. Fix: `torch.manual_seed` is now
unconditional (torch is declared) and numpy is seeded only if importable (it is
not declared yet). Regression test: `test_set_seed_reproduces_torch_random_tensors`.
The remaining guard is to be removed by the layer that declares numpy.

## 10. Known Limitations

1. **torch warns at import: `Failed to initialize NumPy: No module named 'numpy'`.**
   Every `import torch` (and therefore every test run and every verifier run)
   emits this `UserWarning` once. Cause: torch's numpy bridge is optional and
   initialises lazily; nothing in Layer 1 uses `Tensor.numpy()` or
   `torch.from_numpy`, so functionality is unaffected — all 77 tests pass.
   It was deliberately *not* silenced with a warning filter (that would hide a
   real signal) and numpy was *not* installed early to make it go away.
   **Resolution point: Layer 2.** torchvision declares numpy as a hard
   dependency (confirmed by the dry-run in D1), so it arrives there as a
   genuine requirement, and the warning disappears.
2. **CI still unverified on GitHub.** The CPU-index install was validated by a
   Linux-targeted dry-run resolution from this machine, and every gate command
   was run locally, but no GitHub Actions run has executed yet.
3. **No CUDA path executed.** The cuda-first policy branch is covered only via
   injected backends; no CUDA hardware is available to run a tensor on it.
4. **MPS results are compared to CPU at 1e-5 tolerance,** not bit-exactly, for
   transcendental functions. Matmul, reductions and seeding are checked exactly.
5. **`set_seed` does not set `torch.use_deterministic_algorithms`** or cuDNN
   flags. Layer 7 (training) decides whether that determinism/performance
   trade-off is wanted.
6. **Two Layer 0 skips persist** (`test_prediction.py`, `test_preprocessing.py`) —
   unchanged from Layer 0; they belong to later layers.

## 11. Acceptance Gate

**PASS**

| Criterion | Evidence |
|---|---|
| Implementation exists | `src/device.py`; verifier extended |
| Imports work | `import torch` 2.14.0; `from src.device import …` |
| Unit tests pass | 37 new tests pass |
| Integration tests pass | verifier subprocess tests (text + JSON) pass |
| Smoke tests pass | positive, JSON, and no-torch negative runs behave as specified |
| Lint / format / types | ruff, black, mypy all clean |
| Runtime behaviour verified | tensor ops, autograd, seeding, serialization on cpu and mps |
| No blocking bug | B1 fixed at root, regression-tested |
| No unexplained error | the single warning is explained and has a named resolution point |
| No regression | Layer 0's 40 tests unchanged and passing |
| Documentation updated | this file; README status table |

## 12. Next Layer Prerequisites

Layer 2 (TorchVision) must:

- add `torchvision==0.29.0` to `requirements/base.txt` (pairing with torch 2.14.0 already confirmed)
- accept numpy and pillow as its declared transitive dependencies, and record their resolved versions
- remove the numpy guard in `src/utils.set_seed` once numpy is declared, and confirm the import-time warning is gone
- verify torchvision import, version, and that `torchvision.transforms.v2` and `torchvision.models` are importable
- verify a `PIL.Image` → tensor conversion and its shape/dtype/range
- re-run the Layer 0 and Layer 1 suites as regression
