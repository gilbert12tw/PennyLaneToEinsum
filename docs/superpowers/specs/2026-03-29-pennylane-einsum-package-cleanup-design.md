# Design: pennylane-einsum Package Cleanup

**Date:** 2026-03-29
**Author:** gilbert (gilbert12.tw@gmail.com)
**Status:** Approved

---

## Goal

Make the `pennylane-einsum` package shippable: consistent importable name, correct metadata, working CI, and accurate documentation. No new features.

---

## Scope

**In scope:**
- Rename `converter/` module to `pennylane_einsum/`
- Fix `pyproject.toml` metadata (author, license, classifiers, package discovery)
- Update all imports in tests, examples, and scripts
- Fix README (module name, gate support claim)
- Add GitHub Actions CI workflow
- Clean up stale egg-info from git tracking

**Out of scope:**
- Measurement/observable support
- `cotengra` integration
- New gate types
- Performance benchmarks

---

## 1. Module Rename

`converter/` → `pennylane_einsum/`

Files renamed:
- `converter/__init__.py` → `pennylane_einsum/__init__.py`
- `converter/circuit_to_einsum.py` → `pennylane_einsum/circuit_to_einsum.py`
- `converter/index_manager.py` → `pennylane_einsum/index_manager.py`

Internal relative imports (`from .index_manager import IndexManager`) are unchanged.

`pennylane_einsum/__init__.py` gains a `__version__` export:
```python
from .circuit_to_einsum import CircuitToEinsum, contract_einsum
from .index_manager import IndexManager

__version__ = "0.1.0"
__all__ = ["CircuitToEinsum", "contract_einsum", "IndexManager", "__version__"]
```

---

## 2. pyproject.toml

Changes:
- `[tool.setuptools.packages.find]` include: `pennylane_einsum*`
- Add `authors = [{name = "gilbert", email = "gilbert12.tw@gmail.com"}]`
- Add `license = {text = "MIT"}`
- Add `urls` table with `Homepage` and `Repository` fields (placeholder)
- Add PyPI classifiers:
  - `Programming Language :: Python :: 3`
  - `License :: OSI Approved :: MIT License`
  - `Topic :: Scientific/Engineering :: Physics`

---

## 3. Tests, Examples, Scripts

All occurrences of `from converter import` replaced with `from pennylane_einsum import`:

- `tests/test_method_one_basic.py`
- `tests/test_multiqubit_gates.py`
- `tests/test_unsupported_ops.py`
- `examples/basic_circuits.py`
- `scripts/scan_unsupported_ops.py`

---

## 4. README

- Replace all `from converter import` with `from pennylane_einsum import`
- Remove the claim "The converter currently supports only 1- and 2-qubit gates" — replace with: "Any gate that implements `op.matrix()` in PennyLane is supported, including 3-qubit gates like Toffoli."
- Update **Supported Gates** section to reflect this

---

## 5. CI: GitHub Actions

File: `.github/workflows/ci.yml`

Triggers: `push` and `pull_request` on `main`

Matrix:
- Python: `3.9`, `3.11`
- OS: `ubuntu-latest`

Steps:
1. `actions/checkout@v4`
2. `actions/setup-python@v5` with matrix Python version
3. `pip install -e ".[dev]"`
4. `pytest`

---

## 6. Cleanup

- Add `*.egg-info/` and `**/__pycache__/` to `.gitignore` if not already present
- Remove `pennylane_einsum.egg-info/` from git tracking (`git rm -r --cached`)

---

## Success Criteria

- `pip install -e ".[dev]"` works cleanly after rename
- `from pennylane_einsum import CircuitToEinsum, contract_einsum` works
- `pytest` passes all existing tests
- CI workflow runs and passes on push to `main`
- README examples are accurate and runnable
