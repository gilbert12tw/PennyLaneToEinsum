# pennylane-einsum Package Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the `converter` module to `pennylane_einsum`, fix package metadata, update all imports, fix the README, add GitHub Actions CI, and remove stale egg-info from git.

**Architecture:** Pure mechanical rename + metadata fix. No new logic. The existing `CircuitToEinsum` / `IndexManager` / `contract_einsum` implementation is unchanged — only the module name, package metadata, and ancillary files are updated.

**Tech Stack:** Python 3.9+, PennyLane ≥0.37, NumPy ≥1.23, pytest, setuptools, GitHub Actions

---

## File Map

| Action | Path |
|--------|------|
| Rename (git mv) | `converter/` → `pennylane_einsum/` |
| Modify | `pennylane_einsum/__init__.py` |
| Modify | `pyproject.toml` |
| Modify | `tests/test_method_one_basic.py` |
| Modify | `tests/test_multiqubit_gates.py` |
| Modify | `tests/test_unsupported_ops.py` |
| Modify | `examples/basic_circuits.py` |
| Modify | `README.md` |
| Create | `.github/workflows/ci.yml` |
| Remove from git | `pennylane_einsum.egg-info/` |

---

## Task 1: Rename the module directory

**Files:**
- Rename: `converter/` → `pennylane_einsum/` (via `git mv`)

- [ ] **Step 1: Rename the directory with git**

```bash
git mv converter pennylane_einsum
```

Expected: no output, exit 0.

- [ ] **Step 2: Verify the rename**

```bash
ls pennylane_einsum/
```

Expected output (order may vary):
```
__init__.py  circuit_to_einsum.py  index_manager.py
```

---

## Task 2: Update `__init__.py` to add `__version__` and export `IndexManager`

**Files:**
- Modify: `pennylane_einsum/__init__.py`

The renamed file currently contains:
```python
from .circuit_to_einsum import CircuitToEinsum, contract_einsum

__all__ = ["CircuitToEinsum", "contract_einsum"]
```

- [ ] **Step 1: Replace the content of `pennylane_einsum/__init__.py`**

Write this exact content:

```python
from .circuit_to_einsum import CircuitToEinsum, contract_einsum
from .index_manager import IndexManager

__version__ = "0.1.0"
__all__ = ["CircuitToEinsum", "contract_einsum", "IndexManager", "__version__"]
```

- [ ] **Step 2: Verify the import works**

```bash
python -c "from pennylane_einsum import CircuitToEinsum, contract_einsum, IndexManager, __version__; print(__version__)"
```

Expected output: `0.1.0`

- [ ] **Step 3: Commit**

```bash
git add pennylane_einsum/
git commit --no-gpg-sign -m "refactor: rename converter/ to pennylane_einsum/, add __version__"
```

---

## Task 3: Fix `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Replace `pyproject.toml` with the corrected version**

Write this exact content:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pennylane-einsum"
version = "0.1.0"
description = "Convert PennyLane circuits to einsum expressions."
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}
authors = [{name = "gilbert", email = "gilbert12.tw@gmail.com"}]
classifiers = [
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Topic :: Scientific/Engineering :: Physics",
]
dependencies = [
  "pennylane>=0.37",
  "numpy>=1.23",
]

[project.optional-dependencies]
dev = [
  "pytest>=7.0",
  "opt_einsum>=3.3.0",
]

[project.urls]
Homepage = "https://github.com/gilbert/pennylane-einsum"
Repository = "https://github.com/gilbert/pennylane-einsum"

[tool.setuptools.packages.find]
where = ["."]
include = ["pennylane_einsum*"]
```

- [ ] **Step 2: Reinstall the package so setuptools picks up the new config**

```bash
pip install -e ".[dev]"
```

Expected: output ends with `Successfully installed pennylane-einsum-0.1.0` (or "already satisfied" lines followed by install).

- [ ] **Step 3: Verify package is importable after reinstall**

```bash
python -c "import pennylane_einsum; print(pennylane_einsum.__version__)"
```

Expected output: `0.1.0`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit --no-gpg-sign -m "fix: update pyproject.toml metadata and package discovery"
```

---

## Task 4: Update test imports

**Files:**
- Modify: `tests/test_method_one_basic.py`
- Modify: `tests/test_multiqubit_gates.py`
- Modify: `tests/test_unsupported_ops.py`

- [ ] **Step 1: Update `tests/test_method_one_basic.py`**

Change line 4 from:
```python
from converter import CircuitToEinsum
```
to:
```python
from pennylane_einsum import CircuitToEinsum
```

- [ ] **Step 2: Update `tests/test_multiqubit_gates.py`**

Change line 4 from:
```python
from converter import CircuitToEinsum
```
to:
```python
from pennylane_einsum import CircuitToEinsum
```

- [ ] **Step 3: Update `tests/test_unsupported_ops.py`**

Change line 4 from:
```python
from converter import CircuitToEinsum
```
to:
```python
from pennylane_einsum import CircuitToEinsum
```

- [ ] **Step 4: Run the full test suite**

```bash
pytest -v
```

Expected: all tests pass. You should see output like:
```
tests/test_method_one_basic.py::test_single_qubit_gates PASSED
tests/test_method_one_basic.py::test_two_qubit_entangling PASSED
tests/test_method_one_basic.py::test_parameterized_circuit PASSED
tests/test_method_one_basic.py::test_three_qubit_chain PASSED
tests/test_multiqubit_gates.py::test_toffoli_state_matches PASSED
tests/test_unsupported_ops.py::test_three_qubit_gate_supported PASSED
tests/test_unsupported_ops.py::test_state_prep_not_supported PASSED
```

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit --no-gpg-sign -m "fix: update test imports to pennylane_einsum"
```

---

## Task 5: Update `examples/basic_circuits.py`

**Files:**
- Modify: `examples/basic_circuits.py`

- [ ] **Step 1: Update the import in `examples/basic_circuits.py`**

Change line 4 from:
```python
from converter import CircuitToEinsum, contract_einsum
```
to:
```python
from pennylane_einsum import CircuitToEinsum, contract_einsum
```

- [ ] **Step 2: Run the example to verify it works**

```bash
python examples/basic_circuits.py
```

Expected: prints three statevectors and `All close: True`. No errors.

- [ ] **Step 3: Commit**

```bash
git add examples/basic_circuits.py
git commit --no-gpg-sign -m "fix: update example import to pennylane_einsum"
```

---

## Task 6: Fix the README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the full content of `README.md`**

Write this exact content:

```markdown
# PennyLane Circuit to Einsum

Convert PennyLane circuits into einsum expressions and tensors, so you can contract them with `numpy`, `opt_einsum`, or `cotengra`.

## Install (dev)

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import pennylane as qml
from pennylane_einsum import CircuitToEinsum, contract_einsum

def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RZ(0.5, wires=1)

converter = CircuitToEinsum.for_qubits(2)
einsum_data = converter.circuit_to_einsum(circuit)
expr, tensors = converter.generate_full_einsum(einsum_data)

result = contract_einsum(expr, tensors, optimize="optimal")
print(result.reshape(-1))
```

## Supported Gates

Any gate that implements `op.matrix()` in PennyLane is supported, including
multi-qubit gates like Toffoli. This covers all standard single- and multi-qubit
unitary gates (H, X, Y, Z, RX, RY, RZ, CNOT, CZ, SWAP, Toffoli, and more).

## Tests

```bash
pytest
```

## Implementation Notes

See `docs/implementation.md` for a walkthrough of the indexing strategy,
tensor layout, and current limitations.

## Example

```bash
python examples/basic_circuits.py
```

## Unsupported Operations

- State preparation ops (`BasisState`, `StatePrep`, `QubitStateVector`).
- Operations without a matrix representation via `op.matrix()`.
- Measurements, observables, and mid-circuit measurement flows.
- Operations with `num_wires = AnyWires` unless they provide a matrix for a fixed wire count.

## Scan Unsupported Ops

```bash
python scripts/scan_unsupported_ops.py
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit --no-gpg-sign -m "docs: update README — correct module name and gate support claim"
```

---

## Task 7: Add GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create the workflow directory**

```bash
mkdir -p .github/workflows
```

- [ ] **Step 2: Create `.github/workflows/ci.yml`**

Write this exact content:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.11"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run tests
        run: pytest
```

- [ ] **Step 3: Commit**

```bash
git add .github/
git commit --no-gpg-sign -m "ci: add GitHub Actions workflow for pytest on Python 3.9 and 3.11"
```

---

## Task 8: Remove stale egg-info from git tracking

**Files:**
- Remove from git: `pennylane_einsum.egg-info/`

- [ ] **Step 1: Untrack the egg-info directory**

```bash
git rm -r --cached pennylane_einsum.egg-info/
```

Expected: several lines of `rm 'pennylane_einsum.egg-info/...'`

- [ ] **Step 2: Verify `.gitignore` already covers it**

```bash
grep egg-info .gitignore
```

Expected output includes: `*.egg-info/`

- [ ] **Step 3: Commit**

```bash
git commit --no-gpg-sign -m "chore: remove egg-info from git tracking (already in .gitignore)"
```

---

## Verification

After all tasks are complete, run the full suite one final time:

```bash
pytest -v
```

All 7 tests should pass. Then verify the importable package:

```bash
python -c "
from pennylane_einsum import CircuitToEinsum, contract_einsum, IndexManager, __version__
print('version:', __version__)
print('all good')
"
```

Expected:
```
version: 0.1.0
all good
```
