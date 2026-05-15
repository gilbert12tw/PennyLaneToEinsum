# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Before Every Commit

Always run the full test suite and confirm it is green before committing:

```bash
uv run --extra dev pytest -q
```

## Commands

```bash
# Install dev dependencies (preferred — uses uv lockfile)
uv run --extra dev pytest -q

# Or with pip
pip install -e ".[dev]"
pytest -q

# Run a single test file
uv run --extra dev pytest tests/test_method_one_basic.py -q

# Run a single test
uv run --extra dev pytest tests/test_method_one_basic.py::test_two_qubit_entangling -q

# Run example
python examples/basic_circuits.py
```

## Architecture

The package converts PennyLane circuits to einsum tensor networks. It is **dense-matrix based** — it extracts `op.matrix()` from every operation, does not decompose templates, and does not preserve autodiff through gate parameters.

### Data flow

```
circuit_func
    → _build_tape()          # wraps in qml.tape.QuantumTape()
    → tape.operations        # list of PennyLane Operation objects
    → op.matrix()            # each op provides its (2**n × 2**n) dense matrix
    → _matrix_to_tensor()    # reshape + transpose → rank-2n tensor, axes [in..., out...]
    → IndexManager           # allocates fresh index label per qubit per gate
    → circuit_to_einsum()    # returns dict: initial_indices, operations[], final_indices
    → generate_full_einsum() # assembles full einsum string + tensor list
    → contract_einsum()      # contracts via numpy.einsum or opt_einsum
```

### Key types

- **`CircuitToEinsum`** (`circuit_to_einsum.py`) — main converter. Owns an `IndexManager`. Call `for_qubits(n)` to construct. `circuit_to_einsum()` returns a data dict; `generate_full_einsum()` turns that dict into `(expr, tensors)` ready for contraction.
- **`IndexManager`** (`index_manager.py`) — allocates a unique index character per qubit state slot. Uses ASCII a–z/A–Z for the first 52 indices, then falls back to Unicode codepoints for circuits needing more.
- **`contract_einsum`** — thin wrapper around `opt_einsum.contract` (runtime dependency). Accepts an optional `optimize` string; defaults to `"auto"`. Handles arbitrarily large expressions including those needing Unicode index labels.
- **`expval_hermitian_torch`** — computes `<ψ|H_q|ψ>` post-statevector with gradient through H (not through the circuit).

### Einsum index convention

Each qubit starts with a fresh index. A gate on wires `[w0, w1]` with current indices `ab`:
- consumes `ab` (input)
- allocates fresh `cd` (output)
- produces einsum term `ab,abcd` (state slice, gate tensor)
- updates qubit index map: `w0→c, w1→d`

The final expression joins initial state, all gate terms, and output indices:
```
init_str, gate0_str, gate1_str, ... -> final_str
```

### Wire handling

Integer wires map to themselves. Named (non-integer) wires are remapped to contiguous integers via `tape.wires` order before index allocation. Non-contiguous or reversed wire ordering is a known gap in test coverage.

## TDD Approach

This project uses red-green-refactor TDD. The correctness oracle for all conversion tests is `qml.state()` on `qml.device("default.qubit", ...)`. See `_compare_state()` in `tests/test_method_one_basic.py` for the pattern.

## Open Issues (from docs/review.md)

- **P1** Missing wire-semantic tests: named wires, non-contiguous integers, reversed multi-qubit order.
- **P2** `scripts/scan_unsupported_ops.py` uses `qml.operation.WiresEnum` which does not exist in PennyLane ≥0.44.
- **P2** No changelog or compatibility table; CI matrix covers 3.11/3.12 only.
