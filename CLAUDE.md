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
    → contract_einsum()      # contracts via opt_einsum
```

### Key types

- **`CircuitToEinsum`** (`circuit_to_einsum.py`) — main converter. Owns an `IndexManager`. Call `for_qubits(n)` to construct. `circuit_to_einsum()` returns a data dict; `generate_full_einsum()` turns that dict into `(expr, tensors)` ready for contraction.
- **`IndexManager`** (`index_manager.py`) — allocates a unique index character per qubit state slot. Uses ASCII a–z/A–Z for the first 52 indices, then falls back to Unicode codepoints for circuits needing more.
- **`contract_einsum`** — thin wrapper around `opt_einsum.contract` (runtime dependency). Accepts an optional `optimize` string; defaults to `"auto"`. Handles arbitrarily large expressions including those needing Unicode index labels.
- **`generate_expectation_einsum` / `expectation_value`** — build `<ψ|O|ψ>` as a single einsum that contracts straight to a scalar (no `2**n` statevector intermediate). See "Expectation values" below.
- **`expval_hermitian_torch`** — computes `<ψ|H_q|ψ>` post-statevector with gradient through H (not through the circuit). Legacy path; prefer `expectation_value` for new code.

### Expectation values

`CircuitToEinsum.generate_expectation_einsum(data, observable)` returns `(expr, tensors)`
for `⟨0|U† O U|0⟩`, following cuQuantum's `CircuitToEinsum.expectation`. The bra is the
**conjugate mirror** of the forward ket network: every ket tensor is conjugated and its
indices relabeled to fresh chars (the batch index is shared, so U and U† use the same
parameters). The observable bridges the ket/bra frontiers — `O_{bra,ket}` per measured
wire; identity wires collapse bra→ket frontier (traced out). The contraction has no free
output indices (just the batch index when batched), so the result is a scalar / batch vector.

`observable` is normalized by `normalize_observable` into `{wires_tuple: 2**k×2**k matrix}`
and accepts: a Pauli string `"IZZ"`, a dict `{wire: 'Z'}` / `{wire: 2×2 matrix}` /
`{(0,1): 4×4 matrix}`, a `(matrix, wires)` tuple, or a single PennyLane observable
(`qml.PauliZ(0)`, `qml.PauliX(0) @ qml.PauliZ(1)`, `qml.Hermitian(H, wires=...)`). Weighted
sums (`qml.Hamiltonian` / `LinearCombination` / `Sum`) are handled in `expectation_value`
via `_observable_terms`, evaluated by linearity `⟨H⟩ = Σ c_i ⟨O_i⟩` (one contraction per
term). If any matrix is a torch tensor, all operands are promoted to torch (`_match_backend`)
so `opt_einsum` backprops through the observable (gate tensors stay constant — the project
does not autodiff through gate params).

`generate_expectation_einsum(..., lightcone=True)` drops gates outside the observable's
reverse causal cone (`_causal_ops`) and re-threads fresh indices over the kept gates
(`_thread_ket`, needed because dropping gates breaks the pre-allocated index chain). The
value is unchanged; the path becomes observable-dependent. `expectation_value(circuit_func,
observable, n_qubits, params=..., lightcone=...)` is the one-call wrapper. PennyLane
observable wires are assumed to be the integer wires used by the circuit.

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

Integer wires map to themselves. Named (non-integer) wires are remapped to contiguous integers via `tape.wires` order before index allocation. All wire-ordering cases (named, non-contiguous, reversed) are covered in `tests/test_wire_semantics.py`.

## TDD Approach

This project uses red-green-refactor TDD. The correctness oracle for all conversion tests is `qml.state()` on `qml.device("default.qubit", ...)`. See `_compare_state()` in `tests/test_method_one_basic.py` for the pattern. Expectation-value tests use `qml.expval(...)` as the oracle — see `tests/test_expectation.py`.

## Open Issues (from docs/review.md)

- **P2** `scripts/scan_unsupported_ops.py` uses `qml.operation.WiresEnum` which does not exist in PennyLane ≥0.44.
- **P2** No changelog or compatibility table; CI matrix covers 3.11/3.12 only.
