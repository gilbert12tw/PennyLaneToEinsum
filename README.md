# PennyLane Circuit to Einsum

Convert PennyLane circuits composed of matrix-defined qubit operations into an
einsum expression and tensor list. The output is either the final **statevector**
(`generate_full_einsum`) or an **expectation value** `⟨ψ|O|ψ⟩`
(`expectation_value` / `generate_expectation_einsum`), the latter contracted as a
single tensor network without materializing the `2**n` state.

## Install (dev)

```bash
uv sync --extra dev
uv run --extra dev pytest -q
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

## Batched Gate Parameters

PennyLane parameter broadcasting is supported when `op.matrix()` returns a
leading batch axis. For example, `qml.RX(np.array([0.1, 0.2, 0.3]), wires=0)`
produces three statevectors in one einsum expression.

```python
import numpy as np
import pennylane as qml

from pennylane_einsum import CircuitToEinsum, contract_einsum

theta = np.array([0.1, 0.2, 0.3])

def circuit():
    qml.Hadamard(wires=0)
    qml.RX(theta, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(theta + 0.5, wires=1)

converter = CircuitToEinsum.for_qubits(2)
data = converter.circuit_to_einsum(circuit)
expr, tensors = converter.generate_full_einsum(data)

state_tensor = contract_einsum(expr, tensors)
statevectors = state_tensor.reshape(data["batch_size"], -1)
print(statevectors.shape)  # (3, 4)
```

Batch support uses one shared leading batch index. Batched gate tensors use
indices like `ief`, and the final output gets the same leading index, e.g.
`idf`. Fixed gates are unbatched and broadcast across the batch. All batched
operations in one circuit must have the same batch size.

See `docs/batching.md` for supported shapes, limitations, and test coverage.

## Expectation Values

Compute `⟨ψ|O|ψ⟩` for `|ψ⟩ = U|0…0⟩` as a single einsum that contracts straight to
a scalar. Following NVIDIA cuQuantum's `CircuitToEinsum.expectation`, the network is
the sandwich `⟨0|U† O U|0⟩`: the forward circuit `U` (ket), the observable `O`, and
the bra `⟨0|U†` built as the **conjugate mirror** of the ket (each gate conjugated,
indices relabeled). There is no `2**n` statevector intermediate.

```python
import pennylane as qml
from pennylane_einsum import expectation_value

def bell():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])

expectation_value(bell, "ZZ", n_qubits=2)            # 1.0
```

`observable` accepts several forms:

```python
expectation_value(circ, "IZX", n_qubits=3)              # Pauli string ('I' = unmeasured)
expectation_value(circ, {0: "Z", 2: "X"}, n_qubits=3)   # dict of Pauli chars
expectation_value(circ, {1: H2x2}, n_qubits=2)          # per-wire Hermitian
expectation_value(circ, (H4x4, [0, 1]), n_qubits=2)     # (matrix, wires)
expectation_value(circ, qml.PauliX(0) @ qml.PauliZ(1), n_qubits=2)      # PennyLane obs
expectation_value(circ, qml.Hermitian(H4x4, wires=[0, 1]), n_qubits=2)  # multi-wire
expectation_value(circ, qml.Hamiltonian([0.5, -1.2], [qml.PauliZ(0),    # weighted sum
                                                      qml.PauliX(0) @ qml.PauliZ(1)]),
                  n_qubits=2)                            # evaluated by linearity
```

`qml.Hamiltonian` (and `Sum` / `LinearCombination`) are evaluated term-by-term as
`⟨H⟩ = Σ_i c_i ⟨O_i⟩`. Batched circuits return a length-`batch` vector of expectation
values.

**Lightcone cancellation.** Pass `lightcone=True` to drop gates outside the
observable's reverse causal cone (they cancel against their inverse in the bra). The
value is unchanged but the network is smaller; the contraction path then depends on
the observable, so it is not reusable across different observables.

```python
expectation_value(circ, {3: "Z"}, n_qubits=8, lightcone=True)
```

**Gradients (torch).** If any observable matrix is a `torch.Tensor`, all operands are
promoted to torch so `opt_einsum` backpropagates through the observable. Gate tensors
stay constant — the converter does not autodiff through gate parameters.

```python
import torch
H = torch.tensor([[0.3, 0.1j], [-0.1j, -0.3]], dtype=torch.complex128, requires_grad=True)
ev = expectation_value(circ, {0: H}, n_qubits=2)
ev.backward()          # H.grad is populated
```

The lower-level `converter.generate_expectation_einsum(data, observable,
lightcone=...)` returns `(expr, tensors)` if you want to inspect or contract the
network yourself.

## Supported Gates

Any qubit operation that provides a dense square matrix through `op.matrix()` can
be converted. This includes common fixed-size unitary gates such as H, X, Y, Z,
RX, RY, RZ, CNOT, CZ, SWAP, and Toffoli.

Parameterized gates with PennyLane broadcasting are supported when their dense
matrix has shape `(B, 2**n, 2**n)`. Current tests cover batched `RX`, `RY`,
`RZ`, `Rot`, and `CRot`, including mixed fixed/batched circuits and mismatched
batch-size errors.

The converter is intentionally dense-matrix based. It does not decompose
unsupported operations automatically and does not preserve PennyLane autodiff
through gate parameters.

## Tests

```bash
uv run --extra dev pytest -q
```

## Implementation Notes

See `docs/implementation.md` for a walkthrough of the indexing strategy,
tensor layout, and current limitations.

See `docs/benchmarks.md` for the larger-circuit conversion benchmark, batch
contraction benchmark, PennyLane comparison, and generated matplotlib plots.

## Example

```bash
uv run python examples/basic_circuits.py
```

## Unsupported Operations

- State preparation ops (`BasisState`, `StatePrep`, `QubitStateVector`).
- Operations without a matrix representation via `op.matrix()`.
- Mid-circuit measurement flows.
- Channels/noisy operations and non-unitary circuit effects.
- Automatic decomposition of templates or custom operations that do not directly
  provide a matrix.

Observables for expectation values *are* supported via `expectation_value` (see
above); the circuit body itself still may not contain measurement ops.

## Experimental APIs

- `expectation_value` / `generate_expectation_einsum` compute `⟨ψ|O|ψ⟩` as a tensor
  network (see "Expectation Values"), with optional lightcone pruning and torch
  gradients through the observable.
- `expval_hermitian_torch` is an older helper for differentiating through a single
  Hermitian observable *after* the statevector has been produced. Prefer
  `expectation_value` for new code; neither provides gradients through gate
  parameters.

## Scan Unsupported Ops

This script is currently a maintenance helper, not a stable user-facing command.

```bash
uv run python scripts/scan_unsupported_ops.py
```
