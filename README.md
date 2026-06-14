# PennyLane Circuit to Einsum

Convert PennyLane circuits composed of matrix-defined qubit operations into an
einsum expression and tensor list. The current output represents the final
statevector of the circuit.

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

See `docs/benchmarks.md` for the larger-circuit PennyLane conversion benchmark
and generated matplotlib plot.

## Example

```bash
uv run python examples/basic_circuits.py
```

## Unsupported Operations

- State preparation ops (`BasisState`, `StatePrep`, `QubitStateVector`).
- Operations without a matrix representation via `op.matrix()`.
- Measurements, observables, and mid-circuit measurement flows.
- Channels/noisy operations and non-unitary circuit effects.
- Automatic decomposition of templates or custom operations that do not directly
  provide a matrix.

## Experimental APIs

- `expval_hermitian_torch` is a helper for differentiating through a Hermitian
  observable tensor after the statevector has been produced; it does not provide
  gradients through the converted circuit.

## Scan Unsupported Ops

This script is currently a maintenance helper, not a stable user-facing command.

```bash
uv run python scripts/scan_unsupported_ops.py
```
