# PennyLane Circuit to Einsum

Convert PennyLane circuits composed of matrix-defined qubit operations into an
einsum expression and tensor list. The current output represents the final
statevector of the circuit.

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

Any qubit operation that provides a dense square matrix through `op.matrix()` can
be converted. This includes common fixed-size unitary gates such as H, X, Y, Z,
RX, RY, RZ, CNOT, CZ, SWAP, and Toffoli.

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

## Example

```bash
python examples/basic_circuits.py
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
python scripts/scan_unsupported_ops.py
```
