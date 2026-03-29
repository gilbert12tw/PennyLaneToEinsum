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
