# Batched Gate Parameters

The converter supports PennyLane parameter broadcasting when an operation's
`op.matrix()` returns a dense matrix with a leading batch axis.

## Semantics

Supported matrix shapes are:

- Unbatched gate: `(2**n, 2**n)`
- Batched gate: `(B, 2**n, 2**n)`

For a batched gate, the leading axis is represented as a shared einsum index.
For example, a single-qubit gate tensor changes from `ef` to `ief`, and the
final state output changes from `df` to `idf`.

Fixed gates do not receive a batch index. They broadcast across the shared batch
axis created by parameterized gates.

## Example

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

## Supported Scope

- Any operation whose `op.matrix()` returns either a dense square matrix or a
  dense matrix with one leading batch axis.
- All batched operations in one circuit must use the same leading batch size.
- Mixed fixed and batched gates are supported.
- Multi-parameter gates are supported when PennyLane broadcasts them to the same
  leading batch size.
- Multi-qubit parameterized gates are supported when their dense matrix follows
  the same `(B, 2**n, 2**n)` convention.

## Limitations

- Only one shared leading batch axis is supported.
- Batched initial states are not supported; one initial state is reused across
  all batch elements.
- Gate parameters are materialized as NumPy arrays, so autodiff through circuit
  parameters is not preserved.
- Measurements and observables are still outside the converter output; the
  converter returns final statevectors.

## Tested Coverage

The test suite compares each batch element against an independent PennyLane
`qml.state()` execution. Current coverage includes:

- `RX`, `RY`, and `RZ` with batch sizes `1`, `2`, and `5`
- `Rot` with multiple batched parameters
- `CRot` as a two-qubit parameterized batched gate
- Mixed fixed and batched gates
- Mismatched batch sizes raising `ValueError`
