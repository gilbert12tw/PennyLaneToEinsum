# Implementation Notes

## Overview

The converter builds a `QuantumTape` from a PennyLane circuit, walks through the tape's
operations, and maps each operation to a tensor plus an einsum term. It then combines
the initial state, all gate tensors, and final indices into a single einsum expression.

The current implementation is a dense statevector converter. It is designed to
extract a tensor network representation for matrix-defined qubit operations, not
to be a full PennyLane compiler.

## Data Flow

1. `CircuitToEinsum.circuit_to_einsum` builds a tape and collects operations.
2. Each operation provides a dense matrix via `op.matrix()`.
3. The matrix is converted to a NumPy complex array and reshaped into a rank-2n
   tensor with input/output indices.
4. An `IndexManager` allocates fresh index labels for each qubit as gates are applied.
5. `generate_full_einsum` assembles the full einsum expression and tensors list.

## Indexing Strategy

- Each qubit starts with a unique index label.
- Applying a gate replaces the input index with a fresh output index.
- The einsum term for a gate is built as:
  - `in_indices, in_indices + out_indices`
  - Example: a single-qubit gate on index `a` produces `a,ab`.

## Tensor Layout

For an n-qubit gate, the matrix has shape `(2**n, 2**n)`. It is reshaped into:

- `([2] * n) + ([2] * n)` and transposed to `[in..., out...]`
- This results in a tensor with axes ordered as inputs first, outputs second.

This layout matches the einsum string `in_indices, in_indices + out_indices`.

The wire order used for the gate tensor is the operation's own wire order. For
non-integer PennyLane wires, the converter maps `tape.wires` to contiguous
integer positions before assigning indices.

## Contraction

The converter can contract with:

- `numpy.einsum` by default
- `opt_einsum` when installed, when `optimize` is provided, or when a large
  expression needs Unicode index labels beyond NumPy's usual ASCII range

For large circuits, install the development extra or install `opt_einsum`
explicitly.

## Supported Scope

- Fixed-width qubit operations whose `op.matrix()` returns a dense
  `(2**n, 2**n)` matrix.
- Common one-, two-, and multi-qubit unitary gates, including Toffoli.
- Integer wires and basic named-wire circuits, subject to the output-order caveat
  above.

## Limitations

- State preparation operations are not supported.
- Measurements, observables, and mid-circuit measurement flows are not converted.
- Channels/noisy operations are not supported.
- Unsupported operations are not automatically decomposed.
- Gate parameters are materialized into NumPy arrays, so PennyLane autodiff
  through the circuit is not preserved.
- `build_batch_einsum` is experimental and currently incomplete because batch
  dimensions are not represented in the generated einsum expression.
- Error messages for unsupported operations are currently raw PennyLane/NumPy
  exceptions in several cases.
