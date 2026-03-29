# Implementation Notes

## Overview

The converter builds a `QuantumTape` from a PennyLane circuit, walks through the tape's
operations, and maps each operation to a tensor plus an einsum term. It then combines
the initial state, all gate tensors, and final indices into a single einsum expression.

## Data Flow

1. `CircuitToEinsum.circuit_to_einsum` builds a tape and collects operations.
2. Each operation provides a dense matrix via `op.matrix()`.
3. The matrix is reshaped into a rank-2n tensor with input/output indices.
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

## Contraction

The converter can contract with:

- `numpy.einsum` by default
- `opt_einsum` if installed and `optimize` is provided

## Limitations (Current MVP)

- Only 1- and 2-qubit gates are supported.
- Operations must implement `op.matrix()`; decompositions are not expanded.
- State preparation operations are not supported.
- Measurements and observables are not converted; the output is a statevector.

