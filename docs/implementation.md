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
   tensor with input/output indices. Batched matrices keep one leading batch
   axis.
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

## Batched Tensor Layout

PennyLane parameter broadcasting produces gate matrices with shape
`(B, 2**n, 2**n)`. The converter treats `B` as one shared leading batch axis:

- The matrix is reshaped to `(B, [2] * n, [2] * n)`.
- Axes are transposed to `[batch, in..., out...]`.
- A shared batch index is prepended to the gate tensor term.
- The same batch index is prepended to the final state output.

For example, a single-qubit batched gate changes from `ef` to `ief`, and a final
two-qubit output changes from `df` to `idf`.

All batched gates in one circuit must have the same leading batch size. Fixed
gates remain unbatched and broadcast naturally in the einsum expression.

## Contraction

All contractions use `opt_einsum` (a runtime dependency). `opt_einsum` handles
arbitrarily large expressions including those with more than 52 unique index
labels that would require Unicode characters. Pass an `optimize` string to
`contract_einsum` to control the path-finding strategy (default: `"auto"`).

## Expectation Values

`generate_expectation_einsum` builds `⟨ψ|O|ψ⟩ = ⟨0|U† O U|0⟩` as one einsum with no
free output indices (a scalar; just the batch index when batched). The pieces:

- **Ket** `U|0⟩`: the same operands `generate_full_einsum` produces. The frontier
  index of qubit `q` is `final_indices[q]`.
- **Bra** `⟨0|U†`: the conjugate mirror — every ket tensor is `conj`-ed and its
  indices relabeled to fresh labels (`relabel_map`). The batch index is *not*
  relabeled, so `U` and `U†` share the same parameters per batch element.
- **Observable** `O`: per measured wire-group, a `2**k × 2**k` matrix reshaped to
  `([2]*k bra) + ([2]*k ket)` and indexed `bra_legs + ket_legs`. Unmeasured wires set
  `relabel_map[f_q] = f_q`, collapsing bra onto ket (an identity / partial trace), so
  no tensor is emitted there.

`normalize_observable` reduces any single (non-summed) observable — Pauli string,
dict, `(matrix, wires)` tuple, or a PennyLane operator — to `{wires_tuple: matrix}`.
Tensor products are split into disjoint single/multi-wire factors so the observable
tensors stay small. Weighted sums (`qml.Hamiltonian`) are expanded by
`_observable_terms` and evaluated in `expectation_value` as `Σ_i c_i ⟨O_i⟩`.

### Lightcone cancellation

With `lightcone=True`, `_causal_ops` walks the gate list backwards and keeps only
gates that can causally reach a measured wire (a kept gate pulls all its wires into
the cone). Gates outside the cone would form `g g†` identities in the sandwich, so
dropping them leaves the value unchanged. Because dropping gates breaks the
pre-allocated index chain, `_thread_ket` re-derives fresh indices over the kept
gates. The contraction path becomes observable-dependent and is not reusable across
different observables.

### Backends

If any observable matrix is a `torch.Tensor`, `_match_backend` promotes every operand
(state, gates, observable) to torch on that tensor's dtype/device, keeping `opt_einsum`
on a single backend and preserving the autograd graph on the observable. Gate tensors
become constant torch leaves — gradients reach the observable but not gate parameters.

## Supported Scope

- Fixed-width qubit operations whose `op.matrix()` returns a dense
  `(2**n, 2**n)` matrix.
- Parameter-broadcasted operations whose `op.matrix()` returns a dense
  `(B, 2**n, 2**n)` matrix.
- Common one-, two-, and multi-qubit unitary gates, including Toffoli.
- Batched `RX`, `RY`, `RZ`, `Rot`, and `CRot` are covered by tests.
- Integer wires and basic named-wire circuits, subject to the output-order caveat
  above.

## Limitations

- State preparation operations are not supported.
- Mid-circuit measurement flows inside the circuit body are not converted.
  Observables for expectation values are supported separately via
  `expectation_value` (see above), but single multi-wire dense observables are not
  factored and PennyLane observable wires must match the circuit's integer wires.
- Channels/noisy operations are not supported.
- Unsupported operations are not automatically decomposed.
- Gate parameters are materialized into NumPy arrays, so PennyLane autodiff
  through the circuit is not preserved.
- Only one shared leading batch axis is supported.
- Batched initial states are not supported; one initial state is reused for all
  batch elements.
- Unsupported operations raise `UnsupportedOperationError` with the operation
  name, wires, and original failure reason.
