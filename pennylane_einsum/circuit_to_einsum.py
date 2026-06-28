from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pennylane as qml

from .exceptions import UnsupportedOperationError
from .index_manager import IndexManager


def _build_tape(
    circuit_func: Callable[..., Any], params: Optional[Iterable[Any]]
) -> qml.tape.QuantumTape:
    with qml.tape.QuantumTape() as tape:
        if params is None:
            circuit_func()
        else:
            circuit_func(*params)
    return tape


def _matrix_to_tensor(matrix: np.ndarray, n_wires: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=complex)
    dim = 2**n_wires
    if matrix.shape == (dim, dim):
        reshaped = matrix.reshape([2] * (2 * n_wires))
        out_axes = list(range(n_wires))
        in_axes = list(range(n_wires, 2 * n_wires))
        transpose_axes = in_axes + out_axes
        return np.transpose(reshaped, axes=transpose_axes)

    if matrix.ndim == 3 and matrix.shape[1:] == (dim, dim):
        reshaped = matrix.reshape([matrix.shape[0]] + [2] * (2 * n_wires))
        out_axes = list(range(1, n_wires + 1))
        in_axes = list(range(n_wires + 1, 2 * n_wires + 1))
        transpose_axes = [0] + in_axes + out_axes
        return np.transpose(reshaped, axes=transpose_axes)

    if matrix.ndim == 3:
        raise ValueError(
            f"Unexpected batched matrix shape {matrix.shape} for {n_wires} wires"
        )
    else:
        raise ValueError(f"Unexpected matrix shape {matrix.shape} for {n_wires} wires")


def contract_einsum(
    einsum_expr: str,
    tensors: List[np.ndarray],
    optimize: Optional[str] = None,
) -> np.ndarray:
    """Contract tensors using opt_einsum.

    Supports arbitrarily large einsum expressions including those with >52
    unique indices that require Unicode labels.
    """
    import opt_einsum as oe

    return oe.contract(einsum_expr, *tensors, optimize=optimize or "auto")


# ── Observable handling ───────────────────────────────────────────────────────

_PAULI_MATRICES = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _is_torch_tensor(obj: Any) -> bool:
    return type(obj).__module__.startswith("torch") and type(obj).__name__ == "Tensor"


def _conj(tensor: Any) -> Any:
    return tensor.conj() if _is_torch_tensor(tensor) else np.conjugate(tensor)


def _match_backend(operands: List[Any], obs: Dict[int, Any]) -> List[Any]:
    """Promote all operands to torch when any observable is a torch tensor.

    Keeps ``opt_einsum`` on a single backend (mixing numpy and torch fails) and
    preserves the autograd graph on the torch observable. Gate/state tensors become
    constant torch leaves on the observable's dtype/device.
    """
    torch_refs = [m for m in obs.values() if _is_torch_tensor(m)]
    if not torch_refs:
        return operands

    import torch

    ref = torch_refs[0]
    converted: List[Any] = []
    for op in operands:
        if _is_torch_tensor(op):
            converted.append(op.to(dtype=ref.dtype, device=ref.device))
        else:
            converted.append(
                torch.as_tensor(np.asarray(op), dtype=ref.dtype, device=ref.device)
            )
    return converted


def normalize_observable(
    observable: Any,
    n_qubits: int,
    wire_order: Optional[Iterable[int]] = None,
) -> Dict[int, Any]:
    """Normalize an observable spec into ``{wire: 2x2 matrix}`` for measured wires.

    Accepted forms:
      - Pauli string ``"IZZ"`` — one char per wire in ``wire_order`` (default
        ``range(n_qubits)``). ``'I'`` entries are dropped.
      - dict of Pauli chars ``{0: 'Z', 2: 'X'}`` — ``'I'`` entries dropped.
      - dict of 2x2 matrices ``{wire: H}`` — single-qubit Hermitians, passed through
        (numpy arrays or torch tensors).
      - tuple ``(matrix, wires)`` — a single 2x2 matrix applied to each wire.

    Identity (unmeasured) wires are omitted from the result; the caller traces them
    out by collapsing bra/ket frontier indices.
    """
    wires = list(range(n_qubits)) if wire_order is None else list(wire_order)

    if isinstance(observable, str):
        if len(observable) != n_qubits:
            raise ValueError(
                f"Pauli string '{observable}' has length {len(observable)}, "
                f"expected {n_qubits}"
            )
        observable = dict(zip(wires, observable))

    if isinstance(observable, tuple):
        matrix, op_wires = observable
        op_wires = [op_wires] if isinstance(op_wires, int) else list(op_wires)
        observable = {w: matrix for w in op_wires}

    if not isinstance(observable, dict):
        raise TypeError(
            "observable must be a Pauli string, a dict, or a (matrix, wires) tuple; "
            f"got {type(observable).__name__}"
        )

    result: Dict[int, Any] = {}
    for wire, spec in observable.items():
        if isinstance(spec, str):
            if spec not in _PAULI_MATRICES:
                raise ValueError(f"Unknown Pauli character '{spec}' on wire {wire}")
            if spec == "I":
                continue
            result[wire] = _PAULI_MATRICES[spec]
        else:
            if not _is_torch_tensor(spec):
                spec = np.asarray(spec, dtype=complex)
            if tuple(spec.shape) != (2, 2):
                raise ValueError(
                    f"Observable on wire {wire} must be 2x2; got shape {tuple(spec.shape)}"
                )
            result[wire] = spec
    return result



def expval_hermitian_torch(
    statevec: np.ndarray,
    H: "torch.Tensor",
    qubit: int,
    n_qubits: int,
) -> "torch.Tensor":
    """Expectation value <psi|H_q|psi> with autograd through H.

    The statevec (output of contract_einsum) is treated as a constant —
    gradients do not flow through the circuit, only through H.
    H is typically a slice of an nn.Parameter (e.g. LearnablePauliDirection).

    Args:
        statevec: flat numpy statevector of length 2**n_qubits
        H: complex torch tensor of shape (2, 2); gradient-enabled
        qubit: index of the qubit on which H acts (0-based)
        n_qubits: total number of qubits

    Returns:
        Real torch scalar; supports .backward()

    Example (GPU-transparent — works on CPU or CUDA)::

        sv = contract_einsum(expr, tensors).flatten()
        ev = expval_hermitian_torch(sv, obs_module()[q], q, n_qubits)
        loss = some_loss(ev)
        loss.backward()   # gradients reach obs_module.parameters()
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for expval_hermitian_torch; install it with: pip install torch"
        ) from exc

    sv = torch.as_tensor(statevec, dtype=H.dtype, device=H.device)
    state = sv.reshape([2] * n_qubits)
    state = torch.movedim(state, qubit, 0).reshape(2, -1)
    rho = state @ state.conj().T
    return torch.trace(rho @ H).real


@dataclass
class CircuitToEinsum:
    n_qubits: int
    index_manager: IndexManager
    batch_size: Optional[int] = None
    batch_index: Optional[str] = None

    @classmethod
    def for_qubits(cls, n_qubits: int) -> "CircuitToEinsum":
        return cls(n_qubits=n_qubits, index_manager=IndexManager(n_qubits=n_qubits))

    def _wire_map(self, wires: Iterable[Any]) -> Dict[Any, int]:
        wire_list = list(wires)
        if all(isinstance(w, int) for w in wire_list):
            return {int(w): int(w) for w in wire_list}
        return {wire: idx for idx, wire in enumerate(wire_list)}

    def _record_batch_size(self, batch_size: int) -> None:
        if self.batch_size is None:
            self.batch_size = batch_size
            self.batch_index = self.index_manager.fresh_index()
        elif self.batch_size != batch_size:
            raise ValueError(
                "All batched gate matrices must share the same leading batch "
                f"size; got {batch_size} after {self.batch_size}"
            )

    def _apply_gate(
        self, gate_matrix: np.ndarray, wires: List[int]
    ) -> Tuple[str, np.ndarray]:
        n_wires = len(wires)
        if n_wires < 1:
            raise NotImplementedError("Gates with 0 wires not implemented")

        in_indices = [self.index_manager.get(w) for w in wires]
        out_indices = [self.index_manager.fresh_index() for _ in wires]

        for w, out_idx in zip(wires, out_indices):
            self.index_manager.set(w, out_idx)

        tensor = _matrix_to_tensor(gate_matrix, n_wires)
        batch_prefix = ""
        if tensor.ndim == 2 * n_wires + 1:
            self._record_batch_size(tensor.shape[0])
            if self.batch_index is None:
                raise RuntimeError("batch_index was not initialized")
            batch_prefix = self.batch_index

        in_str = "".join(in_indices)
        out_str = "".join(out_indices)
        einsum_str = f"{in_str},{batch_prefix}{in_str}{out_str}"
        return einsum_str, tensor

    def circuit_to_einsum(
        self,
        circuit_func: Callable[..., Any],
        params: Optional[Iterable[Any]] = None,
    ) -> Dict[str, Any]:
        self.index_manager.counter = 0
        self.batch_size = None
        self.batch_index = None
        init_indices = self.index_manager.init_qubits()

        tape = _build_tape(circuit_func, params)

        operations: List[Dict[str, Any]] = []
        for op in tape.operations:
            n_wires = len(op.wires)
            if n_wires < 1:
                raise NotImplementedError("Gates with 0 wires not implemented")

            try:
                gate_matrix = op.matrix()
            except Exception as exc:
                raise UnsupportedOperationError(
                    op_name=op.name,
                    wires=list(op.wires),
                    reason=str(exc),
                ) from exc
            wires = list(op.wires)
            if not wires:
                raise ValueError("Operation has no wires")

            if not all(isinstance(w, int) for w in wires):
                wire_map = self._wire_map(tape.wires)
                wires = [wire_map[w] for w in wires]

            einsum_str, tensor = self._apply_gate(gate_matrix, wires)
            operations.append(
                {
                    "einsum": einsum_str,
                    "tensor": tensor,
                    "gate_name": op.name,
                    "wires": wires,
                    "num_params": op.num_params,
                    "op_class": type(op),
                    "op_params": list(op.parameters),
                }
            )

        return {
            "initial_indices": init_indices,
            "operations": operations,
            "final_indices": self.index_manager.snapshot(),
            "batch_size": self.batch_size,
            "batch_index": self.batch_index,
        }

    def _ket_operands(
        self,
        einsum_data: Dict[str, Any],
        initial_state: Optional[np.ndarray] = None,
    ) -> List[Tuple[str, np.ndarray]]:
        """Build the forward (ket) tensor network ``U|0…0⟩`` as ``[(index_str, tensor)]``.

        The first entry is the initial state; the rest are the gate tensors in
        application order. The free (frontier) index of qubit ``q`` after the last
        gate is ``einsum_data["final_indices"][q]``.
        """
        n_qubits = self.n_qubits

        if initial_state is None:
            state = np.zeros(2**n_qubits, dtype=complex)
            state[0] = 1.0
        else:
            state = np.asarray(initial_state, dtype=complex)
            if state.shape != (2**n_qubits,):
                raise ValueError(
                    "initial_state must be a flat statevector of length 2**n_qubits"
                )

        init_idx_str = "".join(einsum_data["initial_indices"].values())
        terms: List[Tuple[str, np.ndarray]] = [(init_idx_str, state.reshape([2] * n_qubits))]
        for op in einsum_data["operations"]:
            gate_str = op["einsum"].split(",")[1]
            terms.append((gate_str, op["tensor"]))
        return terms

    def generate_full_einsum(
        self,
        einsum_data: Dict[str, Any],
        initial_state: Optional[np.ndarray] = None,
    ) -> Tuple[str, List[np.ndarray]]:
        terms = self._ket_operands(einsum_data, initial_state)
        idx_strs = [idx for idx, _ in terms]
        tensors = [tensor for _, tensor in terms]
        final_idx_str = "".join(einsum_data["final_indices"].values())
        if einsum_data.get("batch_index") is not None:
            final_idx_str = f"{einsum_data['batch_index']}{final_idx_str}"
        full_einsum = f"{','.join(idx_strs)}->{final_idx_str}"
        return full_einsum, tensors

    def generate_expectation_einsum(
        self,
        einsum_data: Dict[str, Any],
        observable: Any,
        initial_state: Optional[np.ndarray] = None,
    ) -> Tuple[str, List[Any]]:
        """Build ``⟨ψ|O|ψ⟩`` as a single einsum that contracts to a scalar.

        Mirrors cuQuantum's ``expectation``: the ket ``U|0⟩``, the observable ``O``,
        and the bra ``⟨0|U†`` (the conjugate mirror of the ket with relabeled
        indices) are assembled into one network with no free output indices (or just
        the batch index when the circuit is batched).

        ``observable`` accepts any form understood by :func:`normalize_observable`.
        If the observable contains torch tensors, all operands are converted to torch
        so ``opt_einsum`` can backpropagate through it.
        """
        ket_terms = self._ket_operands(einsum_data, initial_state)
        batch_index = einsum_data.get("batch_index")
        final_indices = einsum_data["final_indices"]

        obs = normalize_observable(observable, self.n_qubits)
        measured = set(obs.keys())

        ket_chars = set()
        for idx_str, _ in ket_terms:
            ket_chars.update(idx_str)
        if batch_index is not None:
            ket_chars.discard(batch_index)

        # Fresh bra label per ket char; identity wires collapse onto the ket frontier.
        relabel_map = {c: self.index_manager.fresh_index() for c in sorted(ket_chars)}
        for q, f_q in final_indices.items():
            if q not in measured:
                relabel_map[f_q] = f_q

        def relabel(s: str) -> str:
            return "".join(c if c == batch_index else relabel_map[c] for c in s)

        idx_strs: List[str] = []
        operands: List[Any] = []

        for idx_str, tensor in ket_terms:  # ket: U|0⟩
            idx_strs.append(idx_str)
            operands.append(tensor)

        for idx_str, tensor in ket_terms:  # bra: conj(U|0⟩) = ⟨0|U†
            idx_strs.append(relabel(idx_str))
            operands.append(_conj(tensor))

        for q, matrix in obs.items():  # observable: O_{bra, ket}
            f_q = final_indices[q]
            idx_strs.append(f"{relabel_map[f_q]}{f_q}")
            operands.append(matrix)

        operands = _match_backend(operands, obs)
        out_str = batch_index if batch_index is not None else ""
        expr = f"{','.join(idx_strs)}->{out_str}"
        return expr, operands


def expectation_value(
    circuit_func: Callable[..., Any],
    observable: Any,
    n_qubits: int,
    params: Optional[Iterable[Any]] = None,
    initial_state: Optional[np.ndarray] = None,
    optimize: Optional[str] = None,
    return_real: bool = True,
) -> Any:
    """Compute ``⟨ψ|O|ψ⟩`` for ``|ψ⟩ = U|0…0⟩`` via a single tensor-network contraction.

    Convenience wrapper around :meth:`CircuitToEinsum.generate_expectation_einsum`.
    Returns a scalar (a length-``batch`` vector for batched circuits). For a Hermitian
    observable the value is real; ``return_real`` takes ``.real`` to match
    ``qml.expval`` semantics while preserving torch gradients.
    """
    converter = CircuitToEinsum.for_qubits(n_qubits)
    data = converter.circuit_to_einsum(circuit_func, params=params)
    expr, tensors = converter.generate_expectation_einsum(data, observable, initial_state)
    value = contract_einsum(expr, tensors, optimize=optimize)
    return value.real if return_real else value
