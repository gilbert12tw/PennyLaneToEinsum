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

    sv = torch.as_tensor(statevec, dtype=H.dtype).to(H.device)
    I2 = torch.eye(2, dtype=H.dtype, device=H.device)
    full_op = torch.ones(1, 1, dtype=H.dtype, device=H.device)
    for q in range(n_qubits):
        full_op = torch.kron(full_op, H if q == qubit else I2)
    return (sv.conj() @ full_op @ sv).real


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

    def generate_full_einsum(
        self,
        einsum_data: Dict[str, Any],
        initial_state: Optional[np.ndarray] = None,
    ) -> Tuple[str, List[np.ndarray]]:
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

        tensors: List[np.ndarray] = [state.reshape([2] * n_qubits)]
        init_idx_str = "".join(einsum_data["initial_indices"].values())
        gate_strs = [op["einsum"].split(",")[1] for op in einsum_data["operations"]]
        final_idx_str = "".join(einsum_data["final_indices"].values())
        if einsum_data.get("batch_index") is not None:
            final_idx_str = f"{einsum_data['batch_index']}{final_idx_str}"
        full_einsum = f"{init_idx_str},{','.join(gate_strs)}->{final_idx_str}"
        return full_einsum, tensors + [op["tensor"] for op in einsum_data["operations"]]
