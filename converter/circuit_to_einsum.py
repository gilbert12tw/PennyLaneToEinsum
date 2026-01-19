from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pennylane as qml

from .index_manager import IndexManager


def _build_tape(circuit_func: Callable[..., Any], params: Optional[Iterable[Any]]) -> qml.tape.QuantumTape:
    with qml.tape.QuantumTape() as tape:
        if params is None:
            circuit_func()
        else:
            circuit_func(*params)
    return tape


def _matrix_to_tensor(matrix: np.ndarray, n_wires: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=complex)
    dim = 2**n_wires
    if matrix.shape != (dim, dim):
        raise ValueError(f"Unexpected matrix shape {matrix.shape} for {n_wires} wires")
    reshaped = matrix.reshape([2] * (2 * n_wires))
    out_axes = list(range(n_wires))
    in_axes = list(range(n_wires, 2 * n_wires))
    transpose_axes = in_axes + out_axes
    return np.transpose(reshaped, axes=transpose_axes)


def contract_einsum(
    einsum_expr: str,
    tensors: List[np.ndarray],
    optimize: Optional[str] = None,
) -> np.ndarray:
    if optimize is None:
        return np.einsum(einsum_expr, *tensors)
    try:
        import opt_einsum as oe
    except Exception:
        return np.einsum(einsum_expr, *tensors)
    return oe.contract(einsum_expr, *tensors, optimize=optimize)


@dataclass
class CircuitToEinsum:
    n_qubits: int
    index_manager: IndexManager

    @classmethod
    def for_qubits(cls, n_qubits: int) -> "CircuitToEinsum":
        return cls(n_qubits=n_qubits, index_manager=IndexManager(n_qubits=n_qubits))

    def _wire_map(self, wires: Iterable[Any]) -> Dict[Any, int]:
        wire_list = list(wires)
        if all(isinstance(w, int) for w in wire_list):
            return {int(w): int(w) for w in wire_list}
        return {wire: idx for idx, wire in enumerate(wire_list)}

    def _apply_gate(self, gate_matrix: np.ndarray, wires: List[int]) -> Tuple[str, np.ndarray]:
        n_wires = len(wires)
        if n_wires < 1:
            raise NotImplementedError("Gates with 0 wires not implemented")

        in_indices = [self.index_manager.get(w) for w in wires]
        out_indices = [self.index_manager.fresh_index() for _ in wires]

        for w, out_idx in zip(wires, out_indices):
            self.index_manager.set(w, out_idx)

        tensor = _matrix_to_tensor(gate_matrix, n_wires)
        in_str = "".join(in_indices)
        out_str = "".join(out_indices)
        einsum_str = f"{in_str},{in_str}{out_str}"
        return einsum_str, tensor

    def circuit_to_einsum(
        self,
        circuit_func: Callable[..., Any],
        params: Optional[Iterable[Any]] = None,
    ) -> Dict[str, Any]:
        self.index_manager.counter = 0
        init_indices = self.index_manager.init_qubits()

        tape = _build_tape(circuit_func, params)

        operations: List[Dict[str, Any]] = []
        for op in tape.operations:
            n_wires = len(op.wires)
            if n_wires < 1:
                raise NotImplementedError("Gates with 0 wires not implemented")

            gate_matrix = op.matrix()
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
                }
            )

        return {
            "initial_indices": init_indices,
            "operations": operations,
            "final_indices": self.index_manager.snapshot(),
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
                raise ValueError("initial_state must be a flat statevector of length 2**n_qubits")

        tensors: List[np.ndarray] = [state.reshape([2] * n_qubits)]
        init_idx_str = "".join(einsum_data["initial_indices"].values())
        gate_strs = [op["einsum"].split(",")[1] for op in einsum_data["operations"]]
        final_idx_str = "".join(einsum_data["final_indices"].values())
        full_einsum = f"{init_idx_str},{','.join(gate_strs)}->{final_idx_str}"
        return full_einsum, tensors + [op["tensor"] for op in einsum_data["operations"]]

