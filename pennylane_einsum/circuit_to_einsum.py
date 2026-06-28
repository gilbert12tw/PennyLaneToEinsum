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


def _is_pl_operator(obj: Any) -> bool:
    return isinstance(obj, qml.operation.Operator)


def _normalized_term(matrix: Any, wires: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Any]:
    """Validate a single observable factor: a 2**k x 2**k matrix on ``wires``."""
    k = len(wires)
    if not _is_torch_tensor(matrix):
        matrix = np.asarray(matrix, dtype=complex)
    if tuple(matrix.shape) != (2**k, 2**k):
        raise ValueError(
            f"Observable on wires {wires} must be {2**k}x{2**k}; "
            f"got shape {tuple(matrix.shape)}"
        )
    return wires, matrix


def _pl_operator_to_dict(op: Any, n_qubits: int) -> Dict[Tuple[int, ...], Any]:
    """Factor a single (non-summed) PennyLane operator into ``{wires_tuple: matrix}``.

    Tensor products / ``Prod`` are split into their disjoint-wire factors so the
    observable tensors stay small; identity factors are dropped; anything else
    becomes one dense block over its own wires.
    """
    name = getattr(op, "name", "")
    if name in ("Identity", "I"):
        return {}

    operands = getattr(op, "operands", None)
    if operands is None and hasattr(op, "obs"):  # legacy qml.operation.Tensor
        operands = op.obs
    if operands is not None and name in ("Prod", "Tensor"):
        result: Dict[Tuple[int, ...], Any] = {}
        for sub in operands:
            result.update(_pl_operator_to_dict(sub, n_qubits))
        return result

    wires = tuple(int(w) for w in op.wires)
    matrix = qml.matrix(op, wire_order=op.wires)
    key, value = _normalized_term(matrix, wires)
    return {key: value}


def _observable_terms(
    observable: Any, n_qubits: int
) -> List[Tuple[complex, Dict[Tuple[int, ...], Any]]]:
    """Expand any observable into a weighted sum ``[(coeff, single_operator_dict)]``.

    Non-summed inputs yield a single ``(1.0, dict)`` entry; a PennyLane
    ``Hamiltonian`` / ``LinearCombination`` / ``Sum`` yields one entry per term so
    the caller can compute ``<H> = sum_i c_i <O_i>`` by linearity.
    """
    if _is_pl_operator(observable):
        try:
            coeffs, ops = observable.terms()
        except Exception:
            coeffs, ops = [1.0], [observable]
        return [
            (complex(c), _pl_operator_to_dict(op, n_qubits))
            for c, op in zip(coeffs, ops)
        ]
    return [(1.0 + 0j, normalize_observable(observable, n_qubits))]


def normalize_observable(
    observable: Any,
    n_qubits: int,
    wire_order: Optional[Iterable[int]] = None,
) -> Dict[Tuple[int, ...], Any]:
    """Normalize a single (non-summed) observable into ``{wires_tuple: matrix}``.

    Accepted forms:
      - Pauli string ``"IZZ"`` — one char per wire in ``wire_order`` (default
        ``range(n_qubits)``). ``'I'`` entries are dropped.
      - dict ``{wire: 'Z'}`` (Pauli chars) or ``{wire: 2x2 matrix}`` (single-qubit
        Hermitian); a tuple key ``{(0, 1): 4x4 matrix}`` gives a multi-qubit factor.
      - tuple ``(matrix, wires)`` — a 2**k x 2**k matrix on ``k`` wires.
      - a single PennyLane observable (``qml.PauliZ(0)``, ``qml.PauliX(0) @
        qml.PauliZ(1)``, ``qml.Hermitian(H, wires=...)``). For a weighted sum
        (``qml.Hamiltonian``) use :func:`expectation_value`, which sums the terms.

    Keys are wire tuples; values are ``2**k x 2**k`` matrices (numpy or torch).
    Identity / unmeasured wires are omitted — the caller traces them out by
    collapsing bra/ket frontier indices.
    """
    if _is_pl_operator(observable):
        terms = _observable_terms(observable, n_qubits)
        if len(terms) != 1 or abs(terms[0][0] - 1.0) > 1e-12:
            raise ValueError(
                "Weighted / multi-term observable (e.g. qml.Hamiltonian); "
                "use expectation_value(), which sums the terms by linearity."
            )
        return terms[0][1]

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
        op_wires = (op_wires,) if isinstance(op_wires, int) else tuple(op_wires)
        observable = {op_wires: matrix}

    if not isinstance(observable, dict):
        raise TypeError(
            "observable must be a Pauli string, a dict, a (matrix, wires) tuple, "
            f"or a PennyLane observable; got {type(observable).__name__}"
        )

    result: Dict[Tuple[int, ...], Any] = {}
    for key, spec in observable.items():
        key_wires = key if isinstance(key, tuple) else (key,)
        if isinstance(spec, str):
            if spec not in _PAULI_MATRICES:
                raise ValueError(f"Unknown Pauli character '{spec}' on wire {key}")
            if spec == "I":
                continue
            result[key_wires] = _PAULI_MATRICES[spec]
        else:
            k, v = _normalized_term(spec, key_wires)
            result[k] = v
    return result


def _causal_ops(
    operations: List[Dict[str, Any]], measured_wires: Iterable[int]
) -> List[Dict[str, Any]]:
    """Reverse lightcone: keep only gates that can causally affect ``measured_wires``.

    Walking the gate list backwards, a gate is kept iff it touches a wire already in
    the cone; when kept, all its wires join the cone (an entangling gate pulls its
    partners in). Gates never touching the cone are dropped — in the ⟨0|U† O U|0⟩
    network they sit outside the observable's support and cancel against their
    inverse, so removing them leaves the expectation unchanged.
    """
    cone = set(measured_wires)
    kept_reversed: List[Dict[str, Any]] = []
    for op in reversed(operations):
        op_wires = set(op["wires"])
        if op_wires & cone:
            kept_reversed.append(op)
            cone |= op_wires
    kept_reversed.reverse()
    return kept_reversed


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

    def _thread_ket(
        self,
        operations: List[Dict[str, Any]],
        initial_state: Optional[np.ndarray] = None,
    ) -> Tuple[List[Tuple[str, np.ndarray]], Dict[int, str], Optional[str]]:
        """Re-thread fresh einsum indices over an arbitrary subset of operations.

        Unlike :meth:`_ket_operands` (which reuses the labels baked into
        ``einsum_data``), this re-derives the qubit frontier from scratch so a
        *filtered* gate list (lightcone) still chains correctly. Returns
        ``(ket_terms, final_indices, batch_index)``.
        """
        n_qubits = self.n_qubits
        self.index_manager.counter = 0
        init = self.index_manager.init_qubits()

        if initial_state is None:
            state = np.zeros(2**n_qubits, dtype=complex)
            state[0] = 1.0
        else:
            state = np.asarray(initial_state, dtype=complex)
            if state.shape != (2**n_qubits,):
                raise ValueError(
                    "initial_state must be a flat statevector of length 2**n_qubits"
                )

        init_str = "".join(init[q] for q in range(n_qubits))
        terms: List[Tuple[str, np.ndarray]] = [(init_str, state.reshape([2] * n_qubits))]
        frontier = dict(init)
        batch_index: Optional[str] = None

        for op in operations:
            wires = op["wires"]
            tensor = op["tensor"]
            n_wires = len(wires)
            in_idx = [frontier[w] for w in wires]
            out_idx = [self.index_manager.fresh_index() for _ in wires]
            for w, o in zip(wires, out_idx):
                frontier[w] = o
            prefix = ""
            if tensor.ndim == 2 * n_wires + 1:
                if batch_index is None:
                    batch_index = self.index_manager.fresh_index()
                prefix = batch_index
            terms.append((f"{prefix}{''.join(in_idx)}{''.join(out_idx)}", tensor))

        final_indices = {q: frontier[q] for q in range(n_qubits)}
        return terms, final_indices, batch_index

    def generate_expectation_einsum(
        self,
        einsum_data: Dict[str, Any],
        observable: Any,
        initial_state: Optional[np.ndarray] = None,
        lightcone: bool = False,
    ) -> Tuple[str, List[Any]]:
        """Build ``⟨ψ|O|ψ⟩`` as a single einsum that contracts to a scalar.

        Mirrors cuQuantum's ``expectation``: the ket ``U|0⟩``, the observable ``O``,
        and the bra ``⟨0|U†`` (the conjugate mirror of the ket with relabeled
        indices) are assembled into one network with no free output indices (or just
        the batch index when the circuit is batched).

        ``observable`` accepts any single (non-summed) form understood by
        :func:`normalize_observable`, including a PennyLane observable. If it contains
        torch tensors, all operands are converted to torch so ``opt_einsum`` can
        backpropagate through it.

        With ``lightcone=True``, gates outside the observable's reverse causal cone
        are dropped (they would cancel against their inverse in the bra), shrinking the
        network. The result is unchanged; the contraction path depends on the
        observable, so it cannot be reused across different observables.
        """
        obs = normalize_observable(observable, self.n_qubits)
        measured: set = set()
        for wires_tuple in obs:
            measured.update(wires_tuple)

        if lightcone:
            ops = _causal_ops(einsum_data["operations"], measured)
            ket_terms, final_indices, batch_index = self._thread_ket(ops, initial_state)
        else:
            ket_terms = self._ket_operands(einsum_data, initial_state)
            final_indices = einsum_data["final_indices"]
            batch_index = einsum_data.get("batch_index")

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

        for wires_tuple, matrix in obs.items():  # observable: O_{bra, ket}
            ket_legs = [final_indices[w] for w in wires_tuple]
            bra_legs = [relabel_map[leg] for leg in ket_legs]
            shape = (2,) * (2 * len(wires_tuple))
            idx_strs.append(f"{''.join(bra_legs)}{''.join(ket_legs)}")
            operands.append(matrix.reshape(shape))

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
    lightcone: bool = False,
) -> Any:
    """Compute ``⟨ψ|O|ψ⟩`` for ``|ψ⟩ = U|0…0⟩`` via a tensor-network contraction.

    Convenience wrapper around :meth:`CircuitToEinsum.generate_expectation_einsum`.
    ``observable`` accepts any form understood by :func:`normalize_observable`, plus
    weighted-sum PennyLane observables (``qml.Hamiltonian`` / ``LinearCombination`` /
    ``Sum``), evaluated by linearity ``<H> = Σ_i c_i <O_i>``.

    Returns a scalar (a length-``batch`` vector for batched circuits). For a Hermitian
    observable the value is real; ``return_real`` takes ``.real`` to match
    ``qml.expval`` semantics while preserving torch gradients. With ``lightcone=True``
    each term is contracted over only its causal gates.
    """
    converter = CircuitToEinsum.for_qubits(n_qubits)
    data = converter.circuit_to_einsum(circuit_func, params=params)

    terms = _observable_terms(observable, n_qubits)

    total: Any = None
    for coeff, op_dict in terms:
        if not op_dict:  # all-identity term: <I> = 1
            contribution = coeff
        else:
            expr, tensors = converter.generate_expectation_einsum(
                data, op_dict, initial_state, lightcone=lightcone
            )
            contribution = coeff * contract_einsum(expr, tensors, optimize=optimize)
        total = contribution if total is None else total + contribution

    if return_real and hasattr(total, "real"):
        return total.real
    return total
