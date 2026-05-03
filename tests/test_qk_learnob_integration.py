"""
Integration test: PennyLaneToEinsum works with QK-Learnob circuit patterns.

CPU-only, no torch, no tensorflow. Uses random data.
Covers:
  1. statevector correctness
  2. projected kernel (partial-trace density matrix, from projected.ipynb)
  3. single-qubit Hermitian expval (from pauil.ipynb)
  4. kernel matrix is PSD
"""

import numpy as np
import pennylane as qml
import pytest

from pennylane_einsum import CircuitToEinsum, contract_einsum


# ── QK-Learnob circuit (identical structure to both notebooks) ────────────────

def _layer(x, wires):
    for i, w in enumerate(wires):
        qml.RX(x[i], wires=w)
    for i in range(len(wires)):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % len(wires)]])


def ansatz(x, layers, wires):
    for _ in range(layers):
        _layer(x, wires)


# ── Einsum helpers ────────────────────────────────────────────────────────────

def einsum_statevec(x, n_qubits, layers):
    converter = CircuitToEinsum.for_qubits(n_qubits)
    data = converter.circuit_to_einsum(lambda: ansatz(x, layers, range(n_qubits)))
    expr, tensors = converter.generate_full_einsum(data)
    return contract_einsum(expr, tensors).flatten()


def partial_trace_qubit(statevec, qubit, n_qubits):
    """Single-qubit reduced density matrix via partial trace."""
    state = statevec.reshape([2] * n_qubits)
    state = np.moveaxis(state, qubit, 0).reshape(2, -1)
    return state @ state.conj().T


def expval_single_hermitian(statevec, H, qubit, n_qubits):
    """<psi| I^⊗q ⊗ H ⊗ I^⊗rest |psi>, real part."""
    full_op = np.array([[1.0 + 0j]])
    for q in range(n_qubits):
        full_op = np.kron(full_op, H if q == qubit else np.eye(2, dtype=complex))
    return (statevec.conj() @ full_op @ statevec).real


def make_hermitians(n_qubits, seed=7):
    """Random single-qubit Hermitian observables (LearnablePauliDirection style)."""
    rng = np.random.default_rng(seed)
    H_list = []
    for _ in range(n_qubits):
        theta = rng.uniform(0, np.pi)
        phi = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        e_p = np.cos(phi) + 1j * np.sin(phi)
        e_n = np.cos(phi) - 1j * np.sin(phi)
        H_list.append(np.array([[c, e_n * s], [e_p * s, -c]], dtype=complex))
    return H_list


# ── Fixtures ──────────────────────────────────────────────────────────────────

N_QUBITS = 2
LAYERS = 2
RNG = np.random.default_rng(42)


@pytest.fixture
def x1():
    return RNG.uniform(0, 2 * np.pi, N_QUBITS)


@pytest.fixture
def x2():
    return RNG.uniform(0, 2 * np.pi, N_QUBITS)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_statevector_matches_pennylane(x1):
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def circuit_pl(x):
        ansatz(x, LAYERS, range(N_QUBITS))
        return qml.state()

    sv_pl = np.array(circuit_pl(x1))
    sv_ein = einsum_statevec(x1, N_QUBITS, LAYERS)

    # Global phase may differ; compare up to phase on first non-zero element
    phase = sv_pl[np.argmax(np.abs(sv_pl))] / sv_ein[np.argmax(np.abs(sv_ein))]
    np.testing.assert_allclose(sv_ein * phase, sv_pl, atol=1e-10,
                               err_msg="Statevector mismatch")


def test_projected_kernel_entry(x1, x2):
    """K(x1, x2) = sum_q Tr[rho_q(x1) @ rho_q(x2)]  (projected kernel)."""
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def dm_pl(x, q):
        ansatz(x, LAYERS, range(N_QUBITS))
        return qml.density_matrix(wires=[q])

    k_pl = sum(
        np.trace(np.array(dm_pl(x1, q)) @ np.array(dm_pl(x2, q))).real
        for q in range(N_QUBITS)
    )

    sv1 = einsum_statevec(x1, N_QUBITS, LAYERS)
    sv2 = einsum_statevec(x2, N_QUBITS, LAYERS)
    k_ein = sum(
        np.trace(partial_trace_qubit(sv1, q, N_QUBITS) @
                 partial_trace_qubit(sv2, q, N_QUBITS)).real
        for q in range(N_QUBITS)
    )

    np.testing.assert_allclose(k_ein, k_pl, atol=1e-10,
                               err_msg="Projected kernel entry mismatch")


def test_hermitian_expval_matches_pennylane(x1):
    """<psi|H_q|psi> for LearnablePauliDirection-style Hermitian observables."""
    H_list = make_hermitians(N_QUBITS)

    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def circuit_pl(x):
        ansatz(x, LAYERS, range(N_QUBITS))
        return tuple(qml.expval(qml.Hermitian(H_list[q], wires=q))
                     for q in range(N_QUBITS))

    ev_pl = np.array(circuit_pl(x1), dtype=float)

    sv = einsum_statevec(x1, N_QUBITS, LAYERS)
    ev_ein = np.array([expval_single_hermitian(sv, H_list[q], q, N_QUBITS)
                       for q in range(N_QUBITS)])

    np.testing.assert_allclose(ev_ein, ev_pl, atol=1e-10,
                               err_msg="Hermitian expval mismatch")


def test_kernel_matrix_is_psd():
    """Feature kernel K = phi @ phi.T must be positive semi-definite."""
    n_samples = 6
    X = RNG.uniform(0, 2 * np.pi, (n_samples, N_QUBITS))
    H_list = make_hermitians(N_QUBITS)

    features = np.array([
        [expval_single_hermitian(einsum_statevec(x, N_QUBITS, LAYERS), H_list[q], q, N_QUBITS)
         for q in range(N_QUBITS)]
        for x in X
    ])

    K = features @ features.T
    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals >= -1e-10), \
        f"Kernel matrix not PSD: min eigenvalue = {eigvals.min():.3e}"


def test_projected_kernel_matrix_is_psd():
    """Projected kernel matrix must also be PSD."""
    n_samples = 6
    X = RNG.uniform(0, 2 * np.pi, (n_samples, N_QUBITS))

    statevecs = [einsum_statevec(x, N_QUBITS, LAYERS) for x in X]
    rho_per_sample = [
        [partial_trace_qubit(sv, q, N_QUBITS) for q in range(N_QUBITS)]
        for sv in statevecs
    ]

    K = np.array([
        [sum(np.trace(rho_per_sample[i][q] @ rho_per_sample[j][q]).real
             for q in range(N_QUBITS))
         for j in range(n_samples)]
        for i in range(n_samples)
    ])

    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals >= -1e-10), \
        f"Projected kernel matrix not PSD: min eigenvalue = {eigvals.min():.3e}"
