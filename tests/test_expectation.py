"""
Tests for tensor-network expectation values ⟨ψ|O|ψ⟩.

The converter builds ⟨0|U† O U|0⟩ as a single einsum that contracts to a scalar
(no 2**n statevector intermediate). Oracle for every test is qml.expval on
qml.device("default.qubit").
"""

import numpy as np
import pennylane as qml
import pytest

from pennylane_einsum import (
    CircuitToEinsum,
    contract_einsum,
    expectation_value,
    normalize_observable,
)


# ── fixture circuits ──────────────────────────────────────────────────────────

def _circuit_1q():
    qml.Hadamard(wires=0)
    qml.RY(0.7, wires=0)
    qml.RZ(0.3, wires=0)


def _circuit_2q():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(0.7, wires=0)
    qml.RX(1.2, wires=1)


def _circuit_3q():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(0.4, wires=2)


def _pl_expval(circuit_fn, pl_observable, n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def qnode():
        circuit_fn()
        return qml.expval(pl_observable)

    return float(qnode())


# ── single-qubit Pauli ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "pauli, pl_obs",
    [("X", qml.PauliX(0)), ("Y", qml.PauliY(0)), ("Z", qml.PauliZ(0))],
)
def test_single_qubit_pauli(pauli, pl_obs):
    got = expectation_value(_circuit_1q, pauli, n_qubits=1)
    want = _pl_expval(_circuit_1q, pl_obs, n_qubits=1)
    assert np.isclose(got, want, atol=1e-7), f"{pauli}: {got} vs {want}"


# ── multi-qubit Pauli strings ─────────────────────────────────────────────────

def test_two_qubit_pauli_string():
    got = expectation_value(_circuit_2q, "ZX", n_qubits=2)
    want = _pl_expval(_circuit_2q, qml.PauliZ(0) @ qml.PauliX(1), n_qubits=2)
    assert np.isclose(got, want, atol=1e-7)


def test_pauli_dict_subset_of_wires():
    """Observable on a subset of wires — identity wires are traced out."""
    got = expectation_value(_circuit_3q, {2: "Z"}, n_qubits=3)
    want = _pl_expval(_circuit_3q, qml.PauliZ(2), n_qubits=3)
    assert np.isclose(got, want, atol=1e-7)


def test_three_qubit_full_pauli_string():
    got = expectation_value(_circuit_3q, "XYZ", n_qubits=3)
    want = _pl_expval(
        _circuit_3q, qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliZ(2), n_qubits=3
    )
    assert np.isclose(got, want, atol=1e-7)


def test_bell_zz_is_one():
    def bell():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])

    assert np.isclose(expectation_value(bell, "ZZ", n_qubits=2), 1.0, atol=1e-7)


# ── per-wire Hermitian ────────────────────────────────────────────────────────

def _pauli_hermitian(theta, phi):
    c, s = np.cos(theta), np.sin(theta)
    e_p = np.cos(phi) + 1j * np.sin(phi)
    e_n = np.cos(phi) - 1j * np.sin(phi)
    return np.array([[c, e_n * s], [e_p * s, -c]], dtype=complex)


@pytest.mark.parametrize("qubit", [0, 1, 2])
def test_per_wire_hermitian(qubit):
    H = _pauli_hermitian(1.1, 2.3)
    got = expectation_value(_circuit_3q, {qubit: H}, n_qubits=3)
    want = _pl_expval(_circuit_3q, qml.Hermitian(H, wires=qubit), n_qubits=3)
    assert np.isclose(got, want, atol=1e-7), f"qubit={qubit}: {got} vs {want}"


def test_hermitian_matrix_wires_tuple():
    H = _pauli_hermitian(0.5, 1.4)
    got = expectation_value(_circuit_2q, (H, [1]), n_qubits=2)
    want = _pl_expval(_circuit_2q, qml.Hermitian(H, wires=1), n_qubits=2)
    assert np.isclose(got, want, atol=1e-7)


# ── batched parameters ────────────────────────────────────────────────────────

def test_batched_expectation_vector():
    theta = np.linspace(0.1, 0.9, 4)

    def batched():
        qml.Hadamard(wires=0)
        qml.RX(theta, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RZ(theta * 0.5, wires=1)

    got = expectation_value(batched, "ZZ", n_qubits=2)
    assert got.shape == (4,)

    for i, th in enumerate(theta):
        def sample(th=th):
            qml.Hadamard(wires=0)
            qml.RX(th, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(th * 0.5, wires=1)

        want = _pl_expval(sample, qml.PauliZ(0) @ qml.PauliZ(1), n_qubits=2)
        assert np.isclose(got[i], want, atol=1e-7), f"i={i}: {got[i]} vs {want}"


# ── large circuit (>52 indices) ───────────────────────────────────────────────

def test_large_circuit_expectation():
    n = 8

    def circuit(*params):
        for i in range(n):
            qml.Hadamard(wires=i)
        for i in range(n):
            qml.RY(params[i], wires=i)
        for i in range(n - 1):
            qml.CNOT(wires=[i, i + 1])
        for i in range(n):
            qml.RY(params[n + i], wires=i)

    rng = np.random.default_rng(0)
    params = rng.standard_normal(2 * n)

    converter = CircuitToEinsum.for_qubits(n)
    data = converter.circuit_to_einsum(circuit, params=params)
    expr, tensors = converter.generate_expectation_einsum(data, {0: "Z", 7: "Z"})
    assert converter.index_manager.counter > 52, "test requires >52 indices"
    got = contract_einsum(expr, tensors).real

    want = _pl_expval(
        lambda: circuit(*params), qml.PauliZ(0) @ qml.PauliZ(7), n_qubits=n
    )
    assert np.isclose(got, want, atol=1e-7)


# ── normalize_observable ──────────────────────────────────────────────────────

def test_normalize_drops_identity():
    obs = normalize_observable("IZI", n_qubits=3)
    assert set(obs.keys()) == {1}
    np.testing.assert_allclose(obs[1], np.array([[1, 0], [0, -1]], dtype=complex))


def test_normalize_rejects_bad_length():
    with pytest.raises(ValueError, match="length"):
        normalize_observable("ZZ", n_qubits=3)


def test_normalize_rejects_non_2x2():
    with pytest.raises(ValueError, match="2x2"):
        normalize_observable({0: np.eye(4)}, n_qubits=2)


# ── torch autograd ────────────────────────────────────────────────────────────

torch = pytest.importorskip("torch")


def _torch_hermitian(theta, phi, dtype=torch.complex128, requires_grad=False):
    c, s = np.cos(theta), np.sin(theta)
    e_p = np.cos(phi) + 1j * np.sin(phi)
    e_n = np.cos(phi) - 1j * np.sin(phi)
    return torch.tensor(
        [[c, e_n * s], [e_p * s, -c]], dtype=dtype, requires_grad=requires_grad
    )


def test_torch_matches_numpy():
    H_np = _pauli_hermitian(0.9, 0.4)
    H_torch = torch.tensor(H_np, dtype=torch.complex128)
    got = expectation_value(_circuit_2q, {0: H_torch}, n_qubits=2)
    want = expectation_value(_circuit_2q, {0: H_np}, n_qubits=2)
    assert torch.is_tensor(got)
    assert np.isclose(float(got), float(want), atol=1e-7)


def test_torch_gradient_reaches_observable():
    H = _torch_hermitian(0.6, 1.0, requires_grad=True)
    ev = expectation_value(_circuit_2q, {0: H}, n_qubits=2)
    ev.backward()
    assert H.grad is not None
    assert not torch.all(H.grad == 0)


def test_torch_gradient_finite_difference():
    eps = 1e-4
    base = _torch_hermitian(0.8, 1.3)

    H = base.clone().detach().requires_grad_(True)
    ev = expectation_value(_circuit_2q, {1: H}, n_qubits=2)
    ev.backward()
    analytic = H.grad.clone()

    fd = torch.zeros_like(base)
    for i in range(2):
        for j in range(2):
            hp = base.clone().detach()
            hm = base.clone().detach()
            hp[i, j] += eps
            hm[i, j] -= eps
            ep = expectation_value(_circuit_2q, {1: hp}, n_qubits=2)
            em = expectation_value(_circuit_2q, {1: hm}, n_qubits=2)
            fd[i, j] = (ep - em) / (2 * eps)

    torch.testing.assert_close(analytic.real, fd.real, atol=1e-4, rtol=1e-3)


def test_torch_result_on_same_device():
    H = _torch_hermitian(0.5, 0.5)
    ev = expectation_value(_circuit_2q, {0: H}, n_qubits=2)
    assert ev.device == H.device
