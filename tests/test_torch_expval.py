"""
Tests for expval_hermitian_torch — autograd-compatible expectation value.

Covers:
  - numerical match vs PennyLane qml.expval(Hermitian)
  - gradients reach nn.Parameter (backprop works)
  - GPU transparency: result is on same device as H
  - multi-qubit: all qubits in a 3-qubit system
  - ImportError is raised when torch is absent (mocked)
"""

import numpy as np
import pennylane as qml
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from pennylane_einsum import CircuitToEinsum, contract_einsum, expval_hermitian_torch


# ── shared helpers ────────────────────────────────────────────────────────────

def _statevec(circuit_fn, n_qubits):
    converter = CircuitToEinsum.for_qubits(n_qubits)
    data = converter.circuit_to_einsum(circuit_fn)
    expr, tensors = converter.generate_full_einsum(data)
    return contract_einsum(expr, tensors).flatten()


def _pl_expval(circuit_fn, H_np, qubit, n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def qnode():
        circuit_fn()
        return qml.expval(qml.Hermitian(H_np, wires=qubit))

    return float(qnode())


def _pauli_hermitian(theta, phi, dtype=torch.complex64):
    c = np.cos(theta)
    s = np.sin(theta)
    e_p = np.cos(phi) + 1j * np.sin(phi)
    e_n = np.cos(phi) - 1j * np.sin(phi)
    return torch.tensor([[c, e_n * s], [e_p * s, -c]], dtype=dtype)


# ── fixture circuits ──────────────────────────────────────────────────────────

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


# ── numerical correctness ─────────────────────────────────────────────────────

@pytest.mark.parametrize("qubit", [0, 1])
def test_matches_pennylane_2qubit(qubit):
    H_torch = _pauli_hermitian(0.9, 0.4)
    H_np = H_torch.numpy()

    sv = _statevec(_circuit_2q, n_qubits=2)
    ev_torch = expval_hermitian_torch(sv, H_torch, qubit, n_qubits=2).item()
    ev_pl = _pl_expval(_circuit_2q, H_np, qubit, n_qubits=2)

    assert abs(ev_torch - ev_pl) < 1e-5, f"qubit={qubit}: {ev_torch} vs {ev_pl}"


@pytest.mark.parametrize("qubit", [0, 1, 2])
def test_matches_pennylane_3qubit(qubit):
    H_torch = _pauli_hermitian(1.1, 2.3)
    H_np = H_torch.numpy()

    sv = _statevec(_circuit_3q, n_qubits=3)
    ev_torch = expval_hermitian_torch(sv, H_torch, qubit, n_qubits=3).item()
    ev_pl = _pl_expval(_circuit_3q, H_np, qubit, n_qubits=3)

    assert abs(ev_torch - ev_pl) < 1e-5, f"qubit={qubit}: {ev_torch} vs {ev_pl}"


# ── autograd ──────────────────────────────────────────────────────────────────

def test_gradient_reaches_parameter():
    """Gradient must flow back to H (the learnable observable)."""
    H = nn.Parameter(_pauli_hermitian(0.6, 1.0, dtype=torch.complex128))
    sv = _statevec(_circuit_2q, n_qubits=2)

    ev = expval_hermitian_torch(sv, H, qubit=0, n_qubits=2)
    ev.backward()

    assert H.grad is not None, "No gradient reached H"
    assert H.grad.shape == (2, 2)
    assert not torch.all(H.grad == 0), "Gradient is all-zero"


def test_gradient_multi_qubit_sum():
    """Sum of expvals over all qubits — gradients for each H_q."""
    n_qubits = 2
    H_params = nn.ParameterList([
        nn.Parameter(_pauli_hermitian(0.3 * q, 0.5 * q + 0.1, dtype=torch.complex128))
        for q in range(n_qubits)
    ])
    sv = _statevec(_circuit_2q, n_qubits=n_qubits)

    total = sum(
        expval_hermitian_torch(sv, H_params[q], q, n_qubits)
        for q in range(n_qubits)
    )
    total.backward()

    for q, H in enumerate(H_params):
        assert H.grad is not None, f"No gradient for qubit {q}"
        assert not torch.all(H.grad == 0), f"Zero gradient for qubit {q}"


def test_gradient_value_finite_difference():
    """Numerical gradient check via finite differences."""
    eps = 1e-4
    H_val = _pauli_hermitian(0.8, 1.3, dtype=torch.complex128)
    sv = _statevec(_circuit_2q, n_qubits=2)

    H = nn.Parameter(H_val.clone())
    ev = expval_hermitian_torch(sv, H, qubit=1, n_qubits=2)
    ev.backward()
    analytic_grad = H.grad.clone()

    fd_grad = torch.zeros_like(H_val)
    for i in range(2):
        for j in range(2):
            H_plus = H_val.clone().detach()
            H_minus = H_val.clone().detach()
            H_plus[i, j] += eps
            H_minus[i, j] -= eps
            ev_p = expval_hermitian_torch(sv, H_plus, qubit=1, n_qubits=2)
            ev_m = expval_hermitian_torch(sv, H_minus, qubit=1, n_qubits=2)
            fd_grad[i, j] = (ev_p - ev_m) / (2 * eps)

    # Compare real parts (imaginary parts of the gradient are artifacts of
    # the Wirtinger derivative; only real perturbations are physically meaningful)
    torch.testing.assert_close(
        analytic_grad.real, fd_grad.real, atol=1e-4, rtol=1e-3
    )


# ── device transparency ───────────────────────────────────────────────────────

def test_output_on_same_device_as_H():
    """Result tensor must live on the same device as H."""
    H = _pauli_hermitian(0.5, 0.5)
    sv = _statevec(_circuit_2q, n_qubits=2)
    ev = expval_hermitian_torch(sv, H, qubit=0, n_qubits=2)
    assert ev.device == H.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_result_on_cuda():
    H = _pauli_hermitian(0.5, 0.5).cuda()
    sv = _statevec(_circuit_2q, n_qubits=2)
    ev = expval_hermitian_torch(sv, H, qubit=0, n_qubits=2)
    assert ev.is_cuda


# ── error handling ────────────────────────────────────────────────────────────

def test_import_error_without_torch(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    import pennylane_einsum.circuit_to_einsum as mod
    sv = np.zeros(4, dtype=complex)
    H = np.eye(2, dtype=complex)

    with pytest.raises(ImportError, match="torch"):
        mod.expval_hermitian_torch(sv, H, qubit=0, n_qubits=2)
