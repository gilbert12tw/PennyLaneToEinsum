"""
Tests for the trainable observable feature.

The observable is parametrized by two angles (theta, phi) producing a
Pauli-direction matrix

    H(theta, phi) = cos(theta) Z + sin(theta)(cos(phi) X + sin(phi) Y)
                  = [[ cos(theta),           e^{-i phi} sin(theta)],
                     [ e^{i phi} sin(theta), -cos(theta)          ]]

which is Hermitian AND unitary (eigenvalues +/-1) for every (theta, phi), so it
stays a valid observable throughout training. Gradients flow to theta/phi via the
torch backend of ``expectation_value``.

Oracle for the numeric value is qml.expval(qml.Hermitian(H, wires=q)).
"""

import numpy as np
import pennylane as qml
import pytest

torch = pytest.importorskip("torch")

from pennylane_einsum import PauliDirectionObservable, expectation_value


# ── fixture circuit (single, definite Bloch vector) ───────────────────────────

def _circuit_1q():
    qml.Hadamard(wires=0)
    qml.RY(0.7, wires=0)
    qml.RZ(0.3, wires=0)


def _circuit_2q():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(0.7, wires=0)
    qml.RX(1.2, wires=1)


def _reference_matrix(theta, phi):
    c, s = np.cos(theta), np.sin(theta)
    e_p = np.cos(phi) + 1j * np.sin(phi)
    e_n = np.cos(phi) - 1j * np.sin(phi)
    return np.array([[c, e_n * s], [e_p * s, -c]], dtype=complex)


def _pl_expval(circuit_fn, H_np, qubit, n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def qnode():
        circuit_fn()
        return qml.expval(qml.Hermitian(H_np, wires=qubit))

    return float(qnode())


# ── A. matrix validity: Hermitian AND unitary by construction ─────────────────

@pytest.mark.parametrize(
    "theta, phi",
    [(0.0, 0.0), (1.1, 2.3), (0.7, -0.4), (np.pi / 2, np.pi), (2.9, 5.1)],
)
def test_matrix_is_hermitian(theta, phi):
    H = PauliDirectionObservable(theta, phi)()
    torch.testing.assert_close(H, H.conj().T)


@pytest.mark.parametrize("theta, phi", [(1.1, 2.3), (0.7, -0.4), (2.9, 5.1)])
def test_matrix_is_unitary(theta, phi):
    H = PauliDirectionObservable(theta, phi)()
    eye = torch.eye(2, dtype=H.dtype)
    torch.testing.assert_close(H @ H.conj().T, eye, atol=1e-12, rtol=0)


@pytest.mark.parametrize("theta, phi", [(1.1, 2.3), (0.7, -0.4)])
def test_matrix_matches_reference(theta, phi):
    H = PauliDirectionObservable(theta, phi)().detach().numpy()
    np.testing.assert_allclose(H, _reference_matrix(theta, phi), atol=1e-12)


# ── B. fixed observable => stable and correct expectation value ───────────────

def test_expval_is_real():
    obs = PauliDirectionObservable(1.1, 2.3)
    ev = expectation_value(_circuit_1q, {0: obs()}, n_qubits=1)
    assert torch.is_tensor(ev)
    assert abs(float(ev.imag) if torch.is_complex(ev) else 0.0) < 1e-9


def test_fixed_observable_is_deterministic():
    obs = PauliDirectionObservable(1.1, 2.3)
    ev1 = expectation_value(_circuit_1q, {0: obs()}, n_qubits=1).detach()
    ev2 = expectation_value(_circuit_1q, {0: obs()}, n_qubits=1).detach()
    assert float(ev1) == float(ev2)


def test_fixed_observable_matches_pennylane():
    theta, phi = 1.1, 2.3
    obs = PauliDirectionObservable(theta, phi)
    got = float(expectation_value(_circuit_1q, {0: obs()}, n_qubits=1).detach())
    want = _pl_expval(_circuit_1q, _reference_matrix(theta, phi), 0, 1)
    assert np.isclose(got, want, atol=1e-7), f"{got} vs {want}"


def test_fixed_observable_matches_pennylane_multiqubit():
    theta, phi = 0.7, -0.4
    obs = PauliDirectionObservable(theta, phi)
    got = float(expectation_value(_circuit_2q, {1: obs()}, n_qubits=2).detach())
    want = _pl_expval(_circuit_2q, _reference_matrix(theta, phi), 1, 2)
    assert np.isclose(got, want, atol=1e-7), f"{got} vs {want}"


# ── C. changing observable => expectation value changes ───────────────────────

def test_changing_angles_changes_expval():
    ev_a = float(expectation_value(_circuit_1q, {0: PauliDirectionObservable(0.3, 0.0)()}, 1).detach())
    ev_b = float(expectation_value(_circuit_1q, {0: PauliDirectionObservable(2.4, 1.7)()}, 1).detach())
    assert abs(ev_a - ev_b) > 1e-3, f"expval did not change: {ev_a} vs {ev_b}"


def test_expval_traces_bloch_projection():
    """<H(theta,phi)> == n . <sigma> for the fixed state's Bloch vector."""
    r = np.array([
        _pl_expval(_circuit_1q, _reference_matrix(np.pi / 2, 0.0), 0, 1),   # <X>
        _pl_expval(_circuit_1q, _reference_matrix(np.pi / 2, np.pi / 2), 0, 1),  # <Y>
        _pl_expval(_circuit_1q, _reference_matrix(0.0, 0.0), 0, 1),         # <Z>
    ])
    for theta, phi in [(0.4, 0.9), (1.9, -1.2), (2.7, 3.3)]:
        n = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ])
        got = float(expectation_value(_circuit_1q, {0: PauliDirectionObservable(theta, phi)()}, 1).detach())
        assert np.isclose(got, float(n @ r), atol=1e-7), f"{got} vs {n @ r}"


# ── D. trainability: gradients reach angles and a loop converges ──────────────

def test_gradient_reaches_angles():
    obs = PauliDirectionObservable(0.6, 1.0)
    ev = expectation_value(_circuit_1q, {0: obs()}, n_qubits=1)
    ev.backward()
    assert obs.theta.grad is not None and obs.phi.grad is not None
    assert not (obs.theta.grad == 0 and obs.phi.grad == 0)


def test_gradient_finite_difference():
    eps = 1e-5
    theta0, phi0 = 0.8, 1.3

    obs = PauliDirectionObservable(theta0, phi0)
    ev = expectation_value(_circuit_1q, {0: obs()}, n_qubits=1)
    ev.backward()
    g_theta = float(obs.theta.grad)

    ep = float(expectation_value(_circuit_1q, {0: PauliDirectionObservable(theta0 + eps, phi0)()}, 1).detach())
    em = float(expectation_value(_circuit_1q, {0: PauliDirectionObservable(theta0 - eps, phi0)()}, 1).detach())
    fd = (ep - em) / (2 * eps)
    assert np.isclose(g_theta, fd, atol=1e-4), f"{g_theta} vs {fd}"


def test_training_loop_converges_to_target():
    """Optimizing (theta, phi) drives <H> toward a target; the observable moves."""
    target = -0.95
    obs = PauliDirectionObservable(0.1, 0.1)
    theta0, phi0 = obs.theta.item(), obs.phi.item()

    opt = torch.optim.Adam(obs.parameters(), lr=0.1)
    first_loss = None
    for _ in range(400):
        opt.zero_grad()
        ev = expectation_value(_circuit_1q, {0: obs()}, n_qubits=1)
        loss = (ev - target) ** 2
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        opt.step()

    final_ev = float(expectation_value(_circuit_1q, {0: obs()}, n_qubits=1).detach())
    assert loss.item() < first_loss, "loss did not decrease"
    assert np.isclose(final_ev, target, atol=1e-2), f"did not converge: {final_ev}"
    # the observable actually changed during training
    assert abs(obs.theta.item() - theta0) + abs(obs.phi.item() - phi0) > 1e-2
