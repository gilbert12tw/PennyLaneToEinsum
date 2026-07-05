"""Train a single-qubit observable to hit a target expectation value.

The circuit (and therefore the state |psi>) is fixed; only the observable's two
angles (theta, phi) are trained. Because H(theta, phi) is a Pauli direction it stays
Hermitian and unitary throughout, so every intermediate observable is valid.

Run:  python examples/train_observable.py
"""

import numpy as np
import pennylane as qml
import torch

from pennylane_einsum import PauliDirectionObservable, expectation_value


def circuit():
    qml.Hadamard(wires=0)
    qml.RY(0.7, wires=0)
    qml.RZ(0.3, wires=0)


def main():
    torch.manual_seed(0)
    target = -0.9

    obs = PauliDirectionObservable(theta=0.1, phi=0.1)
    opt = torch.optim.Adam(obs.parameters(), lr=0.1)

    print(f"target <H> = {target}")
    start = float(expectation_value(circuit, {0: obs()}, 1).detach())
    print(f"start:  <H> = {start:+.4f}  "
          f"(theta={float(obs.theta):.3f}, phi={float(obs.phi):.3f})")

    for step in range(1, 201):
        opt.zero_grad()
        ev = expectation_value(circuit, {0: obs()}, n_qubits=1)
        loss = (ev - target) ** 2
        loss.backward()
        opt.step()
        if step % 40 == 0:
            print(f"step {step:3d}: <H> = {float(ev):+.4f}  loss = {float(loss):.2e}")

    final = float(expectation_value(circuit, {0: obs()}, 1).detach())
    print(f"final:  <H> = {final:+.4f}  "
          f"(theta={float(obs.theta):.3f}, phi={float(obs.phi):.3f})")
    print(f"converged to target: {np.isclose(final, target, atol=1e-2)}")


if __name__ == "__main__":
    main()
