"""Trainable single-qubit observable parametrized by two angles.

``PauliDirectionObservable(theta, phi)`` is a ``torch.nn.Module`` whose forward
pass returns the 2x2 matrix

    H(theta, phi) = cos(theta) Z + sin(theta) (cos(phi) X + sin(phi) Y)
                  = [[ cos(theta),           e^{-i phi} sin(theta)],
                     [ e^{i phi} sin(theta), -cos(theta)          ]]

This is a Pauli direction ``n . sigma`` with unit ``n``, so it is both **Hermitian**
(a valid observable, giving a real expectation value) and **unitary** (eigenvalues
+/-1) for every value of the angles. Training the two real angles therefore keeps the
observable on the valid manifold — no projection or constraint is needed.

Feed the forward output straight into :func:`pennylane_einsum.expectation_value`::

    obs = PauliDirectionObservable(0.1, 0.1)
    opt = torch.optim.Adam(obs.parameters(), lr=0.1)
    ev = expectation_value(circuit, {q: obs()}, n_qubits)   # torch scalar
    loss = (ev - target) ** 2
    loss.backward()      # gradients reach obs.theta / obs.phi
    opt.step()
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn

    _Base = nn.Module
except ImportError:  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]
    _Base = object  # type: ignore[misc, assignment]


class PauliDirectionObservable(_Base):
    """A trainable single-qubit observable ``H(theta, phi) = n . sigma``.

    Args:
        theta: initial polar angle (real). Trainable ``nn.Parameter``.
        phi: initial azimuthal angle (real). Trainable ``nn.Parameter``.
        dtype: complex dtype of the forward matrix (default ``torch.complex128``).
            Angles are stored in the matching real dtype.
    """

    def __init__(
        self, theta: float = 0.0, phi: float = 0.0, dtype: "Any" = None
    ) -> None:
        if torch is None:  # pragma: no cover - exercised only without torch
            raise ImportError(
                "torch is required for PauliDirectionObservable; "
                "install it with: pip install torch"
            )
        super().__init__()
        self.dtype = torch.complex128 if dtype is None else dtype
        real_dtype = torch.zeros((), dtype=self.dtype).real.dtype
        self.theta = nn.Parameter(torch.tensor(float(theta), dtype=real_dtype))
        self.phi = nn.Parameter(torch.tensor(float(phi), dtype=real_dtype))

    def forward(self) -> "torch.Tensor":
        c, s = torch.cos(self.theta), torch.sin(self.theta)
        cp, sp = torch.cos(self.phi), torch.sin(self.phi)
        zero = torch.zeros_like(c)
        real = torch.stack(
            [torch.stack([c, cp * s]), torch.stack([cp * s, -c])]
        )
        imag = torch.stack(
            [torch.stack([zero, -sp * s]), torch.stack([sp * s, zero])]
        )
        return torch.complex(real, imag).to(self.dtype)
