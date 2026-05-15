from __future__ import annotations


class UnsupportedOperationError(Exception):
    """Raised when a PennyLane operation cannot be converted to an einsum term.

    This covers ops without a dense matrix representation (e.g. state
    preparation, channels) and ops whose matrix() call raises at extraction time.
    """

    def __init__(self, op_name: str, wires: list, reason: str) -> None:
        self.op_name = op_name
        self.wires = wires
        self.reason = reason
        super().__init__(
            f"Cannot convert operation '{op_name}' on wires {wires}: {reason}. "
            f"Only operations that provide a dense matrix via op.matrix() are supported."
        )
