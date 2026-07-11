from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence


def _integer_operands(
    einsum_expr: str, tensors: Sequence[Any]
) -> list[Any]:
    inputs, output = einsum_expr.split("->")
    terms = inputs.split(",")
    if len(terms) != len(tensors):
        raise ValueError("einsum expression and tensor count do not match")

    labels = dict.fromkeys("".join(terms) + output)
    modes = {label: mode for mode, label in enumerate(labels)}
    operands: list[Any] = []
    for tensor, term in zip(tensors, terms):
        operands.extend((tensor, [modes[label] for label in term]))
    operands.append([modes[label] for label in output])
    return operands


class CuQuantumContractor:
    """Reusable cuTensorNet contraction plan for one einsum topology.

    Operands must already be CUDA arrays. Call :meth:`contract_path` once, then
    reuse :meth:`contract` or :meth:`reset_operands` for arrays with identical
    shapes, strides, dtypes, and modes.
    """

    def __init__(
        self,
        einsum_expr: str,
        device_tensors: Sequence[Any],
        *,
        options: Optional[Any] = None,
        stream: Optional[Any] = None,
    ) -> None:
        try:
            from cuquantum.tensornet import Network
        except ImportError as exc:
            raise ImportError(
                "CuQuantumContractor requires the 'cuquantum' extra; "
                "install it with: uv sync --extra cuquantum"
            ) from exc

        self.einsum_expr = einsum_expr
        self._network = Network(
            *_integer_operands(einsum_expr, device_tensors),
            options=options,
            stream=stream,
        )
        self._planned = False

    def contract_path(self, optimize: Optional[Any] = None):
        result = self._network.contract_path(optimize=optimize)
        self._planned = True
        return result

    def autotune(
        self,
        *,
        iterations: int = 3,
        stream: Optional[Any] = None,
        release_workspace: bool = False,
    ) -> None:
        if not self._planned:
            raise RuntimeError("contract_path() must be called before autotune()")
        self._network.autotune(
            iterations=iterations,
            stream=stream,
            release_workspace=release_workspace,
        )

    def contract(
        self,
        *,
        stream: Optional[Any] = None,
        release_workspace: bool = False,
    ):
        if not self._planned:
            raise RuntimeError("contract_path() must be called before contract()")
        return self._network.contract(
            stream=stream,
            release_workspace=release_workspace,
        )

    def reset_operands(
        self, device_tensors: Iterable[Any], *, stream: Optional[Any] = None
    ) -> None:
        self._network.reset_operands(*device_tensors, stream=stream)

    def free(self) -> None:
        self._network.free()

    def __enter__(self) -> "CuQuantumContractor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.free()
