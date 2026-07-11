import sys
import types

import numpy as np
import pytest

from pennylane_einsum import contract_einsum


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown contraction backend"):
        contract_einsum("i->i", [np.ones(2)], backend="unknown")


def test_cuquantum_backend_moves_tensors_and_contracts(monkeypatch):
    calls = {}
    fake_cupy = types.ModuleType("cupy")
    fake_cupy.asarray = lambda value: np.asarray(value)
    fake_cuquantum = types.ModuleType("cuquantum")
    fake_cuquantum.__path__ = []
    fake_tensornet = types.ModuleType("cuquantum.tensornet")

    def fake_contract(*operands, **kwargs):
        calls.update(operands=operands, optimize=kwargs.get("optimize"))
        left, left_modes, right, right_modes, output_modes = operands
        assert left_modes == [0, 1]
        assert right_modes == [1, 2]
        assert output_modes == [0, 2]
        return left @ right

    fake_tensornet.contract = fake_contract
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "cuquantum", fake_cuquantum)
    monkeypatch.setitem(sys.modules, "cuquantum.tensornet", fake_tensornet)

    result = contract_einsum(
        "ij,jk->ik",
        [np.eye(2), np.array([[1.0, 2.0], [3.0, 4.0]])],
        optimize="greedy",
        backend="cuquantum",
    )

    np.testing.assert_allclose(result, [[1.0, 2.0], [3.0, 4.0]])
    assert calls["optimize"] == "greedy"
