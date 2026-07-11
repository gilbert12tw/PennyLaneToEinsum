import sys
import types

import numpy as np
import pytest

from pennylane_einsum.cuquantum_backend import (
    CuQuantumContractor,
    _integer_operands,
)


def test_integer_operands_support_arbitrary_labels():
    operands = _integer_operands("αβ,βγ->αγ", [np.eye(2), np.eye(2)])

    assert operands[1] == [0, 1]
    assert operands[3] == [1, 2]
    assert operands[4] == [0, 2]


def test_integer_operands_reject_mismatched_tensor_count():
    with pytest.raises(ValueError, match="tensor count"):
        _integer_operands("i,j->ij", [np.ones(2)])


def test_reusable_contractor_lifecycle(monkeypatch):
    calls = []

    class FakeNetwork:
        def __init__(self, *operands, options, stream):
            calls.append(("init", operands, options, stream))

        def contract_path(self, optimize=None):
            calls.append(("path", optimize))
            return [], object()

        def autotune(self, **kwargs):
            calls.append(("autotune", kwargs))

        def contract(self, **kwargs):
            calls.append(("contract", kwargs))
            return "result"

        def reset_operands(self, *operands, stream=None):
            calls.append(("reset", operands, stream))

        def free(self):
            calls.append(("free",))

    fake_package = types.ModuleType("cuquantum")
    fake_package.__path__ = []
    fake_tensornet = types.ModuleType("cuquantum.tensornet")
    fake_tensornet.Network = FakeNetwork
    monkeypatch.setitem(sys.modules, "cuquantum", fake_package)
    monkeypatch.setitem(sys.modules, "cuquantum.tensornet", fake_tensornet)

    contractor = CuQuantumContractor("ij,jk->ik", [np.eye(2), np.eye(2)])
    with pytest.raises(RuntimeError, match="contract_path"):
        contractor.contract()
    contractor.contract_path(optimize={"samples": 10})
    contractor.autotune(iterations=2)
    assert contractor.contract() == "result"
    contractor.reset_operands([np.ones((2, 2)), np.ones((2, 2))])
    contractor.free()

    assert [call[0] for call in calls] == [
        "init",
        "path",
        "autotune",
        "contract",
        "reset",
        "free",
    ]
