import inspect
from typing import List, Tuple

import pennylane as qml


def _gate_support_status(op_cls) -> Tuple[bool, str]:
    name = op_cls.__name__
    if not hasattr(op_cls, "num_wires"):
        return False, f"{name}: missing num_wires"
    num_wires = op_cls.num_wires
    if num_wires == 0:
        return False, f"{name}: num_wires=0"
    if num_wires in (qml.operation.WiresEnum.AnyWires, qml.operation.WiresEnum.AllWires):
        return False, f"{name}: AnyWires"
    if not isinstance(num_wires, int):
        return False, f"{name}: num_wires not int"
    try:
        op = op_cls(wires=list(range(int(num_wires))))
        _ = op.matrix()
    except Exception as exc:
        return False, f"{name}: matrix() error -> {exc.__class__.__name__}"
    return True, f"{name}: supported ({num_wires} wires)"


def _iter_qubit_ops():
    ops_attr = getattr(qml.ops.qubit, "ops", None)
    if isinstance(ops_attr, (list, tuple, set)):
        for op_cls in ops_attr:
            if inspect.isclass(op_cls) and issubclass(op_cls, qml.operation.Operation):
                yield op_cls
        return

    for _, op_cls in inspect.getmembers(qml.ops.qubit, inspect.isclass):
        if not issubclass(op_cls, qml.operation.Operation):
            continue
        if op_cls is qml.operation.Operation:
            continue
        yield op_cls


def scan_ops() -> Tuple[List[str], List[str]]:
    supported: List[str] = []
    unsupported: List[str] = []
    for op_cls in _iter_qubit_ops():
        ok, msg = _gate_support_status(op_cls)
        if ok:
            supported.append(msg)
        else:
            unsupported.append(msg)
    return supported, unsupported


def main() -> None:
    supported, unsupported = scan_ops()
    print("Supported:")
    for item in supported:
        print(" -", item)
    print("\nUnsupported:")
    for item in unsupported:
        print(" -", item)


if __name__ == "__main__":
    main()

