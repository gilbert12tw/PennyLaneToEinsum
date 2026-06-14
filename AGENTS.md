# AGENTS.md

## Project Overview

This package converts PennyLane circuits into dense tensor-network einsum
expressions. The converter reads `op.matrix()` from PennyLane operations,
assembles einsum indices with `IndexManager`, and contracts with `opt_einsum`.

The current design does not preserve autodiff through circuit gate parameters.

## Commands

- Run full tests: `uv run --extra dev pytest -q`
- Run one test file: `uv run --extra dev pytest tests/test_method_one_basic.py -q`
- Run one test: `uv run --extra dev pytest tests/test_method_one_basic.py::test_two_qubit_entangling -q`
- Run examples: `uv run python examples/basic_circuits.py`

## Development Notes

- Prefer `uv run` commands so the lockfile-managed environment is used.
- Use `rg` for code search.
- Keep converter changes dense-matrix based unless the task explicitly asks for
  decomposition or sparse support.
- Use PennyLane `qml.state()` as the correctness oracle for statevector
  conversion tests.
- Preserve existing unbatched behavior when adding batched conversion support.

## Testing Guidance

- Add focused tests for every converter behavior change.
- Compare numerical statevectors against PennyLane per sample when testing
  batched parameters.
- Run `uv run --extra dev pytest -q` before committing.
