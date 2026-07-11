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

## NCHC nano4 Slurm

- This workspace is on a login node. Never run GPU workloads directly; submit
  them with `sbatch` and inspect them with `squeue`/`sacct`.
- Charge project jobs to the NCHC internal wallet with
  `--account=gov109211` (project `GOV109211`). Do not use the student
  competition wallet unless the user explicitly requests it.
- The verified short-job partition is `dev` (maximum 4 hours). The GPU nodes
  currently advertise `H200` and `gpu:H200:8`.
- Default to one GPU. A typical allocation is
  `--partition=dev --constraint=H200 --gres=gpu:H200:1 --cpus-per-task=8`.
  Use two GPUs only when the benchmark explicitly supports multi-GPU work.
- Load `cuda/12.6` in CUDA jobs. CUDA 13.0 is the system default, but the
  project cuQuantum/CuPy environment targets CUDA 12.x.
- The system `python3` is 3.9, while current cuQuantum packages require 3.11.
  Run cuQuantum jobs with uv-managed Python 3.11 (`uv run --python 3.11
  --extra cuquantum ...`). Keep the base package compatible with Python 3.9.
- Use `uv` inside jobs and keep benchmark output under `docs/benchmarks/` or a
  dedicated Slurm log directory in the project.
- Before submitting a new job, check `squeue -u "$USER"`. After completion,
  record the job ID and inspect `sacct -j <job-id> --format=JobID,State,Elapsed,MaxRSS,AllocTRES,ExitCode`.

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
