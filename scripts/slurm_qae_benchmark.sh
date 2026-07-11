#!/bin/bash
#SBATCH --job-name=qae-cuquantum
#SBATCH --account=gov109211
#SBATCH --partition=dev
#SBATCH --constraint=H200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=docs/benchmarks/slurm-%j.out
#SBATCH --error=docs/benchmarks/slurm-%j.err

set -euo pipefail

module purge
module load cuda/12.6
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT=".venv-slurm-${SLURM_JOB_ID}"

nvidia-smi
uv run --python 3.11 --extra cuquantum --extra benchmark python -c \
  "import cupy, cuquantum, pennylane, quimb.tensor; print('cupy', cupy.__version__, 'cuquantum', cuquantum.__version__, 'pennylane', pennylane.__version__)"

uv run --python 3.11 --extra cuquantum --extra benchmark \
  scripts/benchmark_qae_backends.py \
  --qubits 4 8 12 16 20 24 \
  --layers 1 --repeats 3 \
  --output docs/benchmarks/qae_backend_comparison_${SLURM_JOB_ID}.csv

uv run --python 3.11 --extra cuquantum --extra benchmark \
  scripts/benchmark_qae_backends.py \
  --qubits 4 8 12 16 20 24 28 32 40 48 56 64 \
  --layers 1 --repeats 3 --methods cuquantum --measurements expval \
  --output docs/benchmarks/qae_cuquantum_expval_scaling_${SLURM_JOB_ID}.csv

uv run --python 3.11 --extra cuquantum --extra benchmark \
  scripts/benchmark_qae_backends.py \
  --qubits 4 8 12 16 20 24 28 30 32 \
  --layers 1 --repeats 3 --methods cuquantum --measurements state \
  --output docs/benchmarks/qae_cuquantum_state_scaling_${SLURM_JOB_ID}.csv
