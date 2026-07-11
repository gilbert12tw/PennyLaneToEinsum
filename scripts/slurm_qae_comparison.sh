#!/bin/bash
#SBATCH --job-name=qae-compare
#SBATCH --account=gov109211
#SBATCH --partition=dev
#SBATCH --constraint=H200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=docs/benchmarks/slurm-compare-%j.out
#SBATCH --error=docs/benchmarks/slurm-compare-%j.err

set -euo pipefail

module purge
module load cuda/12.6
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT=".venv-slurm-${SLURM_JOB_ID}"

uv run --python 3.11 --extra cuquantum --extra benchmark python -c \
  "import quimb.tensor, pennylane; print('pennylane', pennylane.__version__, 'quimb import ok')"

uv run --python 3.11 --extra cuquantum --extra benchmark \
  scripts/benchmark_qae_backends.py \
  --qubits 4 8 12 16 20 24 \
  --layers 1 --repeats 3 \
  --output docs/benchmarks/qae_backend_comparison_${SLURM_JOB_ID}.csv
