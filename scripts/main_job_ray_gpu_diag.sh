#!/bin/bash
#SBATCH -J ray_gpu_diag
#SBATCH -A gts-crozell3
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o /storage/scratch1/9/hzhang3050/%x.%j.out
#SBATCH -e /storage/scratch1/9/hzhang3050/%x.%j.err

set -euo pipefail

module purge
module load cuda/12.1 2>/dev/null || true
module load gcc/10.2.0 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$DEFAULT_REPO}}"
VENV_PATH="${VENV_PATH:-/storage/scratch1/9/hzhang3050/venvs/soccer_gpu38}"
PYTHON_BIN="$VENV_PATH/bin/python"

export SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TMPDIR=/storage/scratch1/9/hzhang3050/tmp
export PIP_CACHE_DIR=/storage/scratch1/9/hzhang3050/pip-cache

cd "$REPO"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python environment not found at: $PYTHON_BIN"
    exit 1
fi

echo "=== Ray GPU Diagnostic Job ==="
echo "Host: $(hostname)"
echo "Repo: $REPO"
echo "Python: $PYTHON_BIN"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-<unset>}"

"$PYTHON_BIN" -u diagnose_ray_gpu_env.py

echo "=== RAY GPU DIAG DONE ==="
