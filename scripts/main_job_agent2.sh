#!/bin/bash
#SBATCH -J soccer_a2
#SBATCH -A gts-crozell3
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
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
PYTHON_BIN=$VENV_PATH/bin/python
LOCAL_DIR=/storage/scratch1/9/hzhang3050/soccer_twos_results/agent2

TIMESTEPS=5000000
NUM_WORKERS=4
NUM_GPUS=1
NUM_ENVS_PER_WORKER=3

export SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export TMPDIR=/storage/scratch1/9/hzhang3050/tmp
export PIP_CACHE_DIR=/storage/scratch1/9/hzhang3050/pip-cache

cd "$REPO"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python environment not found at: $PYTHON_BIN"
    exit 1
fi

mkdir -p "$LOCAL_DIR"

echo "=== SoccerTwos Agent 2 Job ==="
echo "Host: $(hostname)"
echo "Repo: $REPO"
echo "Python: $PYTHON_BIN"
echo "Results dir: $LOCAL_DIR"

"$PYTHON_BIN" -u train_agent2_reward_player.py \
    --timesteps "$TIMESTEPS" \
    --num-workers "$NUM_WORKERS" \
    --num-gpus "$NUM_GPUS" \
    --num-envs-per-worker "$NUM_ENVS_PER_WORKER" \
    --local-dir "$LOCAL_DIR"

echo "=== AGENT 2 DONE ==="
