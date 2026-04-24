#!/bin/bash
#SBATCH -J soccer_a4
#SBATCH -A gts-crozell3
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
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
LOCAL_DIR=/storage/scratch1/9/hzhang3050/soccer_twos_results/agent4

TIMESTEPS=8000000
NUM_WORKERS=4
NUM_GPUS=1
NUM_ENVS_PER_WORKER=3
EVAL_EPISODES=20

AGENT2_CHECKPOINT="${AGENT2_CHECKPOINT:-/storage/home/hcoda1/9/hzhang3050/scratch/soccer_twos_results/agent2/agent2_reward_player_ppo/PPO_SoccerRewardShaped_1faf5_00000_0_2026-04-18_22-40-28/checkpoint_000525}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-$REPO/ceia_baseline_agent/ray_results/PPO_selfplay_twos/PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/checkpoint_002449/checkpoint-2449}"

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

if [ ! -e "$AGENT2_CHECKPOINT" ]; then
    echo "Agent 2 checkpoint path not found: $AGENT2_CHECKPOINT"
    exit 1
fi

if [ ! -e "$BASELINE_CHECKPOINT" ]; then
    echo "Baseline checkpoint path not found: $BASELINE_CHECKPOINT"
    exit 1
fi

mkdir -p "$LOCAL_DIR"

echo "=== SoccerTwos Agent 4 Job ==="
echo "Host: $(hostname)"
echo "Repo: $REPO"
echo "Python: $PYTHON_BIN"
echo "Results dir: $LOCAL_DIR"
echo "Agent2 checkpoint: $AGENT2_CHECKPOINT"
echo "Baseline checkpoint: $BASELINE_CHECKPOINT"

"$PYTHON_BIN" -u train_agent4_vs_baseline.py \
    --timesteps "$TIMESTEPS" \
    --num-workers "$NUM_WORKERS" \
    --num-gpus "$NUM_GPUS" \
    --num-envs-per-worker "$NUM_ENVS_PER_WORKER" \
    --local-dir "$LOCAL_DIR" \
    --agent2-checkpoint "$AGENT2_CHECKPOINT" \
    --baseline-checkpoint "$BASELINE_CHECKPOINT" \
    --eval-episodes "$EVAL_EPISODES"

echo "=== AGENT 4 DONE ==="
