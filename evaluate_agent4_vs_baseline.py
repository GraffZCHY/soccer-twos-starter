import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import get_trainable_cls
from ray.rllib.env.base_env import BaseEnv

from reward_wrappers import DEFAULT_SHAPING, create_reward_shaped_rllib_env
from utils import create_rllib_env, get_multiagent_player_variation


TRAIN_POLICY_ID = "default"
BASELINE_POLICY_ID = "default"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Agent4 best checkpoint vs baseline.")
    parser.add_argument("--agent4-checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--episodes-per-batch", type=int, default=10)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--base-port", type=int, default=56000)
    return parser.parse_args()


def resolve_checkpoint_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    candidates = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.name.startswith("checkpoint-")
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* file found in: {path}")
    return candidates[0]


def load_policy_state_from_checkpoint(checkpoint_path: Path, policy_id: str):
    with checkpoint_path.open("rb") as f:
        checkpoint_data = pickle.load(f)
    worker_state = pickle.loads(checkpoint_data["worker"])
    policy_state = dict(worker_state["state"][policy_id])
    policy_state.pop("_optimizer_variables", None)
    return policy_state


def build_agent4_policy(agent4_checkpoint: Path):
    variation = get_multiagent_player_variation()
    env_name = "SoccerAgent4EvalShaped"
    tune.registry.register_env(env_name, create_reward_shaped_rllib_env)

    temp_env = create_reward_shaped_rllib_env(
        {
            "variation": variation,
            "reward_shaping": DEFAULT_SHAPING,
            "reward_scale": 1.0,
        }
    )
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    trainer = PPOTrainer(
        env=env_name,
        config={
            "num_gpus": 0,
            "num_workers": 0,
            "log_level": "ERROR",
            "framework": "torch",
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "multiagent": {
                "policies": {
                    TRAIN_POLICY_ID: (
                        None,
                        obs_space,
                        act_space,
                        {},
                    ),
                },
                "policy_mapping_fn": tune.function(lambda *_: TRAIN_POLICY_ID),
                "policies_to_train": [TRAIN_POLICY_ID],
            },
            "env_config": {
                "variation": variation,
                "reward_shaping": DEFAULT_SHAPING,
                "reward_scale": 1.0,
            },
        },
    )
    trainer.get_policy(TRAIN_POLICY_ID).set_state(
        load_policy_state_from_checkpoint(agent4_checkpoint, TRAIN_POLICY_ID)
    )
    return trainer.get_policy(TRAIN_POLICY_ID), trainer


def build_baseline_policy(baseline_checkpoint: Path):
    config_path = baseline_checkpoint.parent / "params.pkl"
    if not config_path.exists():
        config_path = baseline_checkpoint.parent.parent / "params.pkl"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find params.pkl for baseline near {baseline_checkpoint}")

    with config_path.open("rb") as f:
        config = pickle.load(f)

    config["num_workers"] = 0
    config["num_gpus"] = 0
    config["env"] = "DummyEnv"
    tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())

    cls = get_trainable_cls("PPO")
    trainer = cls(env=config["env"], config=config)
    trainer.restore(str(baseline_checkpoint))
    return trainer.get_policy(BASELINE_POLICY_ID), trainer


def compute_action(policy, obs):
    action, *_ = policy.compute_single_action(obs, explore=False)
    return action


def play_match(train_policy, baseline_policy, episodes: int, base_port: int):
    variation = get_multiagent_player_variation()
    env = create_rllib_env(
        {
            "variation": variation,
            "base_port": base_port,
        }
    )

    wins = 0
    losses = 0
    draws = 0
    reward_sum = 0.0

    try:
        for episode_idx in range(episodes):
            obs = env.reset()
            train_blue = episode_idx % 2 == 0
            done = {"__all__": False}
            train_team_reward = 0.0
            baseline_team_reward = 0.0

            while not done["__all__"]:
                if train_blue:
                    train_obs = {0: obs[0], 1: obs[1]}
                    baseline_obs = {0: obs[2], 1: obs[3]}
                else:
                    train_obs = {0: obs[2], 1: obs[3]}
                    baseline_obs = {0: obs[0], 1: obs[1]}

                train_actions = {
                    player_id: compute_action(train_policy, player_obs)
                    for player_id, player_obs in train_obs.items()
                }
                baseline_actions = {
                    player_id: compute_action(baseline_policy, player_obs)
                    for player_id, player_obs in baseline_obs.items()
                }

                if train_blue:
                    action_dict = {
                        0: train_actions[0],
                        1: train_actions[1],
                        2: baseline_actions[0],
                        3: baseline_actions[1],
                    }
                else:
                    action_dict = {
                        0: baseline_actions[0],
                        1: baseline_actions[1],
                        2: train_actions[0],
                        3: train_actions[1],
                    }

                obs, reward, done, _info = env.step(action_dict)

                if train_blue:
                    train_team_reward = reward[0] + reward[1]
                    baseline_team_reward = reward[2] + reward[3]
                else:
                    train_team_reward = reward[2] + reward[3]
                    baseline_team_reward = reward[0] + reward[1]

            reward_sum += train_team_reward
            if train_team_reward > baseline_team_reward:
                wins += 1
            elif train_team_reward < baseline_team_reward:
                losses += 1
            else:
                draws += 1
    finally:
        env.close()

    return {
        "episodes": episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / max(1, episodes),
        "mean_reward": reward_sum / max(1, episodes),
    }


def main():
    args = parse_args()
    agent4_checkpoint = resolve_checkpoint_path(args.agent4_checkpoint)
    baseline_checkpoint = resolve_checkpoint_path(args.baseline_checkpoint)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=False,
            local_mode=True,
            num_cpus=1,
            num_gpus=0,
        )

    train_policy, train_trainer = build_agent4_policy(agent4_checkpoint)
    baseline_policy, baseline_trainer = build_baseline_policy(baseline_checkpoint)

    results = []
    try:
        for batch_idx in range(args.batches):
            result = play_match(
                train_policy=train_policy,
                baseline_policy=baseline_policy,
                episodes=args.episodes_per_batch,
                base_port=args.base_port + batch_idx * 10,
            )
            result["batch"] = batch_idx + 1
            results.append(result)
            print(json.dumps(result, ensure_ascii=True), flush=True)
    finally:
        train_trainer.stop()
        baseline_trainer.stop()
        ray.shutdown()

    total_episodes = sum(item["episodes"] for item in results)
    total_wins = sum(item["wins"] for item in results)
    total_losses = sum(item["losses"] for item in results)
    total_draws = sum(item["draws"] for item in results)
    mean_reward = sum(item["mean_reward"] * item["episodes"] for item in results) / max(1, total_episodes)
    summary = {
        "total_episodes": total_episodes,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "total_draws": total_draws,
        "overall_win_rate": total_wins / max(1, total_episodes),
        "overall_mean_reward": mean_reward,
    }
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
