import argparse
import json
import os

import ray

from evaluate_agent4_vs_baseline import build_agent4_policy, resolve_checkpoint_path
from example_player_agent import RandomAgent
from utils import create_rllib_env, get_multiagent_player_variation


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Agent4 checkpoint vs random agent.")
    parser.add_argument("--agent4-checkpoint", required=True)
    parser.add_argument("--episodes-per-batch", type=int, default=10)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--base-port", type=int, default=56300)
    return parser.parse_args()


def compute_action(policy, obs):
    action, *_ = policy.compute_single_action(obs, explore=False)
    return action


def play_match(train_policy, episodes: int, base_port: int):
    variation = get_multiagent_player_variation()
    env = create_rllib_env(
        {
            "variation": variation,
            "base_port": base_port,
        }
    )
    random_agent = RandomAgent(env)

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
            random_team_reward = 0.0

            while not done["__all__"]:
                if train_blue:
                    train_obs = {0: obs[0], 1: obs[1]}
                    random_obs = {0: obs[2], 1: obs[3]}
                else:
                    train_obs = {0: obs[2], 1: obs[3]}
                    random_obs = {0: obs[0], 1: obs[1]}

                train_actions = {
                    player_id: compute_action(train_policy, player_obs)
                    for player_id, player_obs in train_obs.items()
                }
                random_actions = random_agent.act(random_obs)

                if train_blue:
                    action_dict = {
                        0: train_actions[0],
                        1: train_actions[1],
                        2: random_actions[0],
                        3: random_actions[1],
                    }
                else:
                    action_dict = {
                        0: random_actions[0],
                        1: random_actions[1],
                        2: train_actions[0],
                        3: train_actions[1],
                    }

                obs, reward, done, _info = env.step(action_dict)

                if train_blue:
                    train_team_reward = reward[0] + reward[1]
                    random_team_reward = reward[2] + reward[3]
                else:
                    train_team_reward = reward[2] + reward[3]
                    random_team_reward = reward[0] + reward[1]

            reward_sum += train_team_reward
            if train_team_reward > random_team_reward:
                wins += 1
            elif train_team_reward < random_team_reward:
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

    results = []
    try:
        for batch_idx in range(args.batches):
            result = play_match(
                train_policy=train_policy,
                episodes=args.episodes_per_batch,
                base_port=args.base_port + batch_idx * 10,
            )
            result["batch"] = batch_idx + 1
            results.append(result)
            print(json.dumps(result, ensure_ascii=True), flush=True)
    finally:
        train_trainer.stop()
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
