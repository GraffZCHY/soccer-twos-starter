import argparse
import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks

from reward_wrappers import DEFAULT_SHAPING, create_reward_shaped_rllib_env
from utils import get_multiagent_player_variation


DEFAULT_TIMESTEPS = 5_000_000
DEFAULT_NUM_ENVS_PER_WORKER = 3
SELF_PLAY_REWARD_THRESHOLD = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Agent 3: reward-shaped PPO with self-play on SoccerTwos."
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument(
        "--num-envs-per-worker", type=int, default=DEFAULT_NUM_ENVS_PER_WORKER
    )
    parser.add_argument("--local-dir", default="./ray_results")
    parser.add_argument("--experiment-name", default="agent3_selfplay_reward_player_ppo")
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=SELF_PLAY_REWARD_THRESHOLD,
        help="Archive current policy into the opponent pool when mean reward exceeds this threshold.",
    )
    parser.add_argument(
        "--restore",
        default=None,
        help="Optional checkpoint path to warm-start training, e.g. from Agent 2.",
    )
    return parser.parse_args()


def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id in (0, 1):
        return "default"

    return np.random.choice(
        ["opponent_1", "opponent_2", "opponent_3"],
        size=1,
        p=[0.50, 0.30, 0.20],
    )[0]


class SelfPlayRewardUpdateCallback(DefaultCallbacks):
    def on_train_result(self, *, trainer, result, **kwargs):
        if result["episode_reward_mean"] > SELF_PLAY_REWARD_THRESHOLD:
            print("---- Updating self-play opponent pool ----")
            trainer.set_weights(
                {
                    "opponent_3": trainer.get_weights(["opponent_2"])["opponent_2"],
                    "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                    "opponent_1": trainer.get_weights(["default"])["default"],
                }
            )


if __name__ == "__main__":
    args = parse_args()
    variation = get_multiagent_player_variation()

    SELF_PLAY_REWARD_THRESHOLD = args.reward_threshold

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    tune.registry.register_env("SoccerRewardSelfPlay", create_reward_shaped_rllib_env)
    temp_env = create_reward_shaped_rllib_env(
        {
            "variation": variation,
            "reward_shaping": DEFAULT_SHAPING,
        }
    )
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name=args.experiment_name,
        config={
            "num_gpus": args.num_gpus,
            "num_workers": args.num_workers,
            "num_envs_per_worker": args.num_envs_per_worker,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": SelfPlayRewardUpdateCallback,
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["default"],
            },
            "env": "SoccerRewardSelfPlay",
            "env_config": {
                "num_envs_per_worker": args.num_envs_per_worker,
                "variation": variation,
                "reward_shaping": DEFAULT_SHAPING,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "lambda": 0.95,
            "clip_param": 0.2,
            "lr": 3e-4,
            "rollout_fragment_length": 200,
            "train_batch_size": 4000,
            "sgd_minibatch_size": 512,
            "num_sgd_iter": 10,
            "batch_mode": "complete_episodes",
        },
        stop={"timesteps_total": args.timesteps},
        checkpoint_freq=25,
        checkpoint_at_end=True,
        local_dir=args.local_dir,
        restore=args.restore,
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training Agent 3 with reward shaping and self-play.")
