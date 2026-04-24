import argparse

import ray
from ray import tune

from reward_wrappers import DEFAULT_SHAPING, create_reward_shaped_rllib_env
from utils import get_multiagent_player_variation


DEFAULT_TIMESTEPS = 5_000_000
DEFAULT_NUM_ENVS_PER_WORKER = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Agent 2: PPO with reward shaping on SoccerTwos."
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--num-envs-per-worker", type=int, default=DEFAULT_NUM_ENVS_PER_WORKER)
    parser.add_argument("--local-dir", default="./ray_results")
    parser.add_argument("--experiment-name", default="agent2_reward_player_ppo")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    variation = get_multiagent_player_variation()

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    tune.registry.register_env("SoccerRewardShaped", create_reward_shaped_rllib_env)
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
            "multiagent": {
                "policies": {
                    "shared_policy": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(lambda *_: "shared_policy"),
                "policies_to_train": ["shared_policy"],
            },
            "env": "SoccerRewardShaped",
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
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training Agent 2 with reward shaping.")
