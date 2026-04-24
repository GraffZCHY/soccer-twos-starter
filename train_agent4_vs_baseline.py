import argparse
import json
import pickle
import random
import shutil
from datetime import datetime
from pathlib import Path

import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.tune import function
from ray.tune import Trainable
from ray.tune.logger import UnifiedLogger

from reward_wrappers import DEFAULT_SHAPING, create_reward_shaped_rllib_env
from utils import create_rllib_env, get_multiagent_player_variation


TRAIN_POLICY_ID = "default"
BASELINE_POLICY_ID = "baseline_fixed"
SELF_POLICY_IDS = ("self_1", "self_2", "self_3")


class LocalGPUTrainPPOTorchPolicy(PPOTorchPolicy):
    """Use GPU only on the trainer's local worker, CPU on rollout workers."""

    def __init__(self, observation_space, action_space, config):
        config = dict(config)
        if config.get("worker_index", 0) != 0:
            config["num_gpus"] = 0
        super().__init__(observation_space, action_space, config)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Agent 4: warm-start from Agent 2 and target the CEIA baseline."
    )
    parser.add_argument("--timesteps", type=int, default=8_000_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--num-envs-per-worker", type=int, default=3)
    parser.add_argument("--local-dir", default="./ray_results")
    parser.add_argument("--experiment-name", default="agent4_vs_baseline_ppo")
    parser.add_argument(
        "--agent2-checkpoint",
        required=True,
        help="Path to Agent 2 checkpoint file or containing directory.",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        default=str(
            Path(__file__).resolve().parent
            / "ceia_baseline_agent"
            / "ray_results"
            / "PPO_selfplay_twos"
            / "PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02"
            / "checkpoint_002449"
            / "checkpoint-2449"
        ),
        help="Path to CEIA baseline checkpoint file or containing directory.",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--archive-interval", type=int, default=25)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-base-port", type=int, default=55000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rollout-fragment-length", type=int, default=400)
    parser.add_argument("--train-batch-size", type=int, default=8000)
    parser.add_argument("--sgd-minibatch-size", type=int, default=1024)
    parser.add_argument("--num-sgd-iter", type=int, default=10)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--final-reward-scale", type=float, default=0.25)
    parser.add_argument(
        "--reward-anneal-frac",
        type=float,
        default=0.6,
        help="Fraction of total timesteps over which shaping decays to final scale.",
    )
    parser.add_argument("--baseline-prob", type=float, default=0.40)
    parser.add_argument("--self1-prob", type=float, default=0.30)
    parser.add_argument("--self2-prob", type=float, default=0.20)
    parser.add_argument("--self3-prob", type=float, default=0.10)
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


def load_policy_weights(checkpoint_path: Path, policy_id: str):
    with checkpoint_path.open("rb") as f:
        checkpoint_data = pickle.load(f)
    worker_state = pickle.loads(checkpoint_data["worker"])
    policy_state = dict(worker_state["state"][policy_id])
    policy_state.pop("_optimizer_variables", None)
    return policy_state


def build_policy_mapping_fn(probabilities):
    opponent_ids = [BASELINE_POLICY_ID, *SELF_POLICY_IDS]
    weights = [
        probabilities["baseline"],
        probabilities["self_1"],
        probabilities["self_2"],
        probabilities["self_3"],
    ]

    def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
        if episode is None:
            return TRAIN_POLICY_ID if agent_id in (0, 1) else BASELINE_POLICY_ID

        train_blue = episode.user_data.get("train_blue")
        if train_blue is None:
            train_blue = random.random() < 0.5
            episode.user_data["train_blue"] = train_blue
            episode.user_data["opponent_policy"] = random.choices(
                opponent_ids,
                weights=weights,
                k=1,
            )[0]

        is_blue_agent = agent_id in (0, 1)
        if is_blue_agent == episode.user_data["train_blue"]:
            return TRAIN_POLICY_ID
        return episode.user_data["opponent_policy"]

    return policy_mapping_fn


def compute_reward_scale(current_timesteps: int, total_timesteps: int, final_scale: float, anneal_frac: float):
    anneal_steps = max(1, int(total_timesteps * anneal_frac))
    progress = min(1.0, current_timesteps / anneal_steps)
    return 1.0 - progress * (1.0 - final_scale)


def set_env_reward_scale(env, scale: float):
    target = env
    while target is not None:
        if hasattr(target, "set_reward_scale"):
            target.set_reward_scale(scale)
            return
        target = getattr(target, "env", None)


def sanitize_for_storage(value):
    if isinstance(value, dict):
        return {key: sanitize_for_storage(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_storage(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def evaluate_vs_baseline(trainer: PPOTrainer, episodes: int, base_port: int):
    variation = get_multiagent_player_variation()
    env = create_rllib_env(
        {
            "variation": variation,
            "base_port": base_port,
        }
    )

    default_policy = trainer.get_policy(TRAIN_POLICY_ID)
    baseline_policy = trainer.get_policy(BASELINE_POLICY_ID)

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
                    player_id: default_policy.compute_single_action(player_obs, explore=False)[0]
                    for player_id, player_obs in train_obs.items()
                }
                baseline_actions = {
                    player_id: baseline_policy.compute_single_action(player_obs, explore=False)[0]
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

                obs, reward, done, info = env.step(action_dict)

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


def copy_checkpoint_marker(checkpoint_file: str, target_dir: Path):
    checkpoint_file = Path(checkpoint_file).resolve()
    checkpoint_dir = checkpoint_file.parent
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(checkpoint_dir, target_dir)


class Agent4Trainable(Trainable):
    def setup(self, config):
        self.args = dict(config)
        self.variation = get_multiagent_player_variation()
        self.agent2_checkpoint = resolve_checkpoint_path(self.args["agent2_checkpoint"])
        self.baseline_checkpoint = resolve_checkpoint_path(self.args["baseline_checkpoint"])
        self.eval_jsonl = Path(self.logdir) / "baseline_eval.jsonl"
        self.best_eval = None

        probabilities = {
            "baseline": self.args["baseline_prob"],
            "self_1": self.args["self1_prob"],
            "self_2": self.args["self2_prob"],
            "self_3": self.args["self3_prob"],
        }
        if abs(sum(probabilities.values()) - 1.0) > 1e-6:
            raise ValueError(f"Opponent probabilities must sum to 1.0, got {probabilities}")

        self.env_name = "SoccerAgent4VsBaseline"
        ray.tune.registry.register_env(self.env_name, create_reward_shaped_rllib_env)

        temp_env = create_reward_shaped_rllib_env(
            {
                "variation": self.variation,
                "reward_shaping": DEFAULT_SHAPING,
                "reward_scale": 1.0,
            }
        )
        obs_space = temp_env.observation_space
        act_space = temp_env.action_space
        temp_env.close()

        self.rllib_config = {
            "num_gpus": self.args["num_gpus"],
            "num_workers": self.args["num_workers"],
            "num_envs_per_worker": self.args["num_envs_per_worker"],
            "log_level": "INFO",
            "framework": "torch",
            "multiagent": {
                "policies": {
                    TRAIN_POLICY_ID: (
                        LocalGPUTrainPPOTorchPolicy,
                        obs_space,
                        act_space,
                        {
                            "framework": "torch",
                            "num_gpus": self.args["num_gpus"],
                        },
                    ),
                    BASELINE_POLICY_ID: (
                        None,
                        obs_space,
                        act_space,
                        {
                            "framework": "torch",
                            "num_gpus": 0,
                        },
                    ),
                    "self_1": (
                        None,
                        obs_space,
                        act_space,
                        {
                            "framework": "torch",
                            "num_gpus": 0,
                        },
                    ),
                    "self_2": (
                        None,
                        obs_space,
                        act_space,
                        {
                            "framework": "torch",
                            "num_gpus": 0,
                        },
                    ),
                    "self_3": (
                        None,
                        obs_space,
                        act_space,
                        {
                            "framework": "torch",
                            "num_gpus": 0,
                        },
                    ),
                },
                "policy_mapping_fn": function(build_policy_mapping_fn(probabilities)),
                "policies_to_train": [TRAIN_POLICY_ID],
            },
            "env": self.env_name,
            "env_config": {
                "num_envs_per_worker": self.args["num_envs_per_worker"],
                "variation": self.variation,
                "reward_shaping": DEFAULT_SHAPING,
                "reward_scale": 1.0,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "lambda": self.args["gae_lambda"],
            "clip_param": self.args["clip_param"],
            "lr": self.args["lr"],
            "rollout_fragment_length": self.args["rollout_fragment_length"],
            "train_batch_size": self.args["train_batch_size"],
            "sgd_minibatch_size": self.args["sgd_minibatch_size"],
            "num_sgd_iter": self.args["num_sgd_iter"],
            "batch_mode": "complete_episodes",
        }

        stored_config = sanitize_for_storage(self.rllib_config)
        with (Path(self.logdir) / "agent4_rllib_config.json").open("w", encoding="utf-8") as f:
            json.dump(stored_config, f, indent=2, default=str)

        def logger_creator(trainer_config):
            inner_logdir = Path(self.logdir) / "trainer"
            inner_logdir.mkdir(parents=True, exist_ok=True)
            return UnifiedLogger(trainer_config, str(inner_logdir), loggers=None)

        self.trainer = PPOTrainer(
            env=self.env_name,
            config=self.rllib_config,
            logger_creator=logger_creator,
        )

        agent2_weights = load_policy_weights(self.agent2_checkpoint, "shared_policy")
        baseline_weights = load_policy_weights(self.baseline_checkpoint, "default")
        self.trainer.set_weights(
            {
                TRAIN_POLICY_ID: agent2_weights,
                BASELINE_POLICY_ID: baseline_weights,
                "self_1": agent2_weights,
                "self_2": agent2_weights,
                "self_3": agent2_weights,
            }
        )

    def step(self):
        result = self.trainer.train()
        iteration = int(result["training_iteration"])
        timesteps_total = int(result.get("timesteps_total") or 0)

        reward_scale = compute_reward_scale(
            timesteps_total,
            self.args["timesteps"],
            self.args["final_reward_scale"],
            self.args["reward_anneal_frac"],
        )
        self.trainer.workers.foreach_worker(
            lambda worker: worker.foreach_env(
                lambda env: set_env_reward_scale(env, reward_scale)
            )
        )
        result["reward_scale"] = reward_scale

        if iteration % self.args["archive_interval"] == 0:
            current_weights = self.trainer.get_weights([TRAIN_POLICY_ID])[TRAIN_POLICY_ID]
            previous_self_1 = self.trainer.get_weights(["self_1"])["self_1"]
            previous_self_2 = self.trainer.get_weights(["self_2"])["self_2"]
            self.trainer.set_weights(
                {
                    "self_3": previous_self_2,
                    "self_2": previous_self_1,
                    "self_1": current_weights,
                }
            )
            print(f"Archived current policy into self-play pool at iteration {iteration}.")

        if iteration % self.args["eval_interval"] == 0:
            evaluation = evaluate_vs_baseline(
                self.trainer,
                episodes=self.args["eval_episodes"],
                base_port=self.args["eval_base_port"],
            )
            evaluation["training_iteration"] = iteration
            evaluation["timesteps_total"] = timesteps_total
            with self.eval_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(evaluation) + "\n")
            print(f"Baseline evaluation: {evaluation}")

            result.update(
                {
                    "baseline_eval_episodes": evaluation["episodes"],
                    "baseline_eval_wins": evaluation["wins"],
                    "baseline_eval_losses": evaluation["losses"],
                    "baseline_eval_draws": evaluation["draws"],
                    "baseline_eval_win_rate": evaluation["win_rate"],
                    "baseline_eval_mean_reward": evaluation["mean_reward"],
                }
            )

            if self.best_eval is None or evaluation["win_rate"] > self.best_eval["win_rate"]:
                self.best_eval = evaluation
                checkpoint_path = self.trainer.save()
                copy_checkpoint_marker(checkpoint_path, Path(self.logdir) / "best_vs_baseline")
                with (Path(self.logdir) / "best_vs_baseline.json").open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "evaluation": evaluation,
                            "checkpoint": checkpoint_path,
                        },
                        f,
                        indent=2,
                    )
                print(
                    "Updated best-vs-baseline checkpoint "
                    f"(win_rate={evaluation['win_rate']:.3f})."
                )

        return result

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = self.trainer.save(checkpoint_dir)
        metadata_path = Path(str(checkpoint_path) + ".agent4_meta.pkl")
        with metadata_path.open("wb") as f:
            pickle.dump({"best_eval": self.best_eval}, f)
        return checkpoint_path

    def load_checkpoint(self, checkpoint):
        self.trainer.restore(checkpoint)
        metadata_path = Path(str(checkpoint) + ".agent4_meta.pkl")
        if metadata_path.exists():
            with metadata_path.open("rb") as f:
                state = pickle.load(f)
            self.best_eval = state.get("best_eval")

    def cleanup(self):
        if hasattr(self, "trainer"):
            final_checkpoint = self.trainer.save()
            print(f"Final checkpoint: {final_checkpoint}")
            self.trainer.stop()


def main():
    args = parse_args()
    args.agent2_checkpoint = str(Path(args.agent2_checkpoint).expanduser().resolve())
    args.baseline_checkpoint = str(Path(args.baseline_checkpoint).expanduser().resolve())
    args.local_dir = str(Path(args.local_dir).expanduser().resolve())
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        num_gpus=args.num_gpus,
    )

    trainable_config = vars(args).copy()
    analysis = ray.tune.run(
        Agent4Trainable,
        name=args.experiment_name,
        config=trainable_config,
        stop={"timesteps_total": args.timesteps},
        checkpoint_freq=args.checkpoint_interval,
        checkpoint_at_end=True,
        local_dir=args.local_dir,
        resources_per_trial={
            "cpu": 1,
            "gpu": args.num_gpus,
        },
    )

    best_trial = analysis.get_best_trial("baseline_eval_win_rate", mode="max", scope="all")
    if best_trial is not None:
        print(best_trial)
        best_checkpoint = analysis.get_best_checkpoint(
            trial=best_trial,
            metric="baseline_eval_win_rate",
            mode="max",
        )
        print(best_checkpoint)
    print("Done training Agent 4 vs baseline.")
    ray.shutdown()


if __name__ == "__main__":
    main()
