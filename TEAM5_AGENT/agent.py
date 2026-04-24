import os
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.base_env import BaseEnv
from soccer_twos import AgentInterface

class TeamSubmissionAgent(AgentInterface):
    """
    Submission wrapper for the trained checkpoint.

    This package is intended to live in the TEAM5_AGENT folder for submission.
    """

    def __init__(self, env):
        super().__init__()
        self._observation_space = env.observation_space
        self._action_space = env.action_space
        self._policy_id = "default"

        checkpoint_path = Path(__file__).with_name("checkpoint-525")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if not ray.is_initialized():
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                log_to_driver=False,
                local_mode=True,
                num_cpus=1,
                num_gpus=0,
            )

        tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
        self._trainer = PPOTrainer(
            env="DummyEnv",
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
                        self._policy_id: (
                            None,
                            self._observation_space,
                            self._action_space,
                            {},
                        ),
                    },
                    "policy_mapping_fn": tune.function(lambda *_: self._policy_id),
                    "policies_to_train": [self._policy_id],
                },
            },
        )
        self._trainer.get_policy(self._policy_id).set_state(
            self._load_checkpoint_state(checkpoint_path, self._policy_id)
        )

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id, player_obs in observation.items():
            if hasattr(self._trainer, "compute_single_action"):
                action, *_ = self._trainer.compute_single_action(
                    player_obs,
                    policy_id=self._policy_id,
                    explore=False,
                )
            else:
                action = self._trainer.compute_action(
                    player_obs,
                    policy_id=self._policy_id,
                    explore=False,
                )
            actions[player_id] = action
        return actions

    @staticmethod
    def _load_checkpoint_state(checkpoint_path: Path, policy_id: str):
        with checkpoint_path.open("rb") as f:
            checkpoint_data = pickle.load(f)
        worker_state = pickle.loads(checkpoint_data["worker"])
        policy_state = dict(worker_state["state"][policy_id])
        policy_state.pop("_optimizer_variables", None)
        return policy_state
