from copy import deepcopy
from typing import Dict, Optional

import gym
import numpy as np

from utils import RLLibWrapper, load_soccer_twos


BLUE_TEAM = (0, 1)
ORANGE_TEAM = (2, 3)
DEFAULT_SHAPING = {
    "ball_progress_weight": 0.04,
    "player_to_ball_weight": 0.015,
    "defensive_clear_weight": 0.10,
    "step_penalty": 0.001,
    "danger_zone_x": 11.0,
    "clear_threshold": 1.5,
}


class RewardShapingWrapper(gym.Wrapper):
    """
    Adds dense reward terms on top of SoccerTwos' sparse goal reward.

    The wrapper is intentionally defensive about the contents of `info` because
    different environment variations expose slightly different structures.
    """

    def __init__(self, env: gym.Env, shaping_config: Optional[Dict] = None):
        super().__init__(env)
        self.shaping_config = deepcopy(DEFAULT_SHAPING)
        if shaping_config:
            self.shaping_config.update(shaping_config)
        self.reward_scale = 1.0
        self._previous_state = None

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self._previous_state = None
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        current_state = self._extract_state(info)
        shaped_reward = self._apply_shaping(reward, current_state)

        if self._is_done(done):
            self._previous_state = None
        else:
            self._previous_state = current_state

        return observation, shaped_reward, done, info

    def _apply_shaping(self, reward, current_state):
        if self._previous_state is None or current_state["ball_position"] is None:
            return reward

        blue_bonus = -self.shaping_config["step_penalty"]
        orange_bonus = -self.shaping_config["step_penalty"]

        previous_ball = self._previous_state["ball_position"]
        current_ball = current_state["ball_position"]

        if previous_ball is not None and current_ball is not None:
            delta_x = current_ball[0] - previous_ball[0]
            blue_bonus += self.shaping_config["ball_progress_weight"] * delta_x
            orange_bonus -= self.shaping_config["ball_progress_weight"] * delta_x

            danger_x = self.shaping_config["danger_zone_x"]
            clear_threshold = self.shaping_config["clear_threshold"]
            if previous_ball[0] < -danger_x and current_ball[0] > previous_ball[0] + clear_threshold:
                blue_bonus += self.shaping_config["defensive_clear_weight"]
            if previous_ball[0] > danger_x and current_ball[0] < previous_ball[0] - clear_threshold:
                orange_bonus += self.shaping_config["defensive_clear_weight"]

        blue_bonus += self._player_to_ball_bonus(BLUE_TEAM, current_state)
        orange_bonus += self._player_to_ball_bonus(ORANGE_TEAM, current_state)

        return self._merge_reward(
            reward,
            blue_bonus * self.reward_scale,
            orange_bonus * self.reward_scale,
        )

    def set_reward_scale(self, scale: float):
        self.reward_scale = float(scale)

    def _player_to_ball_bonus(self, team_ids, current_state):
        previous_ball = self._previous_state["ball_position"]
        current_ball = current_state["ball_position"]
        if previous_ball is None or current_ball is None:
            return 0.0

        previous_distance = self._nearest_distance(
            self._previous_state["player_positions"], previous_ball, team_ids
        )
        current_distance = self._nearest_distance(
            current_state["player_positions"], current_ball, team_ids
        )
        if previous_distance is None or current_distance is None:
            return 0.0

        improvement = previous_distance - current_distance
        return self.shaping_config["player_to_ball_weight"] * improvement

    def _merge_reward(self, reward, blue_bonus, orange_bonus):
        if isinstance(reward, dict):
            shaped_reward = dict(reward)
            numeric_keys = {key for key in shaped_reward if isinstance(key, int)}

            if BLUE_TEAM[0] in numeric_keys and ORANGE_TEAM[0] in numeric_keys:
                for agent_id in BLUE_TEAM:
                    if agent_id in shaped_reward:
                        shaped_reward[agent_id] += blue_bonus / len(BLUE_TEAM)
                for agent_id in ORANGE_TEAM:
                    if agent_id in shaped_reward:
                        shaped_reward[agent_id] += orange_bonus / len(ORANGE_TEAM)
                return shaped_reward

            team_keys = sorted(numeric_keys)
            if len(team_keys) == 2:
                shaped_reward[team_keys[0]] += blue_bonus
                shaped_reward[team_keys[1]] += orange_bonus
            return shaped_reward

        return reward + blue_bonus

    @staticmethod
    def _nearest_distance(player_positions, ball_position, team_ids):
        distances = []
        for player_id in team_ids:
            if player_id not in player_positions:
                continue
            distances.append(np.linalg.norm(player_positions[player_id] - ball_position))
        if not distances:
            return None
        return min(distances)

    @staticmethod
    def _is_done(done):
        if isinstance(done, dict):
            return done.get("__all__", False)
        return bool(done)

    def _extract_state(self, info):
        return {
            "ball_position": self._extract_ball_position(info),
            "player_positions": self._extract_player_positions(info),
        }

    def _extract_ball_position(self, info):
        ball_sources = [info]
        if isinstance(info, dict):
            ball_sources.extend(value for value in info.values() if isinstance(value, dict))

        for source in ball_sources:
            for key in ("ball_info", "ball", "ball_state"):
                if key in source:
                    position = self._extract_position(source[key])
                    if position is not None:
                        return position
        return None

    def _extract_player_positions(self, info):
        player_positions = {}
        if not isinstance(info, dict):
            return player_positions

        for key, value in info.items():
            if isinstance(key, int):
                position = self._extract_position(value)
                if position is not None:
                    player_positions[key] = position
                continue

            if key in ("players", "player_info", "players_info", "players_states") and isinstance(value, dict):
                for player_id, player_state in value.items():
                    if not isinstance(player_id, int):
                        continue
                    position = self._extract_position(player_state)
                    if position is not None:
                        player_positions[player_id] = position

        return player_positions

    @staticmethod
    def _extract_position(value):
        if not isinstance(value, dict):
            return None

        if "position" in value:
            position = value["position"]
            if isinstance(position, dict) and "x" in position and "y" in position:
                return np.asarray([position["x"], position["y"]], dtype=np.float32)
            if isinstance(position, (list, tuple)) and len(position) >= 2:
                return np.asarray(position[:2], dtype=np.float32)

        if "x" in value and "y" in value:
            return np.asarray([value["x"], value["y"]], dtype=np.float32)

        return None


def create_reward_shaped_rllib_env(env_config: Optional[Dict] = None):
    worker_context = env_config
    env_config = dict(env_config or {})
    shaping_config = env_config.pop("reward_shaping", None)
    reward_scale = env_config.pop("reward_scale", 1.0)

    if hasattr(worker_context, "worker_index"):
        env_config["worker_id"] = (
            worker_context.worker_index * env_config.get("num_envs_per_worker", 1)
            + worker_context.vector_index
        )

    soccer_twos = load_soccer_twos()
    env = soccer_twos.make(**env_config)
    env = RewardShapingWrapper(env, shaping_config=shaping_config)
    env.set_reward_scale(reward_scale)

    if env_config.get("multiagent", True) is False:
        return env
    return RLLibWrapper(env)
