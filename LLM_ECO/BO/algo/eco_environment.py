# Author: baichen318@gmail.com


import importlib
import numpy as np
from typing import Any, Dict, Optional, Tuple

from utils import warn


class BoxActionSpace(object):
    """
        Minimal Box-like action space (gym-lite) used by the ECO wrapper.
    """
    def __init__(self, low: float, high: float, dim: int):
        self.low = np.full(dim, low, dtype=np.float32)
        self.high = np.full(dim, high, dtype=np.float32)
        self.shape = (dim,)

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(np.float32)


def load_custom_env(entrypoint: str, env_kwargs: Optional[Dict[str, Any]], config: Dict[str, Any]):
    """
        Dynamically load a user-specified ECO environment.

        `entrypoint` follows `path.to.module:ClassName`.
        The instantiated class is expected to implement `reset`, `step`, and
        expose `action_space` with `sample()` and `shape`.
    """
    if ":" not in entrypoint:
        raise ValueError("env-entrypoint must be formatted as 'path.to.module:ClassName'")
    module_path, cls_name = entrypoint.split(":")
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    kwargs = env_kwargs.copy() if env_kwargs else {}
    kwargs.setdefault("config", config)
    return cls(**kwargs)


class ECOEnvironment(object):
    def __init__(
        self,
        config: Dict[str, Any],
        use_simulation: bool = True,
        custom_env: Optional[object] = None,
    ):
        self.config = config
        self.use_simulation = use_simulation
        self.custom_env = custom_env
        action_cfg = config.get("action_space", {})
        dim = int(action_cfg.get("dimension", 24))
        low = action_cfg.get("low", 0.0)
        high = action_cfg.get("high", 1.0)
        self.action_space = BoxActionSpace(low=low, high=high, dim=dim)
        self.state_dim = config.get("state_dim", dim)
        self.max_steps = config.get("max_iterations_per_episode", 1)
        self.reward_weights = config.get("reward_weights", {})
        self._step_id = 0

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._step_id = 0
        if self.custom_env:
            return self.custom_env.reset()
        return np.zeros(self.state_dim, dtype=np.float32), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.custom_env:
            obs, reward, terminated, truncated, info = self.custom_env.step(action)
            info = info or {}
            if info.get("reward_components") is None:
                info["reward_components"] = self._build_reward_components(action, reward)
            return obs, reward, terminated, truncated, info
        if not self.use_simulation:
            warn("No custom ECO environment provided; fall back to simulation mode.")
        return self._simulate(action)

    def _simulate(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        act = np.array(action, dtype=np.float32).reshape(-1)
        act = np.clip(act, self.action_space.low, self.action_space.high)
        obs = act
        components = self._build_reward_components(act)
        reward = components["timing_reward"] + components["power_reward"] + components["area_reward"]
        self._step_id += 1
        truncated = self._step_id >= self.max_steps
        info = {"reward_components": components}
        return obs, reward, False, truncated, info

    def _build_reward_components(self, action: np.ndarray, reward: Optional[float] = None) -> Dict[str, float]:
        """
            Derive reward components from the action vector for simulation or as a fallback
            when the custom environment does not expose granular rewards.
        """
        splits = np.array_split(action, 3)
        timing_reward = -float(np.mean(np.square(splits[0]))) if len(splits[0]) else 0.0
        power_reward = -float(np.mean(np.abs(splits[1]))) if len(splits) > 1 else 0.0
        area_reward = -float(np.mean(splits[2])) if len(splits) > 2 else 0.0

        # Apply user-specified weights to keep semantics aligned with RL reward shaping.
        timing_reward *= float(self.reward_weights.get("setup_tns", 1.0))
        power_reward *= float(self.reward_weights.get("power", 1.0))
        area_reward *= float(self.reward_weights.get("area", 1.0))

        if reward is not None:
            # Adjust the area component so that the total reward matches the provided reward.
            total = timing_reward + power_reward + area_reward
            if total != 0:
                scale = reward / total
                timing_reward *= scale
                power_reward *= scale
                area_reward *= scale
        return {
            "timing_reward": timing_reward,
            "power_reward": power_reward,
            "area_reward": area_reward
        }
