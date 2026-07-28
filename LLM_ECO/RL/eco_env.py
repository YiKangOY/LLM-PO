#!/usr/bin/env python3
"""
A3C-style ECO environment that mirrors the reference `dse` BasicEnv/Env design.
It wraps the existing `ECOEnvironment` so the reward/state logic stays the same
while exposing a discrete action space for the custom actor-critic agents.
"""

import copy
import os
import sys
import gymnasium as gym
import numpy as np
from typing import Any, Dict, Optional

# Ensure sibling Agent package is importable when executing from RL/.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)
_AGENT_ROOT = os.path.join(_ROOT, "Agent")
if _AGENT_ROOT not in sys.path:
    sys.path.append(_AGENT_ROOT)

from eco_action_space import ECOActionSpace
from rl_command_executor import RunPaths
from rl_environment import ECOEnvironment
from rl_config import ENV_CONFIG, load_config
from design_configs import get_design_overrides


class BasicEnv(gym.Env):
    """
    Minimal BasicEnv analogue. The tunable component count mirrors the allowed
    iterations per episode so the A3C loop can unroll multiple ECO commands in
    one episode.
    """

    def __init__(self, configs: Dict[str, Any], idx: int):
        super().__init__()
        self.configs = configs
        self.mode = configs.get("algo", {}).get("mode", "train")
        self.idx = idx
        design_name = configs["design_name"]
        config_overrides = copy.deepcopy(get_design_overrides(design_name))
        config_overrides["reward_weights"] = configs["reward_weights"]
        load_config(config_overrides=config_overrides)
        self.action_helper = ECOActionSpace()
        self.test_episode_idx = 0
        self.run_dir_override = self.configs["run_dir_override"]
        self.base_path = ENV_CONFIG["base_path"]
        self.run_paths = self._build_run_paths(idx)
        # Prepare per-env workspace: reports/logs/scripts + baseline session.
        self.run_paths.prepare(skip_if_exists=self.mode == "test")
        # Match the RL environment's per-episode iteration cap so the A3C loop
        # can roll multiple steps per episode (up to 20 by default).
        from rl_config import RL_CONFIG  # Local import to avoid circulars
        if not RL_CONFIG:
            load_config()
        self.dims_of_tunable_state = RL_CONFIG.get("max_iterations_per_episode", 1)

    def get_action_candidates(self, state_idx: int):
        # All states share the same action candidates in this abstraction.
        return list(range(self.action_helper.size))

    def _build_run_paths(self, idx: int) -> RunPaths:
        """
        Choose an appropriate run_dir; in test mode we'll rotate per episode.
        """
        if self.run_dir_override:
            run_dir = os.path.join(self.base_path, "RL", "run_dir", self.run_dir_override)
        elif self.mode == "test":
            run_dir = os.path.join(self.base_path, "RL", "run_dir", f"run_test_{self.test_episode_idx}")
        elif self.configs["algo"]["algo"] == "ppo":
            run_dir = os.path.join(self.base_path, "RL", "run_dir", f"run_ppo_{idx}")
        else:
            run_dir = os.path.join(self.base_path, "RL", "run_dir", f"run_{idx}")
        return RunPaths(
            workspace=self.base_path,
            run_dir=run_dir,
            env_id=idx,
        )

    def _rotate_test_run_dir(self):
        """Switch to a fresh run_dir each test episode to isolate artifacts."""
        if self.mode != "test":
            return
        if self.run_dir_override:
            return
        run_dir = os.path.join(self.base_path, "RL", "run_dir", f"run_test_{self.test_episode_idx}")
        self.test_episode_idx += 1
        self.run_paths = RunPaths(
            workspace=self.base_path,
            run_dir=run_dir,
            env_id=self.idx,
        )
        self.run_paths.prepare(skip_if_exists=True)


class ECOEnv(BasicEnv):
    """
    ECO environment compatible with the A3C-style agent. It hides the Gymnasium
    API details and always returns rewards as a vector for consistency with the
    reference implementation.
    """

    def __init__(self, configs: Dict[str, Any], idx: int):
        super().__init__(configs, idx)
        if not ENV_CONFIG:
            load_config()
        design_name = configs.get("design_name", ENV_CONFIG.get("design_name"))
        use_simulation = configs.get("use_simulation")
        if use_simulation is None:
            use_simulation = not ENV_CONFIG.get("use_real_execution", True)
        self.inner_env = ECOEnvironment(
            design_name=design_name,
            reports_loader=configs.get("reports_loader"),
            use_simulation=use_simulation,
            render_mode=None,
            run_paths=configs.get("run_paths", getattr(self, "run_paths", None)),
        )
        self.observation_space = self.inner_env.observation_space
        self.action_space = gym.spaces.Discrete(self.action_helper.size)
        self.reward_space = 1
        self.state = None
        self._pending_auto_reset = False

    @property
    def current_state(self):
        return getattr(self.inner_env, "current_iteration", 0)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        # In test mode, use a fresh run_dir per episode (run_test_{episode_idx}).
        if self.mode == "test" and self._pending_auto_reset:
            # Auto-reset happens inside SubprocVecEnv after a terminal step.
            # Avoid rotating run_dir or reloading reports until the explicit reset.
            self.inner_env.mark_skip_next_reset()
            self._pending_auto_reset = False
        else:
            self._rotate_test_run_dir()
        # Keep the inner environment in sync with the new run_paths.
        self.inner_env.run_paths = self.run_paths
        obs, info = self.inner_env.reset(seed=seed, options=options)
        self.state = obs
        return obs, info

    def step(self, action: int):
        """
        Convert a discrete action index to the flattened 24-d vector expected by
        `ECOEnvironment` and forward the step. Rewards are wrapped into a shape
        (1,) array to match the reference vectorized rollouts.
        """
        action_vec = self.action_helper.idx_to_action(int(action))
        obs, reward, terminated, truncated, info = self.inner_env.step(action_vec)
        reward_vec = np.array([reward], dtype=np.float32)

        if terminated or truncated:
            # SubprocVecEnv expects a terminal observation for proper bootstrapping.
            info = dict(info)
            info["terminal_observation"] = obs
            self.inner_env.mark_skip_next_reset()
            self.inner_env.close()
            if self.mode == "test":
                self._pending_auto_reset = True

        self.state = obs
        return obs, reward_vec, terminated, truncated, info

    def get_episode_metrics(self) -> Dict[str, Any]:
        """Delegate to the inner env so cached episode metrics are returned."""
        return self.inner_env.get_episode_metrics()

    def render(self):
        return None

    def close(self):
        self.inner_env.close()

    def get_episode_metrics(self):
        return self.inner_env.get_episode_metrics()

    def get_runtime_stats(self):
        return self.inner_env.get_runtime_stats()
