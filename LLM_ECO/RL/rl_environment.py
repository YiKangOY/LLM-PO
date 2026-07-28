#!/usr/bin/env python3
"""
Gymnasium environment for RL ECO System
Provides standard RL interface for ECO optimization
"""

import time
import copy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os
import inspect
import shutil
from typing import Dict, Any, Optional, Tuple, List

# Add parent and Agent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../Agent'))

from Agent.utils import ECOType
from Agent.configs import design_runtime_budget
from Agent.report_parsers import QoRMetrics, PowerMetrics, parse_power_report_file, parse_qor_report_file

from state_encoder import StateEncoder
from action_decoder import ActionDecoder
from reward_calculator import RewardCalculator
from rl_command_executor import execute_tcl_command
from rl_config import (
    RL_CONFIG,
    REWARD_WEIGHTS,
    ENV_CONFIG,
    OPTIMIZATION_TARGETS,
    TIMING_ACTION_SPACE,
    POWER_ACTION_SPACE,
    AREA_ACTION_SPACE,
    load_config,
)
from pt_server_manager import PTServerManager


REQUIRED_QOR_METRIC_FIELDS = [
    "setup_critical_path_slack",
    "setup_total_negative_slack",
    "setup_violating_paths",
    "setup_levels_of_logic",
    "hold_critical_path_slack",
    "hold_total_negative_slack",
    "hold_violating_paths",
    "hold_levels_of_logic",
    "total_cell_area",
    "design_area",
]

REQUIRED_POWER_METRIC_FIELDS = [
    "total_power",
    "internal_power",
    "switching_power",
    "leakage_power",
]


class EnvTimingWrapper(gym.Wrapper):
    """Wrapper to accumulate per-env timing in the subprocess where step executes."""

    def __init__(self, env, env_id: int, report_every: int = 50, include_reset: bool = True):
        super().__init__(env)
        self.env_id = env_id
        self.report_every = max(1, report_every)
        self.include_reset = include_reset
        self.step_time_total = 0.0
        self.steps = 0
        self.reset_time_total = 0.0

    def reset(self, **kwargs):
        t0 = time.perf_counter() if self.include_reset else None
        obs, info = self.env.reset(**kwargs)
        if t0 is not None:
            dt = time.perf_counter() - t0
            self.reset_time_total += dt
            base_env = self._unwrap_base_env()
            if hasattr(base_env, "reset_time_total"):
                base_env.reset_time_total += dt
        base_env = self._unwrap_base_env()
        if hasattr(base_env, "step_count_total"):
            base_env.step_count_total = 0
        return obs, info

    def step(self, action):
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = self.env.step(action)
        dt = time.perf_counter() - t0
        self.step_time_total += dt
        self.steps += 1
        base_env = self._unwrap_base_env()
        if hasattr(base_env, "step_count_total"):
            base_env.step_count_total += 1

        should_report = (
            self.steps % self.report_every == 0
            or terminated
            or truncated
        )

        if should_report:
            base_env = self._unwrap_base_env()
            tool_time = getattr(base_env, "tool_time_total", 0.0)
            reset_time = getattr(base_env, "reset_time_total", 0.0)
            info = dict(info)
            info["_timing"] = {
                "env_id": self.env_id,
                "step_total_s": self.step_time_total,
                "tool_s": tool_time,
                "reset_s": reset_time,
                "steps": self.steps,
            }
        return obs, reward, terminated, truncated, info

    def get_timing_stats(self) -> Dict[str, Any]:
        base_env = self._unwrap_base_env()
        return {
            "env_id": self.env_id,
            "step_total_s": self.step_time_total,
            "tool_s": getattr(base_env, "tool_time_total", 0.0),
            "reset_s": getattr(base_env, "reset_time_total", 0.0),
            "steps": self.steps,
        }

    def get_runtime_stats(self) -> Dict[str, Any]:
        """Delegate to base env runtime stats for vectorized collection."""
        base_env = self._unwrap_base_env()
        if hasattr(base_env, "get_runtime_stats"):
            stats = base_env.get_runtime_stats()
        else:
            stats = {}
        stats.update({
            "env_id": self.env_id,
            "step_total_s": self.step_time_total,
            "reset_wrapper_s": self.reset_time_total,
            "steps_wrapper": self.steps,
        })
        return stats

    def _unwrap_base_env(self):
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        return env


class ECOEnvironment(gym.Env):
    """
    Gymnasium environment for ECO optimization

    State: Design metrics (QoR, power, area) + history + budget
    Action: Hierarchical (target + specialized) or flat action space
    Reward: Weighted PPA improvements with penalties
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        design_name: str = None,
        reports_loader: Optional[callable] = None,
        use_simulation: Optional[bool] = None,
        render_mode: Optional[str] = None,
        run_paths=None
    ):
        """
        Initialize ECO environment

        Args:
            design_name: Name of design for budget lookup
            reports_loader: Function to load reports for iteration
            use_simulation: If True, simulate command execution (for fast training). Defaults to real pt_shell execution based on ENV_CONFIG["use_real_execution"].
            render_mode: Rendering mode for gym
        """
        super().__init__()
        if not ENV_CONFIG:
            load_config()

        self.design_name = design_name or ENV_CONFIG["design_name"]
        self.reports_loader = reports_loader
        if use_simulation is None:
            use_simulation = not ENV_CONFIG.get("use_real_execution", True)
        self.use_simulation = use_simulation
        self.render_mode = render_mode
        self.run_paths = run_paths
        run_paths_supports_server = (
            getattr(self.run_paths, "use_pt_server", True) if self.run_paths else True
        )
        self.use_pt_server = (
            ENV_CONFIG.get("use_pt_server", True)
            and ENV_CONFIG.get("use_real_execution", True)
            and not self.use_simulation
            and run_paths_supports_server
        )
        self.pt_server: Optional[PTServerManager] = None

        # Get runtime budget for design
        self.total_budget = design_runtime_budget.get(self.design_name, 3600.0)

        # Initialize components
        self.state_encoder = StateEncoder()
        self.action_decoder = ActionDecoder()
        baseline_reports_dir = (
            self.run_paths.reports_dir
            if self.run_paths is not None
            else os.path.join(ENV_CONFIG["base_path"], "reports")
        )
        self.reward_calculator = RewardCalculator(
            baseline_reports_dir=baseline_reports_dir
        )

        # Define observation and action spaces
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()

        # Episode state
        self.current_iteration = 0
        self.history = []
        self.remaining_budget = self.total_budget
        self.elapsed_runtime = 0.0
        self.current_reports = {}
        self.last_command = None
        self.last_command_type = None
        self.consecutive_failures = 0
        self.stagnation_count = 0
        self.last_violation_count = None

        # Episode tracking
        self.episode_rewards = []
        self.episode_violations = []

        # Sparse reward tracking
        self.sparse_rewards = RL_CONFIG["sparse_rewards"]
        self.initial_state = None  # Store initial state for episode-end reward
        self.rollout_step_cap = None
        self.rollout_step_counter = 0
        self.tool_time_total = 0.0
        self.env_runtime_total = 0.0
        self.reset_time_total = 0.0
        self.step_count_total = 0
        self.step_count_total = 0
        self._cached_initial_reports = None
        self._cached_initial_observation = None
        self._cached_initial_info = None
        self._cached_run_dir = None
        self._skip_next_reset_reports = False
        self._last_episode_metrics = None

    def _create_observation_space(self) -> spaces.Box:
        """Create observation space (continuous feature vector)"""
        obs_dim = self.state_encoder.observation_dim
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _create_action_space(self) -> spaces.Box:
        """
        Create action space as a flattened Box.

        The environment uses a continuous Box space, which is decoded into
        structured actions by the action_decoder.

        Action vector structure:
        - [0]: optimization_target (0-1 mapped to discrete 0-2)
        - [1]: violation_type (0-1 mapped to discrete 0-1)
        - [2]: site_mode (0-1 mapped to discrete 0-1)
        - [3:6]: timing_actions (3 binary as 0-1 continuous)
        - [6:9]: timing_cell_classes (3 binary as 0-1 continuous)
        - [9]: area_cap (normalized 0-1)
        - [10:12]: power_actions (2 binary as 0-1 continuous)
        - [12:14]: power_cell_classes (2 binary as 0-1 continuous)
        - [14]: power_scope (0-1 mapped to discrete 0-2)
        - [15:17]: area_actions (2 binary as 0-1 continuous)
        - [17:20]: area_cell_classes (3 binary as 0-1 continuous)
        - [20]: slack_above (normalized 0-1)
        - [21]: slack_below (normalized 0-1)
        - [22]: setup_guard (normalized 0-1; power/area)
        - [23]: noop flag
        """
        # Calculate total action dimension based on actual action space sizes
        total_dim = (
            1 +  # optimization_target
            1 +  # violation_type
            1 +  # site_mode
            len(TIMING_ACTION_SPACE["actions"]) +  # 3
            len(TIMING_ACTION_SPACE["cell_classes"]) +  # 3
            1 +  # area_cap
            len(POWER_ACTION_SPACE["actions"]) +  # 2
            len(POWER_ACTION_SPACE["cell_classes"]) +  # 2
            1 +  # power_scope
            len(AREA_ACTION_SPACE["actions"]) +  # 2
            len(AREA_ACTION_SPACE["cell_classes"]) +  # 3
            1 +  # slack_above
            1 +  # slack_below
            1 +  # setup_guard
            1    # noop
        )

        # All actions are continuous in [0, 1] for simplicity
        # Discrete actions will be decoded by rounding/mapping
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(total_dim,),
            dtype=np.float32
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state

        Returns:
            observation: Initial observation vector
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        reset_t0 = time.perf_counter()

        # Reset episode state
        self.current_iteration = 0
        self.history = []
        self.remaining_budget = self.total_budget
        self.elapsed_runtime = 0.0
        self.last_command = None
        self.last_command_type = None
        self.consecutive_failures = 0
        self.stagnation_count = 0
        self.last_violation_count = None

        # Reset tracking
        self.episode_rewards = []
        self.episode_violations = []
        self.rollout_step_counter = 0
        self.tool_time_total = 0.0
        self.env_runtime_total = 0.0
        self.reset_time_total = 0.0
        self.step_count_total = 0
        self._last_episode_metrics = None

        if self.run_paths is not None:
            self.use_pt_server = (
                ENV_CONFIG.get("use_pt_server", True)
                and ENV_CONFIG.get("use_real_execution", True)
                and not self.use_simulation
                and getattr(self.run_paths, "use_pt_server", True)
            )

        # Defer server startup until the first real command executes, so
        # auto-reset in vectorized envs doesn't spawn an unused pt_shell.

        run_dir = self.run_paths.run_dir if self.run_paths is not None else None
        if self._cached_run_dir != run_dir:
            self._cached_initial_reports = None
            self._cached_initial_observation = None
            self._cached_initial_info = None
            self._cached_run_dir = run_dir

        use_cached = (
            self._skip_next_reset_reports
            and self._cached_initial_reports is not None
        )
        self._skip_next_reset_reports = False

        if use_cached:
            self.current_reports = copy.deepcopy(self._cached_initial_reports)
        else:
            # Load initial reports (iteration 0)
            self.current_reports = self._load_reports(
                iteration=0,
                last_command_type=None,
                last_reports={},
                last_executed_command=""
            )
        # Capture iteration-0 metrics as the normalization baseline
        self.reward_calculator.set_baseline_from_reports(self.current_reports)

        # Store initial state for sparse rewards
        self.initial_state = {
            "reports": self.current_reports.copy(),
            "iteration": 0,
        }

        # Encode initial observation
        if use_cached and self._cached_initial_observation is not None:
            observation = copy.deepcopy(self._cached_initial_observation)
        else:
            observation = self.state_encoder.encode_state(
                reports=self.current_reports,
                history=self.history,
                remaining_budget=self.remaining_budget,
                elapsed_runtime=self.elapsed_runtime,
                iteration=self.current_iteration,
                last_command=None,
                fixing_results=None
            )
            if not use_cached:
                self._cached_initial_reports = copy.deepcopy(self.current_reports)
                self._cached_initial_observation = copy.deepcopy(observation)

        if use_cached and self._cached_initial_info is not None:
            info = dict(self._cached_initial_info)
        else:
            info = {
                "iteration": self.current_iteration,
                "remaining_budget": self.remaining_budget,
                "violations": self._get_total_violations(self.current_reports),
            }
            if not use_cached:
                self._cached_initial_info = dict(info)

        reset_dt = time.perf_counter() - reset_t0
        self.reset_time_total += reset_dt
        self.env_runtime_total += reset_dt
        return observation, info

    def step(
        self,
        action: Dict[str, Any]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step

        Args:
            action: Action dictionary from RL agent

        Returns:
            observation: Next state observation
            reward: Reward for this step
            terminated: Whether episode ended (success or failure)
            truncated: Whether episode was truncated (max iterations)
            info: Additional information
        """
        step_t0 = time.perf_counter()

        # Store previous state for reward calculation
        prev_state = {
            "reports": self.current_reports.copy(),
            "iteration": self.current_iteration,
        }

        # Decode action to TCL command
        action_dict = self._convert_action_format(action)
        tcl_command, action_info = self.action_decoder.decode_action(action_dict)
        noop_action = action_info.get("noop", False)
        if noop_action:
            command_valid = True
        else:
            command_valid = self.action_decoder.validate_command(tcl_command)

        # Execute command (real or simulated)
        if command_valid and not noop_action:
            execution_result = self._execute_command(tcl_command, action_info)
            execution_time = execution_result["execution_time"]
            self.last_command = tcl_command
            self.last_command_type = self._get_command_type(action_info["optimization_target"])
        elif noop_action:
            execution_result = {"execution_time": 0.0, "success": True}
            execution_time = 0.0
            self.last_command = ""
            self.last_command_type = None
        else:
            # Invalid command - apply penalty
            execution_result = {"execution_time": 0.0, "success": False}
            execution_time = 0.0
            self.consecutive_failures += 1

        # Update budget and time
        self.elapsed_runtime += execution_time
        self.remaining_budget = max(0.0, self.total_budget - self.elapsed_runtime)

        # Advance to next iteration
        self.current_iteration += 1

        # Load new reports
        if command_valid and self.reports_loader:
            if noop_action:
                self._copy_iteration_reports(prev_state["iteration"], self.current_iteration)
            self.current_reports = self._load_reports(
                iteration=self.current_iteration,
                last_command_type=self.last_command_type,
                last_reports=prev_state["reports"],
                last_executed_command=self.last_command
            )
        else:
            # Simulate or keep same reports for invalid commands
            self.current_reports = self._simulate_reports(
                prev_state["reports"], action_info
            )

        # Update history
        self.history.append({
            "iteration": self.current_iteration - 1,
            "reports": prev_state["reports"],
            "action": action_info,
            "command": tcl_command,
            "execution_time": execution_time,
        })

        # Keep only recent history
        if len(self.history) > RL_CONFIG["history_window_size"]:
            self.history = self.history[-RL_CONFIG["history_window_size"]:]

        # Calculate reward
        curr_state = {
            "reports": self.current_reports,
            "iteration": self.current_iteration,
        }

        # Check termination conditions first
        terminated, truncated, termination_reason = self._check_termination()

        # Track if this step is the end of the rollout chunk (SB3 cutoff)
        rollout_cutoff = False
        if self.rollout_step_cap:
            self.rollout_step_counter += 1
            if self.rollout_step_counter >= self.rollout_step_cap:
                rollout_cutoff = True
                self.rollout_step_counter = 0

        # Calculate reward based on reward mode
        if self.sparse_rewards:
            # Sparse rewards: give reward at episode end, or at rollout cutoff to avoid zeroing last step
            if terminated or truncated:
                # Episode ended - compute reward from initial to final state
                reward_components = self.reward_calculator.calculate_episode_reward(
                    initial_state=self.initial_state,
                    final_state=curr_state,
                    total_time_used=self.elapsed_runtime,
                    total_budget=self.total_budget
                )
                reward = reward_components["total_reward"]
            elif rollout_cutoff:
                # Rollout chunk ended by SB3; give dense transition reward so last collected step isn't zeroed
                reward_components = self.reward_calculator.calculate_reward(
                    prev_state=prev_state,
                    curr_state=curr_state,
                    action_info=action_info,
                    execution_time=execution_time,
                    budget_remaining=self.remaining_budget,
                    command_valid=command_valid
                )
                reward = reward_components["total_reward"]
            else:
                # Episode ongoing - no reward yet
                reward = 0.0
                reward_components = None
        else:
            # Dense rewards: give incremental reward at each step
            reward_components = self.reward_calculator.calculate_reward(
                prev_state=prev_state,
                curr_state=curr_state,
                action_info=action_info,
                execution_time=execution_time,
                budget_remaining=self.remaining_budget,
                command_valid=command_valid
            )
            reward = reward_components["total_reward"]

        if reward_components is not None:
            print(
                "[DEBUG] Reward computed "
                f"(iteration={self.current_iteration}, total={reward:.6f}, "
                f"keys={list(reward_components.keys())})"
            )
        self.episode_rewards.append(reward)

        # Track violations
        current_violations = self._get_total_violations(self.current_reports)
        self.episode_violations.append(current_violations)

        # Check stagnation
        if self.last_violation_count is not None:
            if current_violations == self.last_violation_count:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0
        self.last_violation_count = current_violations

        # Encode next observation
        fixing_results = execution_result.get("fixing_results", None)
        observation = self.state_encoder.encode_state(
            reports=self.current_reports,
            history=self.history,
            remaining_budget=self.remaining_budget,
            elapsed_runtime=self.elapsed_runtime,
            iteration=self.current_iteration,
            last_command=self.last_command,
            fixing_results=fixing_results
        )

        # Compile info
        info = {
            "iteration": self.current_iteration,
            "remaining_budget": self.remaining_budget,
            "violations": current_violations,
            "reward_components": reward_components,
            "action_info": action_info,
            "tcl_command": tcl_command,
            "command_valid": command_valid,
            "termination_reason": termination_reason if (terminated or truncated) else None,
            "rollout_cutoff": rollout_cutoff,
            "metrics_snapshot": self._build_metrics_snapshot(),
            "episode_metrics": None,
        }

        if terminated or truncated:
            self._last_episode_metrics = self._compute_episode_metrics()
            info["episode_metrics"] = self._last_episode_metrics

        self.env_runtime_total += time.perf_counter() - step_t0
        return observation, reward, terminated, truncated, info

    def set_rollout_step_cap(self, cap: Optional[int]):
        """Configure rollout cutoff (e.g., SB3 n_steps per env) to emit reward on chunk end."""
        if cap is not None and cap <= 0:
            raise ValueError("rollout step cap must be positive or None")
        self.rollout_step_cap = cap
        self.rollout_step_counter = 0

    def _load_reports(
        self,
        iteration: int,
        last_command_type: Optional[ECOType],
        last_reports: Dict[str, Any],
        last_executed_command: str
    ) -> Dict[str, Any]:
        """
        Invoke reports_loader while handling optional reports_base_path for
        compatibility with older loader signatures.
        """
        if not self.reports_loader:
            raise ValueError("reports_loader is required; refusing to create dummy report metrics.")

        print(
            f"[DEBUG] Loading reports for iteration {iteration} "
            f"(run_dir={self.run_paths.run_dir if self.run_paths else 'N/A'})"
        )
        loader_params = inspect.signature(self.reports_loader).parameters
        kwargs = {
            "iteration": iteration,
            "last_command_type": last_command_type,
            "last_reports": last_reports,
            "last_executed_command": last_executed_command,
            "tool_using": False,
        }
        if "reports_base_path" in loader_params:
            # Loader expects a base path that itself contains a reports/ subdir
            kwargs["reports_base_path"] = self.run_paths.run_dir if self.run_paths else None

        reports = self.reports_loader(**kwargs)
        reports = self._attach_structured_metrics(reports, iteration)
        self._validate_reports(reports, iteration)
        print(
            "[DEBUG] Reports loaded "
            f"(iteration={iteration}, keys={list(reports.keys())})"
        )
        return reports

    def _copy_iteration_reports(self, prev_iteration: int, new_iteration: int) -> None:
        """Copy previous iteration reports to the new iteration filenames for no-op steps."""
        if not self.run_paths:
            return
        reports_dir = self.run_paths.reports_dir
        for base in ("report_qor", "report_power"):
            src = os.path.join(reports_dir, f"{base}_{prev_iteration}.txt")
            dst = os.path.join(reports_dir, f"{base}_{new_iteration}.txt")
            if os.path.exists(src):
                shutil.copyfile(src, dst)

    def _attach_structured_metrics(self, reports: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """Attach parsed QoR/Power metrics from report files without accepting defaults."""
        if reports is None:
            raise ValueError(f"Reports loader returned None for iteration {iteration}")

        qor_metrics, power_metrics = self._parse_metrics_from_files(iteration)
        reports["qor_metrics"] = qor_metrics
        reports["power_metrics"] = power_metrics
        return reports

    def _require_parsed_metric_fields(
        self,
        metrics_data: Dict[str, Any],
        required_fields,
        report_type: str,
        iteration: int,
        report_path: str,
    ) -> None:
        """Fail before dataclass construction can hide missing parser fields."""
        missing_fields = sorted(set(required_fields) - set(metrics_data))
        if missing_fields:
            raise KeyError(
                "Parsed {} report for iteration {} missing fields {} at {}".format(
                    report_type,
                    iteration,
                    missing_fields,
                    report_path,
                )
            )
        none_fields = [
            field for field in required_fields
            if metrics_data[field] is None
        ]
        if none_fields:
            raise ValueError(
                "Parsed {} report for iteration {} has None fields {} at {}".format(
                    report_type,
                    iteration,
                    none_fields,
                    report_path,
                )
            )

    def _parse_metrics_from_files(self, iteration: int) -> Tuple[QoRMetrics, PowerMetrics]:
        """Parse QoR and power reports directly to avoid defaulting to zeros."""
        reports_root = (
            self.run_paths.reports_dir
            if self.run_paths
            else os.path.join(ENV_CONFIG["base_path"], "reports")
        )
        qor_path = os.path.join(reports_root, f"report_qor_{iteration}.txt")
        power_path = os.path.join(reports_root, f"report_power_{iteration}.txt")

        qor_parsed = parse_qor_report_file(qor_path)
        if not qor_parsed.get("parsing_successful"):
            raise ValueError(f"Failed to parse QoR report for iteration {iteration} at {qor_path}: {qor_parsed.get('parsing_errors')}")
        qor_metrics_data = qor_parsed.get("metrics")
        if not qor_metrics_data:
            raise ValueError(f"No QoR metrics parsed for iteration {iteration} from {qor_path}")
        self._require_parsed_metric_fields(
            qor_metrics_data,
            REQUIRED_QOR_METRIC_FIELDS,
            "QoR",
            iteration,
            qor_path,
        )
        qor_metrics = QoRMetrics(**qor_metrics_data)

        power_parsed = parse_power_report_file(power_path)
        if not power_parsed.get("parsing_successful"):
            raise ValueError(f"Failed to parse power report for iteration {iteration} at {power_path}: {power_parsed.get('parsing_errors')}")
        power_metrics_data = power_parsed.get("metrics")
        if not power_metrics_data:
            raise ValueError(f"No power metrics parsed for iteration {iteration} from {power_path}")
        self._require_parsed_metric_fields(
            power_metrics_data,
            REQUIRED_POWER_METRIC_FIELDS,
            "power",
            iteration,
            power_path,
        )
        power_metrics = PowerMetrics(**power_metrics_data)

        return qor_metrics, power_metrics

    def _validate_reports(self, reports: Dict[str, Any], iteration: int) -> None:
        """Verify loaded reports contain the fields required by the RL stack."""
        if "timing" not in reports or "power" not in reports:
            raise KeyError(f"Missing timing/power sections in reports for iteration {iteration}: keys={list(reports.keys())}")

        qor_metrics = reports.get("qor_metrics")
        power_metrics = reports.get("power_metrics")
        if qor_metrics is None or power_metrics is None:
            raise KeyError(f"Missing structured metrics in reports for iteration {iteration}")

        missing_qor = [field for field in REQUIRED_QOR_METRIC_FIELDS if getattr(qor_metrics, field, None) is None]
        if missing_qor:
            raise ValueError(f"Missing QoR metrics {missing_qor} for iteration {iteration}")

        missing_power = [field for field in REQUIRED_POWER_METRIC_FIELDS if getattr(power_metrics, field, None) is None]
        if missing_power:
            raise ValueError(f"Missing power metrics {missing_power} for iteration {iteration}")

        timing_section = reports["timing"]
        missing_timing = [field for field in ["setup_violating_paths", "hold_violating_paths"] if field not in timing_section]
        if missing_timing:
            raise KeyError(f"Timing section missing {missing_timing} for iteration {iteration}")

    def _check_termination(self) -> Tuple[bool, bool, Optional[str]]:
        """
        Check if episode should terminate

        Returns:
            (terminated, truncated, reason)
        """
        # Fixed-horizon episodes: only truncate when max iterations are reached.
        if self.current_iteration >= RL_CONFIG["max_iterations_per_episode"]:
            return False, True, "max_iterations_reached"

        return False, False, None

    def _execute_command(
        self,
        tcl_command: str,
        action_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute TCL command (real or simulated)"""
        if self.use_simulation:
            # Simulate execution for fast training
            return self._simulate_execution(tcl_command, action_info)
        else:
            # Real pt_shell execution
            if self.use_pt_server and self.run_paths is not None:
                self._ensure_pt_server()
            command_type = action_info.get("optimization_target", "timing")
            tool_start = time.perf_counter()
            result = execute_tcl_command(
                tcl_command=tcl_command,
                iteration=self.current_iteration,
                command_type=command_type,
                run_paths=self.run_paths,
                pt_server=self.pt_server
            )
            tool_dt = time.perf_counter() - tool_start
            self.tool_time_total += tool_dt
            if result is None:
                # Guard against missing execution result to keep training resilient
                result = {
                    "success": False,
                    "execution_time": tool_dt,
                    "error": "execute_tcl_command returned None"
                }
            if "execution_time" not in result:
                result["execution_time"] = tool_dt
            result["tool_wall_time"] = tool_dt
            # Add fixing_results placeholder if not present
            if 'fixing_results' not in result:
                result['fixing_results'] = None
            return result

    def _ensure_pt_server(self):
        """Start or reuse the persistent pt_shell server for this env."""
        if not self.use_pt_server or self.run_paths is None:
            return
        if self.pt_server is None or getattr(self.pt_server, "run_paths", None) is not self.run_paths:
            self.pt_server = PTServerManager(self.run_paths, enable=self.use_pt_server)
        self.pt_server.ensure_running()

    def _shutdown_pt_server(self):
        """Stop the persistent pt_shell server if it is running."""
        if self.pt_server is not None:
            self.pt_server.shutdown()
            self.pt_server = None

    def _simulate_execution(
        self,
        tcl_command: str,
        action_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate command execution for faster training"""
        # Simulate random execution time (30-180 seconds)
        execution_time = np.random.uniform(30, 180)

        # Simulate fixing results (random for now)
        fixing_results = {
            "fixed_count": int(np.random.uniform(0, 10)),
            "worsened_count": int(np.random.uniform(0, 3)),
            "unfixable_count": int(np.random.uniform(0, 5)),
            "delta_violations": int(np.random.uniform(-10, 2)),
            "delta_slack": np.random.uniform(-1.0, 1.0),
            "delta_power": np.random.uniform(-10.0, 5.0),
        }

        return {
            "execution_time": execution_time,
            "success": True,
            "fixing_results": fixing_results,
        }

    def _simulate_reports(
        self,
        prev_reports: Dict[str, Any],
        action_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate report changes based on action (for invalid commands or testing)"""
        # For simulation, add small random noise to previous reports
        # In real system, this would load actual new reports
        return prev_reports.copy()

    def _get_total_violations(self, reports: Dict[str, Any]) -> int:
        """Extract total violations from reports"""
        timing_reports = reports["timing"]
        return int(timing_reports["setup_violating_paths"] + timing_reports["hold_violating_paths"])

    def _build_metrics_snapshot(self) -> Dict[str, Any]:
        """Collect key timing/power/area metrics used in state and reward calculations."""
        return self._build_metrics_snapshot_from_reports(self.current_reports)

    def _build_metrics_snapshot_from_reports(self, reports: Dict[str, Any]) -> Dict[str, Any]:
        """Collect key timing/power/area metrics from a reports payload."""
        qor = reports["qor_metrics"]
        power = reports["power_metrics"]

        timing_setup = {
            "wns": float(qor.setup_critical_path_slack),
            "tns": float(qor.setup_total_negative_slack),
            "violating_paths": int(qor.setup_violating_paths),
        }
        timing_hold = {
            "wns": float(qor.hold_critical_path_slack),
            "tns": float(qor.hold_total_negative_slack),
            "violating_paths": int(qor.hold_violating_paths),
        }

        return {
            "timing": {
                "setup": timing_setup,
                "hold": timing_hold,
                "drc": {
                    "min_capacitance": int(qor.min_capacitance_count or 0),
                    "max_transition": int(qor.max_transition_count or 0),
                },
            },
            "area": {
                "total_cell_area": float(qor.total_cell_area),
                "design_area": float(qor.design_area),
            },
            "power": {
                "total": float(power.total_power),
                "internal": float(power.internal_power),
                "switching": float(power.switching_power),
                "leakage": float(power.leakage_power),
            },
        }

    def get_episode_metrics(self) -> Dict[str, Any]:
        """Expose initial/final episode metrics for external logging."""
        if self._last_episode_metrics is not None:
            return self._last_episode_metrics
        return self._compute_episode_metrics()

    def _compute_episode_metrics(self) -> Dict[str, Any]:
        """Compute initial/final episode metrics for external logging."""
        initial_qor, initial_power = self._parse_metrics_from_files(0)
        initial_reports = {
            "qor_metrics": initial_qor,
            "power_metrics": initial_power,
        }
        final_iteration = max(
            0,
            min(int(self.current_iteration), int(RL_CONFIG["max_iterations_per_episode"]))
        )
        final_qor, final_power = self._parse_metrics_from_files(final_iteration)
        final_reports = {
            "qor_metrics": final_qor,
            "power_metrics": final_power,
        }
        return {
            "initial_metrics": self._build_metrics_snapshot_from_reports(initial_reports),
            "final_metrics": self._build_metrics_snapshot_from_reports(final_reports),
            "runtime_stats": self.get_runtime_stats(),
        }

    def get_runtime_stats(self) -> Dict[str, Any]:
        """Expose runtime counters for instrumentation and logging."""
        return {
            "tool_time_s": self.tool_time_total,
            "env_runtime_s": self.env_runtime_total,
            "other_runtime_s": max(0.0, self.env_runtime_total - self.tool_time_total),
            "elapsed_runtime_s": self.elapsed_runtime,
            "reset_time_s": self.reset_time_total,
            "steps": self.step_count_total,
            "current_iteration": self.current_iteration,
            "total_budget_s": self.total_budget,
        }

    def _get_command_type(self, optimization_target: str) -> ECOType:
        """Convert optimization target string to ECOType"""
        if optimization_target == "timing":
            return ECOType.TIMING
        elif optimization_target == "power":
            return ECOType.POWER
        elif optimization_target == "area":
            return ECOType.AREA
        else:
            return ECOType.TIMING  # Default

    def _convert_action_format(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Convert flattened Box action array to action decoder format

        Action array structure (24 dims total):
        - [0]: optimization_target (0-1 -> discrete 0-2)
        - [1]: violation_type (0-1 -> discrete 0-1)
        - [2]: site_mode (0-1 -> discrete 0-1)
        - [3:6]: timing_actions (3 binary)
        - [6:9]: timing_cell_classes (3 binary)
        - [9]: area_cap (0-1 normalized; decoded using config low/high)
        - [10:12]: power_actions (2 binary)
        - [12:14]: power_cell_classes (2 binary)
        - [14]: power_scope (0-1 -> discrete 0-2)
        - [15:17]: area_actions (2 binary)
        - [17:20]: area_cell_classes (3 binary)
        - [20]: slack_above (0-1 normalized; positive threshold after decode)
        - [21]: slack_below (0-1 normalized; negative threshold after decode)
        - [22]: setup_guard (0-1 normalized; <=0 after decode, for power/area)
        - [23]: noop flag (>=0.5 => no-op)
        """
        if len(action) > 23 and action[23] > 0.5:
            return {"optimization_target": "noop", "noop": True}

        # Map continuous [0,1] to discrete by rounding
        target_idx = int(round(action[0] * 2))  # 0-2
        target_idx = min(max(target_idx, 0), 2)  # Clamp to valid range

        # Build action dictionary based on target
        action_dict = {"optimization_target": target_idx}

        def _range_from_cfg(cfg):
            if isinstance(cfg, (list, tuple, np.ndarray)):
                values = np.asarray(cfg, dtype=np.float32)
                return float(values.min()), float(values.max())
            return float(cfg["low"]), float(cfg["high"])

        def _decode_normalized(value, cfg):
            low, high = _range_from_cfg(cfg)
            norm = float(value)
            if norm < 0.0:
                norm = 0.0
            elif norm > 1.0:
                norm = 1.0
            span = max(high - low, 1e-6)
            return low + norm * span

        if target_idx == 0:  # Timing
            violation_type = int(round(action[1] * 1))  # 0-1
            action_dict["violation_type"] = min(max(violation_type, 0), 1)
            site_mode = int(round(action[2] * 1))  # 0-1
            action_dict["site_mode"] = min(max(site_mode, 0), 1)
            action_dict["actions"] = action[3:6]  # 3 binary values
            action_dict["cell_classes"] = action[6:9]  # 3 binary values

            action_dict["area_cap"] = _decode_normalized(action[9], TIMING_ACTION_SPACE["area_cap"])

            slack_above = _decode_normalized(action[20], TIMING_ACTION_SPACE["slack_above"])
            if slack_above <= 0.0:
                slack_above = max(slack_above, 0.001)

            slack_below = _decode_normalized(action[21], TIMING_ACTION_SPACE["slack_below"])
            if slack_below >= 0.0:
                slack_below = min(slack_below, -0.001)

            action_dict["slack_above"] = slack_above
            action_dict["slack_below"] = slack_below
        elif target_idx == 1:  # Power
            action_dict["actions"] = action[10:12]  # 2 binary values
            action_dict["cell_classes"] = action[12:14]  # 2 binary values
            power_scope = int(round(action[14] * 2))  # 0-2
            action_dict["power_scope"] = min(max(power_scope, 0), 2)
            setup_guard = _decode_normalized(action[22], POWER_ACTION_SPACE["setup_guard"])
            if setup_guard > 0.0:
                setup_guard = 0.0
            action_dict["setup_guard"] = setup_guard
        else:  # Area (target_idx == 2)
            action_dict["actions"] = action[15:17]  # 2 binary values
            action_dict["cell_classes"] = action[17:20]  # 3 binary values
            setup_guard = _decode_normalized(action[22], AREA_ACTION_SPACE["setup_guard"])
            if setup_guard > 0.0:
                setup_guard = 0.0
            action_dict["setup_guard"] = setup_guard

        return action_dict

    def _create_dummy_reports(self) -> Dict[str, Any]:
        """Create dummy reports for testing without real data"""
        from Agent.report_parsers import QoRMetrics, PowerMetrics

        dummy_qor = QoRMetrics(
            setup_critical_path_slack=-2.0,
            setup_total_negative_slack=-20.0,
            setup_violating_paths=15,
            hold_critical_path_slack=0.5,
            hold_violating_paths=0,
            total_cell_area=50000.0,
            design_area=75000.0,
        )

        dummy_power = PowerMetrics(
            total_power=250.0,
            internal_power=150.0,
            switching_power=70.0,
            leakage_power=30.0,
        )

        return {
            "qor_metrics": dummy_qor,
            "power_metrics": dummy_power,
            "timing": {
                "setup_violating_paths": 15,
                "hold_violating_paths": 0,
                "setup_critical_path_slack": -2.0,
                "hold_critical_path_slack": 0.5,
            },
            "power": {
                "total_power": 250.0,
            },
            "area": {
                "design_area": 75000.0,
            }
        }

    def render(self):
        """Render environment state"""
        if self.render_mode == "human":
            violations = self._get_total_violations(self.current_reports)
            print(f"\n=== Iteration {self.current_iteration} ===")
            print(f"Violations: {violations}")
            print(f"Budget: {self.remaining_budget:.1f}s / {self.total_budget:.1f}s")
            print(f"Last Command: {self.last_command}")

    def close(self):
        """Clean up environment resources"""
        self._shutdown_pt_server()

    def mark_skip_next_reset(self) -> None:
        """Avoid reload of baseline reports on auto-reset after an episode ends."""
        self._skip_next_reset_reports = True


if __name__ == "__main__":
    # Test environment
    print("=== Testing ECO Environment ===")

    env = ECOEnvironment(use_simulation=True, render_mode="human")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test reset
    obs, info = env.reset()
    print(f"\n✓ Initial observation shape: {obs.shape}")
    print(f"✓ Initial info: {info}")

    # Test step with random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"\n✓ Step observation shape: {obs.shape}")
    print(f"✓ Reward: {reward:.3f}")
    print(f"✓ Terminated: {terminated}, Truncated: {truncated}")
    print(f"✓ Info: {info['iteration']}, violations: {info['violations']}")

    env.close()
    print("\n✓ Environment test passed!")
