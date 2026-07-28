#!/usr/bin/env python3
"""
Reward Calculator for RL ECO System
Computes rewards based on design state transitions
"""

import sys
import os
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Agent.report_parsers import (
    PowerMetrics,
    QoRMetrics,
    parse_power_report_file,
    parse_qor_report_file,
)
from rl_config import REWARD_WEIGHTS, ENV_CONFIG, load_config


REQUIRED_BASELINE_QOR_FIELDS = [
    "setup_total_negative_slack",
    "hold_total_negative_slack",
    "design_area",
]

REQUIRED_BASELINE_POWER_FIELDS = [
    "total_power",
]


class RewardCalculator:
    """
    Calculates rewards for RL agent based on state transitions

    Reward Structure (normalized to iteration-0 baseline):
        R = w_setup * Δsetup_tns + w_hold * Δhold_tns + w_area * Δarea + w_power * Δpower + penalties

    Components:
    - Δsetup_tns / Δhold_tns: Progress of setup/hold TNS toward more positive values
    - Δarea: Normalized area reduction (smaller is better)
    - Δpower: Normalized total power reduction (smaller is better)
    - penalties: Budget/invalid-command penalties
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        baseline_reports_dir: Optional[str] = None,
    ):
        """
        Initialize reward calculator

        Args:
            weights: Custom reward weights (uses REWARD_WEIGHTS if None)
        """
        self.weights = weights or REWARD_WEIGHTS
        self.baseline_reports_dir = baseline_reports_dir
        self.baseline_metrics: Optional[Dict[str, float]] = None

    def set_baseline_from_reports(self, reports: Dict[str, Any]) -> None:
        """Record baseline metrics using an iteration-0 reports dict."""
        qor = reports.get("qor_metrics")
        power = reports.get("power_metrics")
        if qor is None or power is None:
            raise ValueError("Baseline reports must include qor_metrics and power_metrics.")

        self.baseline_metrics = {
            "setup_tns": float(qor.setup_total_negative_slack),
            "hold_tns": float(qor.hold_total_negative_slack),
            "area": float(qor.design_area),
            "power": float(power.total_power),
        }

    def load_baseline_from_files(self, reports_dir: Optional[str] = None) -> None:
        """
        Load baseline (iteration_0) metrics directly from report files.

        This uses the copied baseline reports under the run directory when
        available; otherwise it falls back to the project-level reports folder.
        """
        if not ENV_CONFIG:
            load_config()
        reports_dir = reports_dir or self.baseline_reports_dir or os.path.join(ENV_CONFIG["base_path"], "reports")
        qor_path = os.path.join(reports_dir, "report_qor_0.txt")
        power_path = os.path.join(reports_dir, "report_power_0.txt")

        qor_parsed = parse_qor_report_file(qor_path)
        if not qor_parsed.get("parsing_successful"):
            raise ValueError(f"Failed to parse baseline QoR report at {qor_path}: {qor_parsed.get('parsing_errors')}")
        power_parsed = parse_power_report_file(power_path)
        if not power_parsed.get("parsing_successful"):
            raise ValueError(f"Failed to parse baseline power report at {power_path}: {power_parsed.get('parsing_errors')}")

        qor_metrics = qor_parsed.get("metrics")
        power_metrics = power_parsed.get("metrics")
        if not qor_metrics:
            raise ValueError(f"No baseline QoR metrics parsed from {qor_path}")
        if not power_metrics:
            raise ValueError(f"No baseline power metrics parsed from {power_path}")
        missing_qor = [key for key in REQUIRED_BASELINE_QOR_FIELDS if key not in qor_metrics]
        if missing_qor:
            raise KeyError(f"Baseline QoR metrics missing fields: {missing_qor}")
        missing_power = [key for key in REQUIRED_BASELINE_POWER_FIELDS if key not in power_metrics]
        if missing_power:
            raise KeyError(f"Baseline power metrics missing fields: {missing_power}")
        none_qor = [key for key in REQUIRED_BASELINE_QOR_FIELDS if qor_metrics[key] is None]
        if none_qor:
            raise ValueError(f"Baseline QoR metrics have None fields: {none_qor}")
        none_power = [key for key in REQUIRED_BASELINE_POWER_FIELDS if power_metrics[key] is None]
        if none_power:
            raise ValueError(f"Baseline power metrics have None fields: {none_power}")

        self.baseline_metrics = {
            "setup_tns": float(qor_metrics["setup_total_negative_slack"]),
            "hold_tns": float(qor_metrics["hold_total_negative_slack"]),
            "area": float(qor_metrics["design_area"]),
            "power": float(power_metrics["total_power"]),
        }

    def _ensure_baseline(self, reports: Optional[Dict[str, Any]] = None) -> None:
        """Ensure baseline metrics are available before computing rewards."""
        if self.baseline_metrics is not None:
            return
        if reports:
            self.set_baseline_from_reports(reports)
            return
        self.load_baseline_from_files()

    @staticmethod
    def _normalized_progress(
        prev_val: float,
        curr_val: float,
        baseline_val: float,
        higher_is_better: bool,
        epsilon: float = 1e-9
    ) -> float:
        """
        Compute normalized progress between two values using a baseline.

        Args:
            prev_val: Value before the action
            curr_val: Value after the action
            baseline_val: Iteration-0 baseline value
            higher_is_better: If True, larger values are better; otherwise lower is better
            epsilon: Small constant to avoid division by zero
        """
        denom = abs(baseline_val)
        if denom < epsilon:
            denom = 1.0

        if higher_is_better:
            prev_norm = (prev_val - baseline_val) / denom
            curr_norm = (curr_val - baseline_val) / denom
        else:
            prev_norm = (baseline_val - prev_val) / denom
            curr_norm = (baseline_val - curr_val) / denom

        return curr_norm - prev_norm

    def _budget_penalty(self, budget_remaining: float) -> float:
        """Apply a hard penalty if the runtime budget is exhausted."""
        if budget_remaining <= 0:
            return self.weights.get("budget_exceeded", 0.0)
        return 0.0

    def _get_timing_weights(self) -> Dict[str, float]:
        """
        Resolve timing weights. Prefer explicit setup/hold weights; fall back to
        the legacy unified timing weight when needed.
        """
        setup_weight = self.weights.get("setup_tns")
        hold_weight = self.weights.get("hold_tns")
        if setup_weight is None or hold_weight is None:
            timing_weight = self.weights.get("timing", 1.0)
            setup_weight = timing_weight
            hold_weight = timing_weight
        return {"setup_tns": float(setup_weight), "hold_tns": float(hold_weight)}

    def calculate_reward(
        self,
        prev_state: Dict[str, Any],
        curr_state: Dict[str, Any],
        action_info: Dict[str, Any],
        execution_time: float,
        budget_remaining: float,
        command_valid: bool = True
    ) -> Dict[str, float]:
        """
        Calculate total reward and component breakdowns

        Args:
            prev_state: Previous design state (reports, metrics)
            curr_state: Current design state after command execution
            action_info: Information about executed action
            execution_time: Time taken to execute command (seconds)
            budget_remaining: Remaining runtime budget (seconds)
            command_valid: Whether command was valid and executed successfully

        Returns:
            Dictionary with total reward and component breakdowns
        """
        reward_components = {
            "setup_tns_reward": 0.0,
            "hold_tns_reward": 0.0,
            "timing_reward": 0.0,
            "power_reward": 0.0,
            "area_reward": 0.0,
            "time_penalty": 0.0,
            "hard_penalties": 0.0,
            "bonuses": 0.0,
            "total_reward": 0.0,
        }

        if not command_valid:
            reward_components["hard_penalties"] = self.weights["invalid_command"]
            reward_components["total_reward"] = self.weights["invalid_command"]
            return reward_components

        # Make sure we have baseline numbers from iteration 0
        self._ensure_baseline(prev_state.get("reports"))
        baseline = self.baseline_metrics

        prev_reports = prev_state["reports"]
        curr_reports = curr_state["reports"]
        prev_qor = prev_reports["qor_metrics"]
        curr_qor = curr_reports["qor_metrics"]
        prev_power = prev_reports["power_metrics"]
        curr_power = curr_reports["power_metrics"]

        # Normalized progress for the four primary metrics
        setup_prog = self._normalized_progress(
            prev_qor.setup_total_negative_slack,
            curr_qor.setup_total_negative_slack,
            baseline["setup_tns"],
            higher_is_better=True,
        )
        hold_prog = self._normalized_progress(
            prev_qor.hold_total_negative_slack,
            curr_qor.hold_total_negative_slack,
            baseline["hold_tns"],
            higher_is_better=True,
        )
        area_prog = self._normalized_progress(
            prev_qor.design_area,
            curr_qor.design_area,
            baseline["area"],
            higher_is_better=False,
        )
        power_prog = self._normalized_progress(
            prev_power.total_power,
            curr_power.total_power,
            baseline["power"],
            higher_is_better=False,
        )

        reward_components["setup_tns_reward"] = setup_prog
        reward_components["hold_tns_reward"] = hold_prog
        reward_components["timing_reward"] = setup_prog + hold_prog
        reward_components["area_reward"] = area_prog
        reward_components["power_reward"] = power_prog

        timing_weights = self._get_timing_weights()
        total_reward = (
            timing_weights["setup_tns"] * reward_components["setup_tns_reward"] +
            timing_weights["hold_tns"] * reward_components["hold_tns_reward"] +
            self.weights["power"] * reward_components["power_reward"] +
            self.weights["area"] * reward_components["area_reward"]
        )

        reward_components["total_reward"] = total_reward
        return reward_components

    def calculate_episode_reward(
        self,
        initial_state: Dict[str, Any],
        final_state: Dict[str, Any],
        total_time_used: float,
        total_budget: float
    ) -> Dict[str, float]:
        """
        Calculate sparse reward based only on initial vs final state

        This is called at episode end to compute reward based on:
        - Final timing/power/area vs initial timing/power/area
        - Total time consumed

        Args:
            initial_state: State at episode start (iteration 0)
            final_state: State at episode end
            total_time_used: Total time consumed in episode
            total_budget: Total time budget for episode

        Returns:
            Dictionary with total reward and component breakdowns
        """
        reward_components = {
            "setup_tns_reward": 0.0,
            "hold_tns_reward": 0.0,
            "timing_reward": 0.0,
            "power_reward": 0.0,
            "area_reward": 0.0,
            "time_penalty": 0.0,
            "hard_penalties": 0.0,
            "bonuses": 0.0,
            "total_reward": 0.0,
        }

        self._ensure_baseline(initial_state.get("reports"))
        baseline = self.baseline_metrics

        initial_qor = initial_state["reports"]["qor_metrics"]
        final_qor = final_state["reports"]["qor_metrics"]
        initial_power = initial_state["reports"]["power_metrics"]
        final_power = final_state["reports"]["power_metrics"]

        setup_prog = self._normalized_progress(
            initial_qor.setup_total_negative_slack,
            final_qor.setup_total_negative_slack,
            baseline["setup_tns"],
            higher_is_better=True,
        )
        hold_prog = self._normalized_progress(
            initial_qor.hold_total_negative_slack,
            final_qor.hold_total_negative_slack,
            baseline["hold_tns"],
            higher_is_better=True,
        )
        area_prog = self._normalized_progress(
            initial_qor.design_area,
            final_qor.design_area,
            baseline["area"],
            higher_is_better=False,
        )
        power_prog = self._normalized_progress(
            initial_power.total_power,
            final_power.total_power,
            baseline["power"],
            higher_is_better=False,
        )

        reward_components["setup_tns_reward"] = setup_prog
        reward_components["hold_tns_reward"] = hold_prog
        reward_components["timing_reward"] = setup_prog + hold_prog
        reward_components["area_reward"] = area_prog
        reward_components["power_reward"] = power_prog

        timing_weights = self._get_timing_weights()
        total_reward = (
            timing_weights["setup_tns"] * reward_components["setup_tns_reward"] +
            timing_weights["hold_tns"] * reward_components["hold_tns_reward"] +
            self.weights["power"] * reward_components["power_reward"] +
            self.weights["area"] * reward_components["area_reward"]
        )

        reward_components["total_reward"] = total_reward
        return reward_components

    def get_reward_summary(self, reward_components: Dict[str, float]) -> str:
        """Generate human-readable reward summary"""
        summary_lines = [
            "Reward Breakdown:",
            f"  Setup TNS: {reward_components.get('setup_tns_reward', 0.0):+8.3f}",
            f"  Hold TNS:  {reward_components.get('hold_tns_reward', 0.0):+8.3f}",
            f"  Timing Σ:  {reward_components['timing_reward']:+8.3f}",
            f"  Area:      {reward_components['area_reward']:+8.3f}",
            f"  Power:     {reward_components['power_reward']:+8.3f}",
            f"  Penalties: {reward_components['hard_penalties']:+8.3f}",
            f"  ─────────────────────",
            f"  Total:     {reward_components['total_reward']:+8.3f}",
        ]
        return "\n".join(summary_lines)


if __name__ == "__main__":
    # Test reward calculator
    print("=== Testing Reward Calculator ===")

    from Agent.report_parsers import QoRMetrics, PowerMetrics

    calculator = RewardCalculator()

    # Create test states
    prev_qor = QoRMetrics(
        setup_critical_path_slack=-2.5,
        setup_total_negative_slack=-15.0,
        setup_violating_paths=10,
        hold_critical_path_slack=0.5,
        hold_violating_paths=0,
        total_cell_area=50000.0,
        design_area=75000.0,
    )

    curr_qor = QoRMetrics(
        setup_critical_path_slack=-1.8,  # Improved
        setup_total_negative_slack=-10.0,  # Improved
        setup_violating_paths=7,  # Improved
        hold_critical_path_slack=0.5,
        hold_violating_paths=0,
        total_cell_area=50500.0,  # Slightly worse
        design_area=75500.0,
    )

    prev_power = PowerMetrics(total_power=250.0, leakage_power=30.0)
    curr_power = PowerMetrics(total_power=245.0, leakage_power=28.0)  # Improved

    prev_state = {
        "reports": {"qor_metrics": prev_qor, "power_metrics": prev_power}
    }
    curr_state = {
        "reports": {"qor_metrics": curr_qor, "power_metrics": curr_power}
    }

    # Seed baseline using the initial (iteration 0) reports
    calculator.set_baseline_from_reports(prev_state["reports"])

    # Calculate reward
    action_info = {"optimization_target": "timing"}
    reward_components = calculator.calculate_reward(
        prev_state=prev_state,
        curr_state=curr_state,
        action_info=action_info,
        execution_time=120.0,
        budget_remaining=1800.0,
        command_valid=True
    )

    print("\n" + calculator.get_reward_summary(reward_components))
    print(f"\n✓ Total Reward: {reward_components['total_reward']:.3f}")
