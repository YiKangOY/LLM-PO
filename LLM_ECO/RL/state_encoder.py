#!/usr/bin/env python3
"""
State Encoder for RL ECO System
Converts reports and history into observation vectors for neural network input
"""

import numpy as np
import sys
import os
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Agent.report_parsers import PowerMetrics, QoRMetrics
from rl_config import RL_CONFIG, NORMALIZATION_RANGES, OPTIMIZATION_TARGETS


class StateEncoder:
    """
    Encodes ECO state into observation vectors for RL agent

    Handles:
    - Current design metrics (QoR, power, area)
    - Historical metrics from previous iterations
    - Budget tracking (remaining budget, elapsed time, iteration)
    - Last command information (categorical encoding)
    - Fixing results from previous command
    """

    def __init__(self, history_window: int = None, normalize: bool = True):
        """
        Initialize state encoder

        Args:
            history_window: Number of previous iterations to include
            normalize: Whether to normalize continuous features
        """
        self.history_window = history_window or RL_CONFIG["history_window_size"]
        self.normalize = normalize
        self.normalization_ranges = NORMALIZATION_RANGES.copy()

        # Calculate observation space dimensions
        self.observation_dim = self._calculate_observation_dim()

    def _calculate_observation_dim(self) -> int:
        """Calculate total observation vector dimension"""
        # Current metrics (keep in sync with _encode_current_metrics)
        current_metrics_dim = 15  # QoR (9, minimal) + Power (6)

        # Budget tracking (disabled for now; keep dimension at 0)
        budget_dim = 0

        # Historical metrics (if enabled)
        history_dim = 0
        if RL_CONFIG["include_history"]:
            history_dim = current_metrics_dim * self.history_window

        # Last command categorical features (one-hot encoded)
        last_command_dim = (
            len(OPTIMIZATION_TARGETS) +  # optimization type (3)
            5 +  # method types (simplified aggregation)
            4    # cell types
        )

        # Fixing results from last command
        fixing_results_dim = 6  # fixed, worsened, unfixable, delta_violations, delta_slack, delta_power

        total_dim = (
            current_metrics_dim +
            budget_dim +
            history_dim +
            last_command_dim +
            fixing_results_dim
        )

        return total_dim

    def encode_state(
        self,
        reports: Dict[str, Any],
        history: List[Dict[str, Any]],
        remaining_budget: float,
        elapsed_runtime: float,
        iteration: int,
        last_command: Optional[str] = None,
        fixing_results: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Encode complete state into observation vector

        Args:
            reports: Current iteration reports (QoR, power, area)
            history: List of previous iteration states
            remaining_budget: Remaining runtime budget (seconds)
            elapsed_runtime: Total elapsed runtime (seconds)
            iteration: Current iteration number
            last_command: Last executed TCL command (optional)
            fixing_results: Results from last command execution (optional)

        Returns:
            Encoded observation vector
        """
        features = []

        # 1. Current design metrics
        current_metrics = self._encode_current_metrics(reports)
        features.append(current_metrics)

        # 2. Budget tracking (disabled for now; keep block for quick re-enable)
        # budget_features = self._encode_budget(remaining_budget, elapsed_runtime, iteration)
        # features.append(budget_features)

        # 3. Historical metrics
        if RL_CONFIG["include_history"]:
            history_features = self._encode_history(history)
            features.append(history_features)

        # 4. Last command information
        last_command_features = self._encode_last_command(last_command)
        features.append(last_command_features)

        # 5. Fixing results
        fixing_features = self._encode_fixing_results(fixing_results)
        features.append(fixing_features)

        # Concatenate all features
        observation = np.concatenate(features)

        # Verify dimension
        if observation.shape[0] != self.observation_dim:
            raise ValueError(
                f"Observation dimension mismatch: expected {self.observation_dim}, "
                f"got {observation.shape[0]}"
            )

        return observation.astype(np.float32)

    def _encode_current_metrics(self, reports: Dict[str, Any]) -> np.ndarray:
        """Extract and encode current design metrics from reports"""
        features = []

        # Extract QoR metrics
        qor = reports.get("qor_metrics")
        if qor is None:
            raise ValueError("Missing 'qor_metrics' in reports for state encoding")
        features.extend([
            self._normalize("setup_critical_path_slack", qor.setup_critical_path_slack),
            self._normalize("setup_total_negative_slack", qor.setup_total_negative_slack),
            self._normalize("setup_violating_paths", qor.setup_violating_paths),
            qor.setup_levels_of_logic / 100.0,  # Normalize by typical max

            self._normalize("hold_critical_path_slack", qor.hold_critical_path_slack),
            self._normalize("hold_total_negative_slack", qor.hold_total_negative_slack),
            self._normalize("hold_violating_paths", qor.hold_violating_paths),
            qor.hold_levels_of_logic / 100.0,

            # qor.min_capacitance_count / 100.0,
            # qor.max_transition_count / 100.0,
            # self._normalize("total_cell_area", qor.total_cell_area),
            self._normalize("design_area", qor.design_area),
        ])

        # Extract power metrics
        power = reports.get("power_metrics")
        if power is None:
            raise ValueError("Missing 'power_metrics' in reports for state encoding")
        features.extend([
            self._normalize("total_power", power.total_power),
            self._normalize("internal_power", power.internal_power),
            self._normalize("switching_power", power.switching_power),
            self._normalize("leakage_power", power.leakage_power),
            power.internal_power_pct / 100.0,
            power.leakage_power_pct / 100.0,
        ])

        # Area already included in QoR metrics (disabled for now; keep for re-enable)
        # features.extend([
        #     self._normalize("total_cell_area", qor.total_cell_area),
        #     self._normalize("design_area", qor.design_area),
        # ])

        return np.array(features, dtype=np.float32)

    def _encode_budget(
        self,
        remaining_budget: float,
        elapsed_runtime: float,
        iteration: int
    ) -> np.ndarray:
        """Encode budget tracking information"""
        return np.array([
            self._normalize("remaining_budget", remaining_budget),
            self._normalize("elapsed_runtime", elapsed_runtime),
            self._normalize("iteration", iteration),
        ], dtype=np.float32)

    def _encode_history(self, history: List[Dict[str, Any]]) -> np.ndarray:
        """Encode historical metrics from previous iterations"""
        history_features = []

        # Get last N iterations (or pad with zeros if not enough history)
        recent_history = history[-self.history_window:] if history else []

        for i in range(self.history_window):
            if i < len(recent_history):
                # Extract metrics from historical iteration
                hist_reports = recent_history[i].get("reports", {})
                hist_metrics = self._encode_current_metrics(hist_reports)
            else:
                # Pad with zeros for missing history
                hist_metrics = np.zeros(15, dtype=np.float32)  # Same dim as current metrics

            history_features.append(hist_metrics)

        return np.concatenate(history_features)

    def _encode_last_command(self, last_command: Optional[str]) -> np.ndarray:
        """Encode categorical information about last executed command"""
        features = []

        if not last_command:
            # No previous command - all zeros
            return np.zeros(len(OPTIMIZATION_TARGETS) + 5 + 4, dtype=np.float32)

        # 1. Optimization type (one-hot: timing, power, area)
        opt_type = np.zeros(len(OPTIMIZATION_TARGETS), dtype=np.float32)
        if "opt_timing" in last_command.lower():
            opt_type[0] = 1.0
        elif "opt_power" in last_command.lower():
            opt_type[1] = 1.0
        elif "opt_area" in last_command.lower():
            opt_type[2] = 1.0
        features.append(opt_type)

        # 2. Methods used (simplified binary indicators)
        actions = np.zeros(5, dtype=np.float32)
        if "gate_sizing" in last_command.lower():
            actions[0] = 1.0
        if "swap_pin" in last_command.lower():
            actions[1] = 1.0
        if "buffer" in last_command.lower():
            actions[2] = 1.0
        if "vth" in last_command.lower():
            actions[3] = 1.0
        if "fanout" in last_command.lower():
            actions[4] = 1.0
        features.append(actions)

        # 3. Cell types targeted (binary indicators)
        cell_classes = np.zeros(4, dtype=np.float32)
        if "combinational" in last_command.lower():
            cell_classes[0] = 1.0
        if "sequential" in last_command.lower():
            cell_classes[1] = 1.0
        if "buffer" in last_command.lower():
            cell_classes[2] = 1.0
        if "clock" in last_command.lower():
            cell_classes[3] = 1.0
        features.append(cell_classes)

        return np.concatenate(features)

    def _encode_fixing_results(self, fixing_results: Optional[Dict[str, Any]]) -> np.ndarray:
        """Encode results from last command execution"""
        if not fixing_results:
            return np.zeros(6, dtype=np.float32)

        return np.array([
            fixing_results.get("fixed_count", 0) / 100.0,  # Normalize by typical max
            fixing_results.get("worsened_count", 0) / 100.0,
            fixing_results.get("unfixable_count", 0) / 100.0,
            np.clip(fixing_results.get("delta_violations", 0) / 100.0, -1.0, 1.0),
            np.clip(fixing_results.get("delta_slack", 0) / 10.0, -1.0, 1.0),
            np.clip(fixing_results.get("delta_power", 0) / 100.0, -1.0, 1.0),
        ], dtype=np.float32)

    def _normalize(self, feature_name: str, value: float) -> float:
        """Normalize feature value to [0, 1] or [-1, 1] range"""
        if not self.normalize:
            return value

        if feature_name not in self.normalization_ranges:
            # No normalization range defined - return as is
            return value

        range_info = self.normalization_ranges[feature_name]
        min_val = range_info["min"]
        max_val = range_info["max"]

        if max_val == min_val:
            return 0.0

        # Normalize to [0, 1]
        normalized = (value - min_val) / (max_val - min_val)

        # Clip to valid range
        return np.clip(normalized, 0.0, 1.0)

    def get_observation_space_shape(self) -> Tuple[int]:
        """Return observation space shape for Gym environment"""
        return (self.observation_dim,)

    def update_normalization_ranges(
        self,
        feature_name: str,
        min_val: float,
        max_val: float
    ):
        """Update normalization ranges based on observed data"""
        if feature_name in self.normalization_ranges:
            current = self.normalization_ranges[feature_name]
            self.normalization_ranges[feature_name] = {
                "min": min(current["min"], min_val),
                "max": max(current["max"], max_val),
            }
        else:
            self.normalization_ranges[feature_name] = {
                "min": min_val,
                "max": max_val,
            }


if __name__ == "__main__":
    # Test state encoder
    print("=== Testing State Encoder ===")

    encoder = StateEncoder()
    print(f"Observation dimension: {encoder.observation_dim}")
    print(f"Observation shape: {encoder.get_observation_space_shape()}")

    # Create dummy reports for testing
    from Agent.report_parsers import QoRMetrics, PowerMetrics

    dummy_qor = QoRMetrics(
        setup_critical_path_slack=-2.5,
        setup_total_negative_slack=-15.0,
        setup_violating_paths=10,
        setup_levels_of_logic=25,
        hold_critical_path_slack=0.5,
        hold_total_negative_slack=0.0,
        hold_violating_paths=0,
        total_cell_area=50000.0,
        design_area=75000.0,
    )

    dummy_power = PowerMetrics(
        total_power=250.0,
        internal_power=150.0,
        switching_power=70.0,
        leakage_power=30.0,
        internal_power_pct=60.0,
        leakage_power_pct=12.0,
    )

    dummy_reports = {
        "qor_metrics": dummy_qor,
        "power_metrics": dummy_power,
    }

    # Encode state
    observation = encoder.encode_state(
        reports=dummy_reports,
        history=[],
        remaining_budget=1800.0,
        elapsed_runtime=300.0,
        iteration=5,
        last_command="opt_timing -violation_type setup -actions {gate_sizing swap_pin}",
        fixing_results={"fixed_count": 5, "worsened_count": 1, "delta_violations": -5}
    )

    print(f"✓ Encoded observation shape: {observation.shape}")
    print(f"✓ Observation sample (first 10): {observation[:10]}")
    print(f"✓ Observation range: [{observation.min():.3f}, {observation.max():.3f}]")
