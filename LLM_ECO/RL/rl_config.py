#!/usr/bin/env python3
"""
Unified configuration for the RL ECO system.

All RL, reward, environment, and action-space settings now live in this module
so they can be imported directly without a separate YAML file. Optional
overrides can be provided programmatically via `load_config(config_overrides=...)`.
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

# Embedded default configuration (previously in rl_config.yaml)
DEFAULT_CONFIG: Dict[str, Any] = {
    "rl": {
        "action_space_type": "hierarchical",
        "history_window_size": 5,
        # Cap on steps within a single episode per env; the ECO flow supports up to 20
        "max_iterations_per_episode": 10,
        "include_history": True,
        "sparse_rewards": True,
    },
    "reward_weights": {
        "setup_tns": 3.0,
        "hold_tns": 1.0,
        "timing": 1.0,
        "power": 1.0,
        "area": 1.0,
        "time_penalty": 0.1,
        "budget_exceeded": -100.0,
        "invalid_command": -50.0,
        "no_improvement": -10.0,
        "violation_fixed_bonus": 10.0,
        "significant_improvement": 5.0,
    },
    "environment": {
        "base_path": "data/NVDLA_partition_m",
        "design_name": "NV_NVDLA_partition_m",
        "use_real_execution": True,
        # Default to using the persistent pt_shell server/client flow; fallback
        # to per-command pt_shell invocation when disabled.
        "use_pt_server": True,
        "pt_server_host": "127.0.0.1",
        # Base port for env-0; env_id offsets will be added on top.
        "pt_server_base_port": 9009,
        # Port stride between envs to reduce collision chance; auto-search will
        # increment further if a port is occupied.
        "pt_server_port_stride": 10,
        "pt_server_start_timeout_s": 60.0,
        # Allow long-running commands to finish before socket timeouts.
        "pt_server_command_timeout_s": 7200.0,
    },
        "action_spaces": {
        "timing": {
            "violation_type": ["setup", "hold"],
            "actions": ["gate_sizing", "buffer_insertion", "gate_sizing_side_load"],
            "cell_classes": ["combinational", "sequential", "clock_tree"],
            "site_mode": ["open_slot", "occupied_slot"],
            "area_cap": {"low": 0.0, "high": 20.0},
            "slack_above": {"low": 0, "high": 100},
            "slack_below": {"low": -0.300, "high": 0},
        },
        "power": {
            "actions": ["gate_sizing", "buffer_removal"],
            "cell_classes": ["combinational", "sequential"],
            "power_scope": ["total", "dynamic", "leakage"],
            "setup_guard": {"low": -0.10, "high": 0.0},
        },
        "area": {
            "actions": ["gate_sizing", "buffer_removal"],
            "cell_classes": ["combinational", "sequential", "clock_tree"],
            "setup_guard": {"low": -0.10, "high": 0.0},
        },
    },
    "optimization_targets": ["timing", "power", "area"],
    "normalization_ranges": {
        "setup_critical_path_slack": {"min": -10.0, "max": 5.0},
        "setup_total_negative_slack": {"min": -100.0, "max": 0.0},
        "setup_violating_paths": {"min": 0, "max": 1000},
        "hold_critical_path_slack": {"min": -10.0, "max": 5.0},
        "hold_total_negative_slack": {"min": -100.0, "max": 0.0},
        "hold_violating_paths": {"min": 0, "max": 1000},
        "total_power": {"min": 0.0, "max": 1000.0},
        "internal_power": {"min": 0.0, "max": 500.0},
        "switching_power": {"min": 0.0, "max": 300.0},
        "leakage_power": {"min": 0.0, "max": 200.0},
        "total_cell_area": {"min": 0.0, "max": 100000.0},
        "design_area": {"min": 0.0, "max": 150000.0},
        "remaining_budget": {"min": 0.0, "max": 3600.0},
        "elapsed_runtime": {"min": 0.0, "max": 3600.0},
        "iteration": {"min": 0, "max": 20},
    },
}

# Exposed, in-place updated config sections
CONFIG: Dict[str, Any] = {}
RL_CONFIG: Dict[str, Any] = {}
REWARD_WEIGHTS: Dict[str, Any] = {}
ENV_CONFIG: Dict[str, Any] = {}
ACTION_SPACES: Dict[str, Dict[str, Any]] = {}
TIMING_ACTION_SPACE: Dict[str, Any] = {}
POWER_ACTION_SPACE: Dict[str, Any] = {}
AREA_ACTION_SPACE: Dict[str, Any] = {}
OPTIMIZATION_TARGETS: List[str] = []
NORMALIZATION_RANGES: Dict[str, Any] = {}
ACTIVE_CONFIG_PATH: Optional[Path] = None


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge nested dictionaries."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _update_mapping(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    target.clear()
    target.update(source)


def _update_sequence(target: List[Any], source: List[Any]) -> None:
    target.clear()
    target.extend(source)


def _apply_config(config: Dict[str, Any], config_path: Optional[Path]) -> None:
    """Populate module-level config mappings in place."""
    required_sections = [
        "rl",
        "reward_weights",
        "environment",
        "action_spaces",
        "optimization_targets",
        "normalization_ranges",
    ]
    missing = [key for key in required_sections if key not in config]
    if missing:
        raise ValueError(f"Config missing required sections: {', '.join(missing)}")

    _update_mapping(CONFIG, config)
    _update_mapping(RL_CONFIG, config["rl"])
    _update_mapping(REWARD_WEIGHTS, config["reward_weights"])
    _update_mapping(ENV_CONFIG, config["environment"])

    _update_mapping(ACTION_SPACES, config["action_spaces"])
    _update_mapping(TIMING_ACTION_SPACE, ACTION_SPACES.get("timing", {}))
    _update_mapping(POWER_ACTION_SPACE, ACTION_SPACES.get("power", {}))
    _update_mapping(AREA_ACTION_SPACE, ACTION_SPACES.get("area", {}))

    _update_sequence(OPTIMIZATION_TARGETS, config["optimization_targets"])
    _update_mapping(NORMALIZATION_RANGES, config["normalization_ranges"])

    global ACTIVE_CONFIG_PATH
    ACTIVE_CONFIG_PATH = config_path


def load_config(
    config_overrides: Optional[Dict[str, Any]] = None,
    refresh_globals: bool = True
) -> Dict[str, Any]:
    """
    Load configuration from the embedded defaults and optional overrides.

    Args:
        config_overrides: Optional nested dictionary of overrides.
        refresh_globals: Whether to update module-level views.
    """
    config_data = copy.deepcopy(DEFAULT_CONFIG)
    if config_overrides:
        config_data = _deep_merge(config_data, copy.deepcopy(config_overrides))

    if refresh_globals:
        _apply_config(config_data, Path(__file__).resolve())
    return config_data


def get_action_space_config(optimization_target: str) -> Dict[str, Any]:
    """Get action space configuration for a specific optimization target."""
    if optimization_target not in ACTION_SPACES:
        raise ValueError(f"Unknown optimization target: {optimization_target}")
    return ACTION_SPACES[optimization_target]


def validate_config(config_data: Optional[Dict[str, Any]] = None) -> None:
    """Validate configuration consistency."""
    cfg = config_data or CONFIG
    if not cfg:
        raise ValueError("Config not loaded. Call load_config() first.")

    reward_weights = cfg["reward_weights"]
    if "setup_tns" in reward_weights and "hold_tns" in reward_weights:
        objective_weights = (
            reward_weights["setup_tns"] +
            reward_weights["hold_tns"] +
            reward_weights["power"] +
            reward_weights["area"]
        )
    else:
        objective_weights = (
            reward_weights["timing"] +
            reward_weights["power"] +
            reward_weights["area"]
        )
    if objective_weights <= 0:
        raise ValueError("Objective weights must sum to a positive value")

    rl_cfg = cfg["rl"]
    if rl_cfg["action_space_type"] not in ["hierarchical", "flat"]:
        raise ValueError("action_space_type must be 'hierarchical' or 'flat'")
    if rl_cfg["history_window_size"] < 0:
        raise ValueError("history_window_size must be non-negative")


# Load default config on import
load_config()


if __name__ == "__main__":
    validate_config()
    print(f"\n=== RL Configuration ({ACTIVE_CONFIG_PATH}) ===")
    print(f"Action Space Type: {RL_CONFIG.get('action_space_type')}")
    print(f"Max Iterations: {RL_CONFIG.get('max_iterations_per_episode')}")
    print(f"History Window: {RL_CONFIG.get('history_window_size')}")
    print(f"\nReward Weights:")
    print(f"  Setup TNS: {REWARD_WEIGHTS.get('setup_tns')}")
    print(f"  Hold TNS: {REWARD_WEIGHTS.get('hold_tns')}")
    print(f"  Timing (legacy): {REWARD_WEIGHTS.get('timing')}")
    print(f"  Power: {REWARD_WEIGHTS.get('power')}")
    print(f"  Area: {REWARD_WEIGHTS.get('area')}")
