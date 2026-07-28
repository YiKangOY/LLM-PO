#!/usr/bin/env python3
"""
Design-specific overrides for the unified RL configuration.
"""

import copy

from rl_config import DEFAULT_CONFIG


DEFAULT_ACTION_SPACES = copy.deepcopy(DEFAULT_CONFIG["action_spaces"])
DEFAULT_NORMALIZATION_RANGES = copy.deepcopy(DEFAULT_CONFIG["normalization_ranges"])


def _build_normalization_ranges(runtime_budget, max_iterations):
    ranges = copy.deepcopy(DEFAULT_NORMALIZATION_RANGES)
    ranges["remaining_budget"] = {"min": 0.0, "max": runtime_budget}
    ranges["elapsed_runtime"] = {"min": 0.0, "max": runtime_budget}
    ranges["iteration"] = {"min": 0, "max": max_iterations}
    return ranges


DESIGN_CONFIG_OVERRIDES = {

    "NV_NVDLA_partition_m": {
        "environment": {
            "base_path": "data/NVDLA_partition_m",
            "design_name": "NV_NVDLA_partition_m",
            "model_path": "data/NVDLA_partition_m/RL/models/eco_a3c.pt",
        },
        "rl": {"max_iterations_per_episode": 10},
        "action_spaces": {
            # unit in pico seconds
            "timing": {
                "violation_type": ["setup", "hold"],
                "actions": ["gate_sizing", "buffer_insertion", "gate_sizing_side_load"],
                "cell_classes": ["combinational", "sequential", "clock_tree"],
                "site_mode": ["open_slot", "occupied_slot"],
                "area_cap": {"low": 2, "high": 20.0},
                "slack_above": {"low": 0, "high": 1000},
                "slack_below": {"low": -1000, "high": 0},
            },
            "power": {
                "actions": ["gate_sizing", "buffer_removal"],
                "cell_classes": ["combinational", "sequential"],
                "power_scope": ["total", "dynamic", "leakage"],
                "setup_guard": {"low": -1000, "high": 0.0},
            },
            "area": {
                "actions": ["gate_sizing", "buffer_removal"],
                "cell_classes": ["combinational", "sequential", "clock_tree"],
                "setup_guard": {"low": -1000, "high": 0.0},
            },
        },
        "normalization_ranges": {
            "setup_critical_path_slack": {"min": -1000, "max": 1000},
            "setup_total_negative_slack": {"min": -25000, "max": 0.0},
            "setup_violating_paths": {"min": 0, "max": 1000},
            "hold_critical_path_slack": {"min": -1000, "max": 1000},
            "hold_total_negative_slack": {"min": -25000, "max": 0.0},
            "hold_violating_paths": {"min": 0, "max": 1000},
            "total_power": {"min": 0.0, "max": 2e-2},
            "internal_power": {"min": 0.0, "max": 2e-2},
            "switching_power": {"min": 0.0, "max": 2e-2},
            "leakage_power": {"min": 0.0, "max": 1e-5},
            "total_cell_area": {"min": 0.0, "max": 5000.0},
            "design_area": {"min": 0.0, "max": 5000.0},
            "remaining_budget": {"min": 0.0, "max": 9999999999.0},
            "elapsed_runtime": {"min": 0.0, "max": 99999999999.0},
            "iteration": {"min": 0, "max": 10},
        },
    },
    "NV_NVDLA_partition_p": {
        "environment": {
            "base_path": "data/NVDLA_partition_p",
            "design_name": "NV_NVDLA_partition_p",
            "model_path": "data/NVDLA_partition_p/RL/models/eco_a3c.pt",
        },
        "rl": {"max_iterations_per_episode": 10},
        "action_spaces": {
            # unit in pico seconds
            "timing": {
                "violation_type": ["setup", "hold"],
                "actions": ["gate_sizing", "buffer_insertion", "gate_sizing_side_load"],
                "cell_classes": ["combinational", "sequential", "clock_tree"],
                "site_mode": ["open_slot", "occupied_slot"],
                "area_cap": {"low": 2, "high": 20.0},
                "slack_above": {"low": 0, "high": 1000},
                "slack_below": {"low": -1000, "high": 0},
            },
            "power": {
                "actions": ["gate_sizing", "buffer_removal"],
                "cell_classes": ["combinational", "sequential"],
                "power_scope": ["total", "dynamic", "leakage"],
                "setup_guard": {"low": -1000, "high": 0.0},
            },
            "area": {
                "actions": ["gate_sizing", "buffer_removal"],
                "cell_classes": ["combinational", "sequential", "clock_tree"],
                "setup_guard": {"low": -1000, "high": 0.0},
            },
        },
        "normalization_ranges": {
            "setup_critical_path_slack": {"min": -1000, "max": 1000},
            "setup_total_negative_slack": {"min": -250000, "max": 0.0},
            "setup_violating_paths": {"min": 0, "max": 1000},
            "hold_critical_path_slack": {"min": -1000, "max": 1000},
            "hold_total_negative_slack": {"min": -250000, "max": 0.0},
            "hold_violating_paths": {"min": 0, "max": 1000},
            "total_power": {"min": 0.0, "max": 2e-2},
            "internal_power": {"min": 0.0, "max": 2e-2},
            "switching_power": {"min": 0.0, "max": 2e-2},
            "leakage_power": {"min": 0.0, "max": 1e-5},
            "total_cell_area": {"min": 0.0, "max": 12000.0},
            "design_area": {"min": 0.0, "max": 12000.0},
            "remaining_budget": {"min": 0.0, "max": 9999999999.0},
            "elapsed_runtime": {"min": 0.0, "max": 99999999999.0},
            "iteration": {"min": 0, "max": 10},
        },
    },
    "ariane136": {
        "environment": {
            "base_path": "data/ariane136",
            "design_name": "ariane136",
            "model_path": "data/ariane136/RL/models/eco_a3c.pt",
        },
        "rl": {"max_iterations_per_episode": 10},
        "action_spaces": {
            # unit in pico seconds
            "timing": {
                "violation_type": ["setup", "hold"],
                "actions": ["gate_sizing", "buffer_insertion", "gate_sizing_side_load"],
                "cell_classes": ["combinational", "sequential", "clock_tree"],
                "site_mode": ["open_slot", "occupied_slot"],
                "area_cap": {"low": 2, "high": 20.0},
                "slack_above": {"low": 0, "high": 1000},
                "slack_below": {"low": -1000, "high": 0},
            },
            "power": {
                "actions": ["gate_sizing", "buffer_removal"],
                "cell_classes": ["combinational", "sequential"],
                "power_scope": ["total", "dynamic", "leakage"],
                "setup_guard": {"low": -1000, "high": 0.0},
            },
            "area": {
                "actions": ["gate_sizing", "buffer_removal"],
                "cell_classes": ["combinational", "sequential", "clock_tree"],
                "setup_guard": {"low": -1000, "high": 0.0},
            },
        },
        "normalization_ranges": {
            "setup_critical_path_slack": {"min": -1000, "max": 1000},
            "setup_total_negative_slack": {"min": -260000, "max": 0.0},
            "setup_violating_paths": {"min": 0, "max": 1000},
            "hold_critical_path_slack": {"min": -1000, "max": 1000},
            "hold_total_negative_slack": {"min": -260000, "max": 0.0},
            "hold_violating_paths": {"min": 0, "max": 1000},
            "total_power": {"min": 0.0, "max": 1},
            "internal_power": {"min": 0.0, "max": 1},
            "switching_power": {"min": 0.0, "max": 1},
            "leakage_power": {"min": 0.0, "max": 0.03},
            "total_cell_area": {"min": 0.0, "max": 50000.0},
            "design_area": {"min": 0.0, "max": 50000.0},
            "remaining_budget": {"min": 0.0, "max": 9999999999.0},
            "elapsed_runtime": {"min": 0.0, "max": 99999999999.0},
            "iteration": {"min": 0, "max": 10},
        },
    },
    "mempool_tile_wrap": {
        "environment": {
            "base_path": "data/mempool_tile_wrap",
            "design_name": "mempool_tile_wrap",
            "model_path": "data/mempool_tile_wrap/RL/models/eco_a3c.pt",
        },
        "rl": {"max_iterations_per_episode": 10},
        "action_spaces": {
            # unit in pico seconds
            "timing": {
                "violation_type": ["setup", "hold"],
                "actions": ["gate_sizing", "buffer_insertion", "gate_sizing_side_load"],
                "cell_classes": ["combinational", "sequential", "clock_tree"],
                "site_mode": ["open_slot", "occupied_slot"],
                "area_cap": {"low": 2, "high": 20.0},
                "slack_above": {"low": 0, "high": 1000},
                "slack_below": {"low": -1000, "high": 0},
            },
            "power": {
                "actions": ["gate_sizing", "buffer_removal"],
                "cell_classes": ["combinational", "sequential"],
                "power_scope": ["total", "dynamic", "leakage"],
                "setup_guard": {"low": -1000, "high": 0.0},
            },
            "area": {
                "actions": ["gate_sizing", "buffer_removal"],
                "cell_classes": ["combinational", "sequential", "clock_tree"],
                "setup_guard": {"low": -1000, "high": 0.0},
            },
        },
        "normalization_ranges": {
            "setup_critical_path_slack": {"min": -1000, "max": 1000},
            "setup_total_negative_slack": {"min": -2600000, "max": 0.0},
            "setup_violating_paths": {"min": 0, "max": 1000},
            "hold_critical_path_slack": {"min": -1000, "max": 1000},
            "hold_total_negative_slack": {"min": -2600000, "max": 0.0},
            "hold_violating_paths": {"min": 0, "max": 1000},
            "total_power": {"min": 0.0, "max": 0.1},
            "internal_power": {"min": 0.0, "max": 0.1},
            "switching_power": {"min": 0.0, "max": 0.1},
            "leakage_power": {"min": 0.0, "max": 3e-3},
            "total_cell_area": {"min": 0.0, "max": 50000.0},
            "design_area": {"min": 0.0, "max": 50000.0},
            "remaining_budget": {"min": 0.0, "max": 9999999999.0},
            "elapsed_runtime": {"min": 0.0, "max": 99999999999.0},
            "iteration": {"min": 0, "max": 10},
        },
    },
    "aes_256": {
        "environment": {
            "base_path": "data/aes_256",
            "design_name": "aes_256",
            "model_path": "data/aes_256/RL/models/eco_a3c.pt",
        },
        "rl": {"max_iterations_per_episode": 10},
        "action_spaces": {
            # unit in pico seconds
            "timing": {
                "violation_type": ["setup", "hold"],
                "actions": ["gate_sizing", "buffer_insertion", "gate_sizing_side_load"],
                "cell_classes": ["combinational", "sequential", "clock_tree"],
                "site_mode": ["open_slot", "occupied_slot"],
                "area_cap": {"low": 2, "high": 20.0},
                "slack_above": {"low": 0, "high": 100},
                "slack_below": {"low": -100, "high": 0},
            },
            "power": {
                "actions": ["gate_sizing", "buffer_removal"],
                "cell_classes": ["combinational", "sequential"],
                "power_scope": ["total", "dynamic", "leakage"],
                "setup_guard": {"low": -100, "high": 0.0},
            },
            "area": {
                "actions": ["gate_sizing", "buffer_removal"],
                "cell_classes": ["combinational", "sequential", "clock_tree"],
                "setup_guard": {"low": -100, "high": 0.0},
            },
        },
        "normalization_ranges": {
            "setup_critical_path_slack": {"min": -100, "max": 100},
            "setup_total_negative_slack": {"min": -100000, "max": 0.0},
            "setup_violating_paths": {"min": 0, "max": 10000},
            "hold_critical_path_slack": {"min": -1000, "max": 1000},
            "hold_total_negative_slack": {"min": -100000, "max": 0.0},
            "hold_violating_paths": {"min": 0, "max": 10000},
            "total_power": {"min": 0.0, "max": 1},
            "internal_power": {"min": 0.0, "max": 1},
            "switching_power": {"min": 0.0, "max": 1},
            "leakage_power": {"min": 0.0, "max": 3e-5},
            "total_cell_area": {"min": 0.0, "max": 40000.0},
            "design_area": {"min": 0.0, "max": 40000.0},
            "remaining_budget": {"min": 0.0, "max": 9999999999.0},
            "elapsed_runtime": {"min": 0.0, "max": 99999999999.0},
            "iteration": {"min": 0, "max": 10},
        },
    },
    "hidden1": {
        "environment": {
            "base_path": "data/hidden1",
            "design_name": "hidden1",
            "model_path": "data/hidden1/RL/models/eco_a3c.pt",
        },
        "rl": {"max_iterations_per_episode": 10},
        "action_spaces": {
            # unit in pico seconds
            "timing": {
                "violation_type": ["setup", "hold"],
                "actions": ["gate_sizing", "buffer_insertion", "gate_sizing_side_load"],
                "cell_classes": ["combinational", "sequential", "clock_tree"],
                "site_mode": ["open_slot", "occupied_slot"],
                "area_cap": {"low": 2, "high": 20.0},
                "slack_above": {"low": 0, "high": 100},
                "slack_below": {"low": -100, "high": 0},
            },
            "power": {
                "actions": ["gate_sizing", "buffer_removal"],
                "cell_classes": ["combinational", "sequential"],
                "power_scope": ["total", "dynamic", "leakage"],
                "setup_guard": {"low": -100, "high": 0.0},
            },
            "area": {
                "actions": ["gate_sizing", "buffer_removal"],
                "cell_classes": ["combinational", "sequential", "clock_tree"],
                "setup_guard": {"low": -100, "high": 0.0},
            },
        },
        "normalization_ranges": {
            "setup_critical_path_slack": {"min": -200, "max": 200},
            "setup_total_negative_slack": {"min": -50000, "max": 0.0},
            "setup_violating_paths": {"min": 0, "max": 1000},
            "hold_critical_path_slack": {"min": -200, "max": 200},
            "hold_total_negative_slack": {"min": -50000, "max": 0.0},
            "hold_violating_paths": {"min": 0, "max": 1000},
            "total_power": {"min": 0.0, "max": 0.1},
            "internal_power": {"min": 0.0, "max": 0.1},
            "switching_power": {"min": 0.0, "max": 0.1},
            "leakage_power": {"min": 0.0, "max": 1e-5},
            "total_cell_area": {"min": 0.0, "max": 10000.0},
            "design_area": {"min": 0.0, "max": 10000.0},
            "remaining_budget": {"min": 0.0, "max": 9999999999.0},
            "elapsed_runtime": {"min": 0.0, "max": 99999999999.0},
            "iteration": {"min": 0, "max": 10},
        },
    },
    "hidden2": {
        "environment": {
            "base_path": "data/hidden2",
            "design_name": "hidden2",
            "model_path": "data/hidden2/RL/models/eco_a3c.pt",
        },
        "rl": {"max_iterations_per_episode": 10},
        "action_spaces": {
            # unit in pico seconds
            "timing": {
                "violation_type": ["setup", "hold"],
                "actions": ["gate_sizing", "buffer_insertion", "gate_sizing_side_load"],
                "cell_classes": ["combinational", "sequential", "clock_tree"],
                "site_mode": ["open_slot", "occupied_slot"],
                "area_cap": {"low": 2, "high": 20.0},
                "slack_above": {"low": 0, "high": 200},
                "slack_below": {"low": -200, "high": 0},
            },
            "power": {
                "actions": ["gate_sizing", "buffer_removal"],
                "cell_classes": ["combinational", "sequential"],
                "power_scope": ["total", "dynamic", "leakage"],
                "setup_guard": {"low": -200, "high": 0.0},
            },
            "area": {
                "actions": ["gate_sizing", "buffer_removal"],
                "cell_classes": ["combinational", "sequential", "clock_tree"],
                "setup_guard": {"low": -200, "high": 0.0},
            },
        },
        "normalization_ranges": {
            "setup_critical_path_slack": {"min": -200, "max": 200},
            "setup_total_negative_slack": {"min": -200000, "max": 0.0},
            "setup_violating_paths": {"min": 0, "max": 3000},
            "hold_critical_path_slack": {"min": -200, "max": 200},
            "hold_total_negative_slack": {"min": -200000, "max": 0.0},
            "hold_violating_paths": {"min": 0, "max": 1000},
            "total_power": {"min": 0.0, "max": 1},
            "internal_power": {"min": 0.0, "max": 1},
            "switching_power": {"min": 0.0, "max": 1},
            "leakage_power": {"min": 0.0, "max": 0.02},
            "total_cell_area": {"min": 0.0, "max": 40000.0},
            "design_area": {"min": 0.0, "max": 40000.0},
            "remaining_budget": {"min": 0.0, "max": 9999999999.0},
            "elapsed_runtime": {"min": 0.0, "max": 99999999999.0},
            "iteration": {"min": 0, "max": 10},
        },
    },
    "hidden3": {
        "environment": {
            "base_path": "data/hidden3",
            "design_name": "hidden3",
            "model_path": "data/hidden3/RL/models/eco_a3c.pt",
        },
        "rl": {"max_iterations_per_episode": 10},
        "action_spaces": {
            # unit in pico seconds
            "timing": {
                "violation_type": ["setup", "hold"],
                "actions": ["gate_sizing", "buffer_insertion", "gate_sizing_side_load"],
                "cell_classes": ["combinational", "sequential", "clock_tree"],
                "site_mode": ["open_slot", "occupied_slot"],
                "area_cap": {"low": 2, "high": 20.0},
                "slack_above": {"low": 0, "high": 200},
                "slack_below": {"low": -200, "high": 0},
            },
            "power": {
                "actions": ["gate_sizing", "buffer_removal"],
                "cell_classes": ["combinational", "sequential"],
                "power_scope": ["total", "dynamic", "leakage"],
                "setup_guard": {"low": -200, "high": 0.0},
            },
            "area": {
                "actions": ["gate_sizing", "buffer_removal"],
                "cell_classes": ["combinational", "sequential", "clock_tree"],
                "setup_guard": {"low": -200, "high": 0.0},
            },
        },
        "normalization_ranges": {
            "setup_critical_path_slack": {"min": -200, "max": 200},
            "setup_total_negative_slack": {"min": -20000, "max": 0.0},
            "setup_violating_paths": {"min": 0, "max": 1000},
            "hold_critical_path_slack": {"min": -200, "max": 200},
            "hold_total_negative_slack": {"min": -20000, "max": 0.0},
            "hold_violating_paths": {"min": 0, "max": 1000},
            "total_power": {"min": 0.0, "max": 1},
            "internal_power": {"min": 0.0, "max": 1},
            "switching_power": {"min": 0.0, "max": 1},
            "leakage_power": {"min": 0.0, "max": 0.1},
            "total_cell_area": {"min": 0.0, "max": 100000.0},
            "design_area": {"min": 0.0, "max": 1000000.0},
            "remaining_budget": {"min": 0.0, "max": 9999999999.0},
            "elapsed_runtime": {"min": 0.0, "max": 99999999999.0},
            "iteration": {"min": 0, "max": 10},
        },
    },
    "hidden5": {
        "environment": {
            "base_path": "data/hidden5",
            "design_name": "hidden5",
            "model_path": "data/hidden5/RL/models/eco_a3c.pt",
        },
        "rl": {"max_iterations_per_episode": 10},
        "action_spaces": {
            # unit in pico seconds
            "timing": {
                "violation_type": ["setup", "hold"],
                "actions": ["gate_sizing", "buffer_insertion", "gate_sizing_side_load"],
                "cell_classes": ["combinational", "sequential", "clock_tree"],
                "site_mode": ["open_slot", "occupied_slot"],
                "area_cap": {"low": 2, "high": 20.0},
                "slack_above": {"low": 0, "high": 200},
                "slack_below": {"low": -200, "high": 0},
            },
            "power": {
                "actions": ["gate_sizing", "buffer_removal"],
                "cell_classes": ["combinational", "sequential"],
                "power_scope": ["total", "dynamic", "leakage"],
                "setup_guard": {"low": -200, "high": 0.0},
            },
            "area": {
                "actions": ["gate_sizing", "buffer_removal"],
                "cell_classes": ["combinational", "sequential", "clock_tree"],
                "setup_guard": {"low": -200, "high": 0.0},
            },
        },
        "normalization_ranges": {
            "setup_critical_path_slack": {"min": -200, "max": 200},
            "setup_total_negative_slack": {"min": -200000, "max": 0.0},
            "setup_violating_paths": {"min": 0, "max": 3000},
            "hold_critical_path_slack": {"min": -200, "max": 200},
            "hold_total_negative_slack": {"min": -200000, "max": 0.0},
            "hold_violating_paths": {"min": 0, "max": 3000},
            "total_power": {"min": 0.0, "max": 1},
            "internal_power": {"min": 0.0, "max": 1},
            "switching_power": {"min": 0.0, "max": 1},
            "leakage_power": {"min": 0.0, "max": 1e-4},
            "total_cell_area": {"min": 0.0, "max": 40000.0},
            "design_area": {"min": 0.0, "max": 300000.0},
            "remaining_budget": {"min": 0.0, "max": 9999999999.0},
            "elapsed_runtime": {"min": 0.0, "max": 99999999999.0},
            "iteration": {"min": 0, "max": 10},
        },
    },

}

DEFAULT_DESIGN_NAME = "ECO_Vex"


def get_design_overrides(design_name):
    return DESIGN_CONFIG_OVERRIDES[design_name]


def list_design_names():
    return list(DESIGN_CONFIG_OVERRIDES)
