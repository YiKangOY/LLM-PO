#!/usr/bin/env python3
"""
Default configuration for the ECO A3C runner. The structure mirrors the
reference `dse` configs for the knobs that actually drive training (episodes,
steps per episode, optimizer settings) while omitting unused clutter.
"""

import copy
import os
import sys
from typing import Dict, Any

# Ensure sibling Agent package is importable when running from RL/.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)
_AGENT_ROOT = os.path.join(_ROOT, "Agent")
if _AGENT_ROOT not in sys.path:
    sys.path.append(_AGENT_ROOT)

from Agent.report_parsers import parse_power_report_file, parse_qor_report_file, QoRMetrics, PowerMetrics
from rl_config import RL_CONFIG, ENV_CONFIG, load_config
from design_configs import DEFAULT_DESIGN_NAME, get_design_overrides


def load_reports_for_iteration(
    iteration: int,
    last_command_type=None,
    last_reports=None,
    last_executed_command: str = "",
    tool_using: bool = False,
    reports_base_path: str = None
) -> Dict[str, Any]:
    del last_command_type
    del last_reports
    del last_executed_command
    del tool_using
    if reports_base_path is None:
        reports_root = os.path.join(ENV_CONFIG["base_path"], "reports")
    else:
        reports_root = os.path.join(reports_base_path, "reports")

    qor_path = os.path.join(reports_root, "report_qor_{}.txt".format(iteration))
    power_path = os.path.join(reports_root, "report_power_{}.txt".format(iteration))

    qor_parsed = parse_qor_report_file(qor_path)
    power_parsed = parse_power_report_file(power_path)
    if not qor_parsed["parsing_successful"]:
        raise ValueError(
            "Failed to parse QoR report at {}: {}".format(
                qor_path, qor_parsed["parsing_errors"]
            )
        )
    if not power_parsed["parsing_successful"]:
        raise ValueError(
            "Failed to parse power report at {}: {}".format(
                power_path, power_parsed["parsing_errors"]
            )
        )

    qor_metrics = QoRMetrics(**qor_parsed["metrics"])
    power_metrics = PowerMetrics(**power_parsed["metrics"])

    return {
        "qor_metrics": qor_metrics,
        "power_metrics": power_metrics,
        "timing": {
            "setup_critical_path_slack": qor_metrics.setup_critical_path_slack,
            "setup_total_negative_slack": qor_metrics.setup_total_negative_slack,
            "setup_violating_paths": qor_metrics.setup_violating_paths,
            "hold_critical_path_slack": qor_metrics.hold_critical_path_slack,
            "hold_total_negative_slack": qor_metrics.hold_total_negative_slack,
            "hold_violating_paths": qor_metrics.hold_violating_paths,
        },
        "power": {
            "total_power": power_metrics.total_power,
            "total": power_metrics.total_power,
            "internal_power": power_metrics.internal_power,
            "switching_power": power_metrics.switching_power,
            "leakage_power": power_metrics.leakage_power,
        },
        "area": {
            "design_area": qor_metrics.design_area,
            "total_cell_area": qor_metrics.total_cell_area,
        },
    }


def default_eco_a3c_config(
    model_dir: str = None,
    log_dir: str = None,
    design_name: str = None
) -> Dict[str, Any]:
    design_name = design_name or DEFAULT_DESIGN_NAME
    reward_weights = {
        "setup_tns": 10.0,
        "hold_tns": 1.0,
        "power": 3.0,
        "area": 3.0,
    }
    config_overrides = copy.deepcopy(get_design_overrides(design_name))
    config_overrides["reward_weights"] = reward_weights
    load_config(config_overrides=config_overrides)
    base_path = ENV_CONFIG["base_path"]

    model_dir = model_dir or os.path.join(base_path, "RL", "models")
    log_dir = log_dir or os.path.join(base_path, "RL", "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    return {
        "algo": {
            "algo": "a3c",
            "mode": "train",
            "design": "eco",
            "use-cuda": False,
            "gpu-id": 0,
            "random-seed": 0,
            "num-parallel": 1,
            # Reference framework trains for a fixed episode count rather than
            # total timesteps; mirror that here.
            "max-episode": 100,
            "train": {
                "sample-size": 1,
                # Run up to the environment's per-episode iteration cap (20 by default).
                "num-step": RL_CONFIG["max_iterations_per_episode"],
                "gamma": 0.99,
                "lambda": 0.95,
                "beta": 0.5,
                "alpha": 0.01,
                "episode-when-apply-envelope-operator": 5,
                "clip-grad-norm": 0.5,
                "learning-rate": 1e-3,
                "update-critic-episode": 10,
                "save-interval": 5,
                "temperature": 1.0,
                "eval-frequency": 0,
                "log-runtime-breakdown": False,
                "ppo-clip": 0.2,
                "ppo-epoch": 1,
                "resume": False,
                "resume-episode": 0,
                "resume-path": "",
            },
            "test": {
                "ppa-preference": [1.0],
                "max-search-round": 1,
                "rl-model": ENV_CONFIG["model_path"],
            },
        },
        "env": {
            "sim": {"idx": 0},
        },
        "design_name": design_name,
        "use_simulation": False,
        "reports_loader": load_reports_for_iteration,
        "run_dir_override": "",
        "model-path": model_dir,
        "log-path": os.path.join(log_dir, "eco_a3c.log"),
        "reward_weights": reward_weights,
    }
