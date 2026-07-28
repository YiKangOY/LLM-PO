# Author: baichen318@gmail.com


import importlib.util
import os
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple


ECO_CONFIG: Dict[str, Any] = {
    # Maximum number of ECO steps per episode (mirrors the RL config).
    "max_iterations_per_episode": 15,
    # Flat action space configuration used by the BO wrapper.
    "action_space": {
        # Default bounds for normalized ECO actions.
        "low": 0.0,
        "high": 1.0,
        # Flattened action dimension (kept at 24 to match the RL setup).
        "dimension": 24,
    },
    # Optional structured action definition; kept for compatibility with the RL config
    # layout so users can keep their action/state naming in one place.
    "action_spaces": {},
    # Optional observation normalization ranges (min/max for each metric).
    "normalization_ranges": {},
    "reward_weights": {
        "setup_tns": 3.0,
        "hold_tns": 1.0,
        "power": 1.0,
        "area": 1.0,
    },
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _normalize_overrides(config_overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
        Accept legacy RL-style overrides (keys like `rl` and `action_spaces`)
        and normalize them into the local ECO_CONFIG shape.
    """
    overrides = deepcopy(config_overrides)
    normalized: Dict[str, Any] = {}

    if "rl" in overrides:
        rl_cfg = overrides.pop("rl") or {}
        if "max_iterations_per_episode" in rl_cfg:
            normalized["max_iterations_per_episode"] = rl_cfg["max_iterations_per_episode"]
        if "action_space" in rl_cfg:
            normalized["action_space"] = rl_cfg["action_space"]

    # Preserve structured action space and reward weights if provided.
    if "action_spaces" in overrides:
        normalized["action_spaces"] = overrides.pop("action_spaces")
    if "normalization_ranges" in overrides:
        normalized["normalization_ranges"] = overrides.pop("normalization_ranges")
    if "reward_weights" in overrides:
        normalized["reward_weights"] = overrides.pop("reward_weights")

    # Any remaining keys are copied verbatim.
    for k, v in overrides.items():
        normalized[k] = v
    return normalized


def load_design_config(
    path: Optional[str],
    design_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
        Load a design-specific ECO config from a python file containing
        a `DESIGN_CONFIGS` dict. This mirrors the RL environment layout.
    """
    if not path:
        return None
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"design_configs.py not found at {abs_path}")

    spec = importlib.util.spec_from_file_location("design_configs_module", abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load design configs from {abs_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    design_map = getattr(module, "DESIGN_CONFIGS", None)
    if not isinstance(design_map, dict):
        raise ValueError("DESIGN_CONFIGS must be a dict in design_configs.py")

    name = design_name or next(iter(design_map.keys()))
    if name not in design_map:
        raise KeyError(f"Design '{name}' not found in DESIGN_CONFIGS")
    return deepcopy(design_map[name])


def load_config(
    config_overrides: Optional[Dict[str, Any]] = None,
    design_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
        Build an ECO configuration, applying optional design-specific defaults
        and overrides that may follow the RL config layout.
    """
    cfg = deepcopy(design_config) if design_config else deepcopy(ECO_CONFIG)
    if not config_overrides:
        return cfg
    normalized = _normalize_overrides(config_overrides)
    return _deep_update(cfg, normalized)
