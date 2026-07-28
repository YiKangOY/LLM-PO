#!/usr/bin/env python3
"""
Discrete ECO action space utilities mirroring the design-space helpers in the
reference A3C framework. The original environment consumed a flattened Box
action; here we enumerate all reasonable combinations so an A3C-style discrete
policy can pick a single index and the wrapper will translate it back to the
expected 24-d vector.
"""

import itertools
import numpy as np
from typing import List

from rl_config import TIMING_ACTION_SPACE, POWER_ACTION_SPACE, AREA_ACTION_SPACE, load_config


class ECOActionSpace:
    """
    Enumerate the discrete ECO actions to mirror the reference design-space
    approach. Each entry is a 24-d vector aligned with `_convert_action_format`
    in `ECOEnvironment`, so downstream decoding stays unchanged.
    """

    def __init__(self, area_cap_bins: int = 5):
        if not TIMING_ACTION_SPACE:
            load_config()
        area_cfg = TIMING_ACTION_SPACE["area_cap"]
        area_values = self._build_value_grid(area_cfg, max(area_cap_bins, 2))
        area_low, area_high = self._extract_range(area_cfg, area_values)
        self.area_cap_values = self._normalize_values(area_values, area_low, area_high)

        slack_lesser_cfg = TIMING_ACTION_SPACE["slack_above"]
        slack_lesser_values = self._build_value_grid(slack_lesser_cfg, 3)
        slack_lesser_low, slack_lesser_high = self._extract_range(slack_lesser_cfg, slack_lesser_values)
        self.slack_lesser_values = self._normalize_values(slack_lesser_values, slack_lesser_low, slack_lesser_high)

        slack_greater_cfg = TIMING_ACTION_SPACE["slack_below"]
        slack_greater_values = self._build_value_grid(slack_greater_cfg, 3)
        slack_greater_low, slack_greater_high = self._extract_range(slack_greater_cfg, slack_greater_values)
        self.slack_greater_values = self._normalize_values(slack_greater_values, slack_greater_low, slack_greater_high)

        power_setup_guard_cfg = POWER_ACTION_SPACE["setup_guard"]
        power_setup_guard_values = self._build_value_grid(power_setup_guard_cfg, 3)
        power_setup_guard_low, power_setup_guard_high = self._extract_range(power_setup_guard_cfg, power_setup_guard_values)
        self.power_setup_guard_values = self._normalize_values(power_setup_guard_values, power_setup_guard_low, power_setup_guard_high)

        area_setup_guard_cfg = AREA_ACTION_SPACE["setup_guard"]
        area_setup_guard_values = self._build_value_grid(area_setup_guard_cfg, 3)
        area_setup_guard_low, area_setup_guard_high = self._extract_range(area_setup_guard_cfg, area_setup_guard_values)
        self.area_setup_guard_values = self._normalize_values(area_setup_guard_values, area_setup_guard_low, area_setup_guard_high)

        self.timing_actions = TIMING_ACTION_SPACE["actions"]
        self.timing_cell_classes = TIMING_ACTION_SPACE["cell_classes"]
        self.violation_types = TIMING_ACTION_SPACE.get("violation_type", ["setup", "hold"])
        self.site_modes = TIMING_ACTION_SPACE.get("site_mode", ["open_slot", "occupied_slot"])

        self.power_actions = POWER_ACTION_SPACE["actions"]
        self.power_cell_classes = POWER_ACTION_SPACE["cell_classes"]
        self.power_scopes = POWER_ACTION_SPACE.get("power_scope", ["total", "dynamic", "leakage"])

        self.area_actions = AREA_ACTION_SPACE["actions"]
        self.area_cell_classes = AREA_ACTION_SPACE["cell_classes"]

        self.actions = self._enumerate_actions()

    @property
    def size(self) -> int:
        return len(self.actions)

    def _binary_masks(self, length: int) -> List[np.ndarray]:
        """
        Generate all non-empty binary masks for a given length. This mirrors the
        multi-binary selections in the original action decoder.
        """
        masks = []
        for bits in itertools.product([0, 1], repeat=length):
            if sum(bits) == 0:
                continue  # skip all-zero selections
            masks.append(np.array(bits, dtype=np.float32))
        return masks

    def _enumerate_actions(self) -> List[np.ndarray]:
        """
        Build the full list of flattened actions. Layout matches
        `_convert_action_format` in `ECOEnvironment`:
            [0]: optimization_target (0-1 mapped to 0-2)
            [1]: violation_type (0-1 mapped to 0-1)
            [2]: site_mode (0-1 mapped to 0-1)
            [3:6]: timing actions (binary)
            [6:9]: timing cell types (binary)
            [9]: area_cap (normalized 0-1; decoded via config range)
            [10:12]: power actions (binary)
            [12:14]: power cell types (binary)
            [14]: power_scope (0-2 mapped to 0-1)
            [15:17]: area actions (binary)
            [17:20]: area cell types (binary)
            [20]: slack_above (normalized 0-1)
            [21]: slack_below (normalized 0-1)
            [22]: setup_guard (normalized 0-1; power/area only)
            [23]: noop flag (1 for no-op action)
        """
        actions = []

        # timing target
        for violation_type_idx, physical_idx, actions_mask, cell_mask, area_cap, slack_lt, slack_gt in itertools.product(
            range(len(self.violation_types)),
            range(len(self.site_modes)),
            self._binary_masks(len(self.timing_actions)),
            self._binary_masks(len(self.timing_cell_classes)),
            self.area_cap_values,
            self.slack_lesser_values,
            self.slack_greater_values,
        ):
            vec = np.zeros(24, dtype=np.float32)
            vec[0] = 0.0  # timing -> target_idx = 0
            vec[1] = float(violation_type_idx)  # already 0/1
            vec[2] = float(physical_idx)     # already 0/1
            vec[3:6] = actions_mask
            vec[6:9] = cell_mask
            vec[9] = float(area_cap)
            vec[20] = float(slack_lt)
            vec[21] = float(slack_gt)
            vec[22] = 0.0
            vec[23] = 0.0
            actions.append(vec)

        # power target
        for actions_mask, cell_mask, power_scope_idx, setup_guard in itertools.product(
            self._binary_masks(len(self.power_actions)),
            self._binary_masks(len(self.power_cell_classes)),
            range(len(self.power_scopes)),
            self.power_setup_guard_values,
        ):
            vec = np.zeros(24, dtype=np.float32)
            vec[0] = 0.5  # power -> target_idx = 1 -> 1/2
            vec[10:12] = actions_mask
            vec[12:14] = cell_mask
            vec[14] = float(power_scope_idx / max(len(self.power_scopes) - 1, 1))
            vec[22] = float(setup_guard)
            vec[23] = 0.0
            actions.append(vec)

        # area target
        for actions_mask, cell_mask, setup_guard in itertools.product(
            self._binary_masks(len(self.area_actions)),
            self._binary_masks(len(self.area_cell_classes)),
            self.area_setup_guard_values,
        ):
            vec = np.zeros(24, dtype=np.float32)
            vec[0] = 1.0  # area -> target_idx = 2 -> 2/2
            vec[15:17] = actions_mask
            vec[17:20] = cell_mask
            vec[22] = float(setup_guard)
            vec[23] = 0.0
            actions.append(vec)

        noop_vec = np.zeros(24, dtype=np.float32)
        noop_vec[23] = 1.0
        actions.append(noop_vec)

        return actions

    def _extract_range(self, cfg, values: np.ndarray):
        if isinstance(cfg, (list, tuple, np.ndarray)):
            return float(values.min()), float(values.max())
        return float(cfg["low"]), float(cfg["high"])

    def _build_value_grid(self, cfg, bins: int) -> np.ndarray:
        if isinstance(cfg, (list, tuple, np.ndarray)):
            return np.asarray(cfg, dtype=np.float32)
        return np.linspace(float(cfg["low"]), float(cfg["high"]), bins)

    def _normalize_values(self, values: np.ndarray, low: float, high: float) -> np.ndarray:
        span = max(high - low, 1e-6)
        return (values - low) / span

    def idx_to_action(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= len(self.actions):
            raise IndexError(f"Action index {idx} out of range [0, {len(self.actions) - 1}]")
        return self.actions[idx]

    def action_dims(self) -> int:
        """Return the flattened action dimensionality."""
        return len(self.actions[0]) if self.actions else 0
