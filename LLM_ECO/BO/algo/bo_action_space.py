import itertools
import numpy as np


class BOActionSpace(object):
    """
    BO-local discrete ECO action space mirroring RL/eco_action_space.py.
    Each action is a normalized 24-d vector accepted by RL's decoder.
    """

    def __init__(self, action_spaces: dict, area_cap_bins: int = 5):
        self.action_spaces = action_spaces
        self.timing_action_space = action_spaces["timing"]
        self.power_action_space = action_spaces["power"]
        self.area_action_space = action_spaces["area"]

        area_cfg = self.timing_action_space["area_cap"]
        area_values = self._build_value_grid(area_cfg, max(area_cap_bins, 2))
        area_low, area_high = self._extract_range(area_cfg, area_values)
        self.area_cap_values = self._normalize_values(area_values, area_low, area_high)

        slack_lesser_cfg = self.timing_action_space["slack_above"]
        slack_lesser_values = self._build_value_grid(slack_lesser_cfg, 3)
        slack_lesser_low, slack_lesser_high = self._extract_range(
            slack_lesser_cfg,
            slack_lesser_values,
        )
        self.slack_lesser_values = self._normalize_values(
            slack_lesser_values,
            slack_lesser_low,
            slack_lesser_high,
        )

        slack_greater_cfg = self.timing_action_space["slack_below"]
        slack_greater_values = self._build_value_grid(slack_greater_cfg, 3)
        slack_greater_low, slack_greater_high = self._extract_range(
            slack_greater_cfg,
            slack_greater_values,
        )
        self.slack_greater_values = self._normalize_values(
            slack_greater_values,
            slack_greater_low,
            slack_greater_high,
        )

        power_setup_guard_cfg = self.power_action_space["setup_guard"]
        power_setup_guard_values = self._build_value_grid(power_setup_guard_cfg, 3)
        power_setup_guard_low, power_setup_guard_high = self._extract_range(
            power_setup_guard_cfg,
            power_setup_guard_values,
        )
        self.power_setup_guard_values = self._normalize_values(
            power_setup_guard_values,
            power_setup_guard_low,
            power_setup_guard_high,
        )

        area_setup_guard_cfg = self.area_action_space["setup_guard"]
        area_setup_guard_values = self._build_value_grid(area_setup_guard_cfg, 3)
        area_setup_guard_low, area_setup_guard_high = self._extract_range(
            area_setup_guard_cfg,
            area_setup_guard_values,
        )
        self.area_setup_guard_values = self._normalize_values(
            area_setup_guard_values,
            area_setup_guard_low,
            area_setup_guard_high,
        )

        self.timing_actions = self.timing_action_space["actions"]
        self.timing_cell_classes = self.timing_action_space["cell_classes"]
        self.violation_types = self.timing_action_space["violation_type"]
        self.site_modes = self.timing_action_space["site_mode"]

        self.power_actions = self.power_action_space["actions"]
        self.power_cell_classes = self.power_action_space["cell_classes"]
        self.power_scopes = self.power_action_space["power_scope"]

        self.area_actions = self.area_action_space["actions"]
        self.area_cell_classes = self.area_action_space["cell_classes"]

        self.actions = np.stack(self._enumerate_actions(), axis=0).astype(np.float32)

    @property
    def size(self) -> int:
        return self.actions.shape[0]

    def _binary_masks(self, length: int):
        masks = []
        for bits in itertools.product([0, 1], repeat=length):
            if sum(bits) == 0:
                continue
            masks.append(np.array(bits, dtype=np.float32))
        return masks

    def _enumerate_actions(self):
        actions = []

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
            vec[0] = 0.0
            vec[1] = float(violation_type_idx)
            vec[2] = float(physical_idx)
            vec[3:6] = actions_mask
            vec[6:9] = cell_mask
            vec[9] = float(area_cap)
            vec[20] = float(slack_lt)
            vec[21] = float(slack_gt)
            actions.append(vec)

        for actions_mask, cell_mask, power_scope_idx, setup_guard in itertools.product(
            self._binary_masks(len(self.power_actions)),
            self._binary_masks(len(self.power_cell_classes)),
            range(len(self.power_scopes)),
            self.power_setup_guard_values,
        ):
            vec = np.zeros(24, dtype=np.float32)
            vec[0] = 0.5
            vec[10:12] = actions_mask
            vec[12:14] = cell_mask
            vec[14] = float(power_scope_idx / max(len(self.power_scopes) - 1, 1))
            vec[22] = float(setup_guard)
            actions.append(vec)

        for actions_mask, cell_mask, setup_guard in itertools.product(
            self._binary_masks(len(self.area_actions)),
            self._binary_masks(len(self.area_cell_classes)),
            self.area_setup_guard_values,
        ):
            vec = np.zeros(24, dtype=np.float32)
            vec[0] = 1.0
            vec[15:17] = actions_mask
            vec[17:20] = cell_mask
            vec[22] = float(setup_guard)
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

    def sample(self, count: int) -> np.ndarray:
        indices = np.random.randint(0, self.size, size=count)
        return self.actions[indices]
