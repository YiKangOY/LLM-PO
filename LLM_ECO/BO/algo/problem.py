# Author: baichen318@gmail.com


import os
import sys
import copy
import torch
import numpy as np
import torch.multiprocessing as mp
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, NoReturn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
ALGO_ROOT = os.path.join(ROOT, "BO", "algo")
UTILS_ROOT = os.path.join(ROOT, "BO", "utils")
if UTILS_ROOT not in sys.path:
    sys.path.insert(0, UTILS_ROOT)
if ALGO_ROOT not in sys.path:
    sys.path.append(ALGO_ROOT)

from utils import assert_error
from eco_config import load_config as load_eco_config, load_design_config
from eco_environment import ECOEnvironment, load_custom_env
from bo_action_space import BOActionSpace
from bo_report_parser import parse_power_report, parse_qor_report


OBJECTIVE_NAMES = ("area", "hold_tns", "power", "setup_tns")
MINIMIZE_OBJECTIVES = {"area", "power"}
OBJECTIVE_RANGE_KEYS = {
    "area": ("design_area", "total_cell_area", "area"),
    "hold_tns": ("hold_total_negative_slack", "hold_tns"),
    "power": ("total_power", "power"),
    "setup_tns": ("setup_total_negative_slack", "setup_tns"),
}


def _is_dict(value) -> bool:
    return isinstance(value, dict)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _first_float(values: List[Any], default: float = 0.0) -> float:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)


def _safe_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _metric_value(container: Any, key: str) -> Any:
    if isinstance(container, dict):
        return container.get(key)
    if hasattr(container, key):
        return getattr(container, key)
    return None


def _objectives_from_metrics(metrics: Optional[dict]) -> List[float]:
    metrics = _safe_dict(metrics)
    timing = _safe_dict(metrics.get("timing"))
    area = _safe_dict(metrics.get("area"))
    power = _safe_dict(metrics.get("power"))
    qor_metrics = metrics.get("qor_metrics")
    power_metrics = metrics.get("power_metrics")
    setup = _safe_dict(timing.get("setup"))
    hold = _safe_dict(timing.get("hold"))

    area_val = _first_float([
        area.get("design_area"),
        area.get("total_cell_area"),
        metrics.get("design_area"),
        _metric_value(qor_metrics, "design_area"),
        _metric_value(qor_metrics, "total_cell_area"),
    ])
    hold_tns_val = _first_float([
        timing.get("hold_tns"),
        timing.get("hold_total_negative_slack"),
        hold.get("tns"),
        metrics.get("hold_tns"),
        _metric_value(qor_metrics, "hold_total_negative_slack"),
    ])
    power_val = _first_float([
        power.get("total"),
        power.get("total_power"),
        metrics.get("power"),
        metrics.get("total_power"),
        _metric_value(power_metrics, "total_power"),
    ])
    setup_tns_val = _first_float([
        timing.get("setup_tns"),
        timing.get("setup_total_negative_slack"),
        setup.get("tns"),
        metrics.get("setup_tns"),
        _metric_value(qor_metrics, "setup_total_negative_slack"),
    ])
    return [area_val, hold_tns_val, power_val, setup_tns_val]


def _directional_ref_point_from_ranges(normalization_ranges: Any) -> Optional[List[float]]:
    normalization_ranges = _safe_dict(normalization_ranges)
    if not normalization_ranges:
        return None

    ref_point = []
    for name in OBJECTIVE_NAMES:
        bounds = None
        for key in OBJECTIVE_RANGE_KEYS[name]:
            cfg = normalization_ranges.get(key)
            if _is_dict(cfg) and "min" in cfg and "max" in cfg:
                bounds = (_to_float(cfg["min"]), _to_float(cfg["max"]))
                break
        if bounds is None:
            return None
        if name in MINIMIZE_OBJECTIVES:
            ref_point.append(bounds[1])
        else:
            ref_point.append(bounds[0])
    return ref_point


def _coerce_ref_point_to_worst_direction(
    ref_point: Optional[List[float]],
    directional_floor: Optional[List[float]],
) -> Optional[List[float]]:
    if ref_point is None:
        return directional_floor
    if directional_floor is None:
        return [_to_float(v) for v in ref_point]

    coerced = []
    for idx, name in enumerate(OBJECTIVE_NAMES):
        value = _to_float(ref_point[idx])
        floor = _to_float(directional_floor[idx])
        if name in MINIMIZE_OBJECTIVES:
            coerced.append(max(value, floor))
        else:
            coerced.append(min(value, floor))
    return coerced


def _read_report(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _append_path(paths: List[str], value: Optional[str], suffix: Optional[str] = None) -> None:
    if not value:
        return
    path = os.path.abspath(os.path.join(value, suffix)) if suffix else os.path.abspath(value)
    if path not in paths:
        paths.append(path)


def _candidate_reports_dirs(env_kwargs: dict) -> List[str]:
    dirs: List[str] = []
    run_paths = env_kwargs.get("run_paths")
    if _is_dict(run_paths):
        _append_path(dirs, run_paths.get("reports_dir"))
        _append_path(dirs, run_paths.get("run_dir"), "reports")
        _append_path(dirs, run_paths.get("workspace"), "reports")
        _append_path(dirs, run_paths.get("work_root"), "reports")
        _append_path(dirs, run_paths.get("base_path"), "reports")
    _append_path(dirs, env_kwargs.get("base_path"), "reports")
    _append_path(dirs, env_kwargs.get("workspace"), "reports")
    return dirs


def _read_reference_point_from_reports(env_kwargs: dict) -> Optional[List[float]]:
    for reports_dir in _candidate_reports_dirs(env_kwargs):
        qor_path = os.path.join(reports_dir, "report_qor_0.txt")
        power_path = os.path.join(reports_dir, "report_power_0.txt")
        if not (os.path.exists(qor_path) and os.path.exists(power_path)):
            continue
        try:
            qor_data = parse_qor_report(_read_report(qor_path))
            power_data = parse_power_report(_read_report(power_path))
            return [
                _to_float(qor_data.get("design_area")),
                _to_float(qor_data.get("hold_total_negative_slack")),
                _to_float(power_data.get("total_power")),
                _to_float(qor_data.get("setup_total_negative_slack")),
            ]
        except Exception:
            continue
    return None


def _build_bo_run_dir(workspace: str, env_id: int) -> str:
    return os.path.join(workspace, "BO", "run_dir", "run_{}".format(env_id))


def _resolve_workspace(env_kwargs: dict, run_paths: dict) -> Optional[str]:
    # Match RL: pt_shell workspace should be the design base_path when present.
    return (
        env_kwargs.get("base_path")
        or env_kwargs.get("workspace")
        or run_paths.get("workspace")
        or run_paths.get("work_root")
        or run_paths.get("base_path")
    )


def _normalize_run_paths_for_bo(env_kwargs: dict) -> dict:
    run_paths = env_kwargs.get("run_paths")
    if _is_dict(run_paths):
        normalized = run_paths.copy()
    else:
        normalized = {}

    env_id = int(normalized.get("env_id", 0))
    normalized["env_id"] = env_id
    workspace = _resolve_workspace(env_kwargs, normalized)
    if workspace:
        normalized["workspace"] = workspace
        if "run_dir" not in normalized or not normalized["run_dir"]:
            normalized["run_dir"] = _build_bo_run_dir(workspace, env_id)

    # Keep reports rooted at run_dir like RL; legacy reports_root should not
    # drive runtime workspace selection.
    normalized.pop("reports_root", None)
    return normalized


def _clone_runner_config_for_worker(runner_cfg: dict, worker_idx: int) -> dict:
    """
    Clone and specialize runner config so each parallel worker uses isolated
    runtime paths (env_id + run_dir), avoiding cross-worker file races.
    """
    worker_cfg = copy.deepcopy(runner_cfg)
    env_kwargs = worker_cfg.get("env-kwargs")
    if not _is_dict(env_kwargs):
        return worker_cfg

    run_paths = _normalize_run_paths_for_bo(env_kwargs)
    run_paths["env_id"] = int(worker_idx)
    workspace = _resolve_workspace(env_kwargs, run_paths)
    if workspace:
        run_paths["workspace"] = workspace
        run_paths["run_dir"] = _build_bo_run_dir(workspace, int(worker_idx))

    env_kwargs = env_kwargs.copy()
    env_kwargs["run_paths"] = run_paths
    worker_cfg["env-kwargs"] = env_kwargs
    return worker_cfg


def _evaluate_sequence_worker(args):
    runner_cfg, env_config, sequence, sequence_length, worker_idx = args
    runner_cfg = _clone_runner_config_for_worker(runner_cfg, worker_idx)
    env = _build_env(runner_cfg, env_config)
    action_dim = int(env.action_space.shape[0])
    action_low = np.array(env.action_space.low).reshape(-1)
    action_high = np.array(env.action_space.high).reshape(-1)
    _sequence = sequence.view(sequence_length, action_dim).cpu().numpy()
    _sequence = np.clip(_sequence, action_low, action_high)
    env.reset()
    last_metrics = {}
    for i in range(sequence_length):
        _, reward, terminated, truncated, info = env.step(_sequence[i])
        info = info or {}
        if info.get("metrics_snapshot") is not None:
            last_metrics = info["metrics_snapshot"]
        if terminated or truncated:
            break
    return {
        "objectives": _objectives_from_metrics(last_metrics),
        "metrics": last_metrics,
    }


def _build_env(configs: dict, env_config: dict) -> ECOEnvironment:
    env_entrypoint = configs.get("env-entrypoint")
    env_kwargs = (configs.get("env-kwargs") or {}).copy()
    env_kwargs["run_paths"] = _normalize_run_paths_for_bo(env_kwargs)
    custom_env = None
    if env_entrypoint:
        if "use_simulation" not in env_kwargs:
            env_kwargs["use_simulation"] = configs.get("use-simulation", True)
        custom_env = load_custom_env(env_entrypoint, env_kwargs, env_config)
    env = ECOEnvironment(
        config=env_config,
        use_simulation=configs.get("use-simulation", True),
        custom_env=custom_env
    )
    return env


class BaseProblem(torch.nn.Module, ABC):
    """
        base class for construction a problem.
    """

    dim: int
    _bounds: List[Tuple[float, float]]
    _check_grad_at_opt: bool = True

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        """
            base class for construction a problem.

        args:
            noise_std: standard deviation of the observation noise.
            negate: if True, negate the function.
        """
        super().__init__()
        self.noise_std = noise_std
        self.negate = negate
        self.register_buffer(
            "bounds", torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)
        )

    def forward(self, X: Tensor, noise: bool = True) -> Tensor:
        """
            evaluate the function on a set of points.

        args:
            X: a `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                function.
            noise: if `True`, add observation noise as specified by `noise_std`.

        returns:
            a `batch_shape`-dim tensor of function evaluations.
        """
        batch = X.ndimension() > 1
        X = X if batch else X.unsqueeze(0)
        f = self.evaluate_true(X=X)
        if noise and self.noise_std is not None:
            f += self.noise_std * torch.randn_like(f)
        if self.negate:
            f = -f
        return f if batch else f.squeeze(0)

    @abstractmethod
    def evaluate_true(self, X: Tensor) -> Tensor:
        """
            evaluate the function (w/o observation noise) on a set of points.
        """
        raise NotImplementedError


class MultiObjectiveProblem(BaseProblem):
    """
        base class for a multi-objective problem.
    """

    num_objectives: int
    _ref_point: List[float]
    _max_hv: float

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        """
            base constructor for multi-objective test functions.

        args:
            noise_std: standard deviation of the observation noise.
            negate: if True, negate the objectives.
        """
        super().__init__(noise_std=noise_std, negate=negate)
        ref_point = torch.tensor(self._ref_point, dtype=torch.float)
        if negate:
            ref_point *= -1
        self.register_buffer("ref_point", ref_point)

    @property
    def max_hv(self) -> float:
        try:
            return self._max_hv
        except AttributeError:
            raise NotImplementedError(
                error_message("problem {} does not specify maximal hypervolume".format(
                    self.__class__.__name__)
                )
            )

    def gen_pareto_front(self, n: int) -> Tensor:
        """
            generate `n` pareto optimal points.
        """
        raise NotImplementedError


class RLSequenceRunner(object):
    def __init__(self, configs: dict):
        super(RLSequenceRunner, self).__init__()
        self.configs = configs
        config_overrides = configs.get("bo-config-overrides")
        if config_overrides is None:
            config_overrides = configs.get("rl-config-overrides")
        design_cfg = load_design_config(
            configs.get("design-configs"),
            configs.get("design-name")
        )
        if design_cfg and design_cfg.get("env_kwargs"):
            merged_kwargs = design_cfg.get("env_kwargs", {}).copy()
            merged_kwargs.update(configs.get("env-kwargs", {}) or {})
            self.configs["env-kwargs"] = merged_kwargs
        if design_cfg and design_cfg.get("run_paths"):
            self.configs.setdefault("env-kwargs", {})
            self.configs["env-kwargs"].setdefault("run_paths", design_cfg["run_paths"])
        self.env_config = load_eco_config(
            config_overrides=config_overrides,
            design_config=design_cfg
        )
        self.discrete_action_space = None
        if "action_spaces" in self.env_config and self.env_config["action_spaces"]:
            self.discrete_action_space = BOActionSpace(self.env_config["action_spaces"])
        self.sequence_length = configs.get("episode-length")
        if not self.sequence_length:
            self.sequence_length = self.env_config["max_iterations_per_episode"]
        self.num_envs = configs.get("num-envs", 1)
        self.env = _build_env(self.configs, self.env_config)
        self.action_dim = int(self.env.action_space.shape[0])
        self.action_low, self.action_high = self._resolve_action_bounds(design_cfg)
        self.sequence_low = np.tile(self.action_low, self.sequence_length).astype(np.float32)
        self.sequence_high = np.tile(self.action_high, self.sequence_length).astype(np.float32)
        self.n_dim = self.sequence_length * self.action_dim
        self.ref_point = self._resolve_ref_point()

    def _resolve_action_bounds(self, design_cfg: Optional[dict]) -> Tuple[np.ndarray, np.ndarray]:
        action_cfg = _safe_dict(_safe_dict(design_cfg).get("action_space"))
        low = action_cfg.get("low")
        high = action_cfg.get("high")
        if low is None or high is None:
            low = np.array(self.env.action_space.low, dtype=np.float32).reshape(-1)
            high = np.array(self.env.action_space.high, dtype=np.float32).reshape(-1)
            return low, high
        low_arr = np.array(low, dtype=np.float32).reshape(-1)
        high_arr = np.array(high, dtype=np.float32).reshape(-1)
        if low_arr.size == 1:
            low_arr = np.full(self.action_dim, low_arr[0], dtype=np.float32)
        if high_arr.size == 1:
            high_arr = np.full(self.action_dim, high_arr[0], dtype=np.float32)
        if low_arr.size != self.action_dim or high_arr.size != self.action_dim:
            env_low = np.array(self.env.action_space.low, dtype=np.float32).reshape(-1)
            env_high = np.array(self.env.action_space.high, dtype=np.float32).reshape(-1)
            return env_low, env_high
        return low_arr, high_arr

    def _resolve_ref_point(self) -> List[float]:
        env_kwargs = _safe_dict(self.configs.get("env-kwargs"))
        ref_point = _read_reference_point_from_reports(env_kwargs)
        if ref_point is not None:
            return ref_point
        directional_ref = _directional_ref_point_from_ranges(
            _safe_dict(self.env_config).get("normalization_ranges")
        )
        cfg_ref = self.configs.get("ref-point")
        if isinstance(cfg_ref, (list, tuple)) and len(cfg_ref) == len(OBJECTIVE_NAMES):
            return _coerce_ref_point_to_worst_direction(
                [_to_float(v) for v in cfg_ref],
                directional_ref,
            )
        if directional_ref is not None:
            return directional_ref
        return [0.0 for _ in OBJECTIVE_NAMES]

    def _sample_action_sequence(self) -> np.ndarray:
        """
        Sample one full action sequence [sequence_length, action_dim], where each
        action dimension respects the selected design-config action bounds.
        """
        if self.discrete_action_space is not None:
            return self.discrete_action_space.sample(self.sequence_length).astype(np.float32)
        return np.random.uniform(
            self.action_low,
            self.action_high,
            size=(self.sequence_length, self.action_dim),
        ).astype(np.float32)

    def clip_flat_sequences(self, sequences: torch.Tensor) -> torch.Tensor:
        if sequences.numel() == 0:
            return sequences.to(torch.float32)
        low = torch.from_numpy(self.sequence_low).to(sequences.device, dtype=sequences.dtype)
        high = torch.from_numpy(self.sequence_high).to(sequences.device, dtype=sequences.dtype)
        if sequences.ndimension() == 1:
            return torch.max(torch.min(sequences, high), low).to(torch.float32)
        return torch.max(torch.min(sequences, high.unsqueeze(0)), low.unsqueeze(0)).to(torch.float32)

    def sample_candidates(self, pool_size: int) -> torch.Tensor:
        candidates = []
        for i in range(pool_size):
            sequence = self._sample_action_sequence().reshape(-1)
            sequence = np.clip(sequence, self.sequence_low, self.sequence_high)
            candidates.append(sequence)
        candidates = np.stack(candidates, axis=0)
        return torch.tensor(candidates, dtype=torch.float32)

    def evaluate_sequence(self, sequence: torch.Tensor, return_details: bool = False):
        sequence = self.clip_flat_sequences(sequence.reshape(-1))
        _sequence = sequence.view(self.sequence_length, self.action_dim).cpu().numpy()
        _sequence = np.clip(_sequence, self.action_low, self.action_high)
        self.env.reset()
        last_metrics = {}
        for i in range(self.sequence_length):
            _, reward, terminated, truncated, info = self.env.step(_sequence[i])
            info = info or {}
            if info.get("metrics_snapshot") is not None:
                last_metrics = info["metrics_snapshot"]
            if terminated or truncated:
                break
        value = torch.tensor(_objectives_from_metrics(last_metrics), dtype=torch.float32)
        if return_details:
            return value, last_metrics
        return value

    def evaluate_sequences(self, sequences: torch.Tensor, return_details: bool = False):
        if sequences.ndimension() > 1 and sequences.size()[0] == 0:
            values = torch.empty((0, len(OBJECTIVE_NAMES)), dtype=torch.float32)
            if return_details:
                return values, []
            return values
        if sequences.ndimension() == 1:
            return self.evaluate_sequence(sequences, return_details=return_details)
        if self.num_envs <= 1:
            outputs = []
            details = []
            for i in range(sequences.size()[0]):
                if return_details:
                    value, detail = self.evaluate_sequence(sequences[i], return_details=True)
                    outputs.append(value)
                    details.append(detail)
                else:
                    outputs.append(self.evaluate_sequence(sequences[i]))
            values = torch.stack(outputs)
            if return_details:
                return values, details
            return values
        params = []
        for i in range(sequences.size()[0]):
            params.append((self.configs, self.env_config, sequences[i], self.sequence_length, i))
        start_method = "spawn"
        if "fork" in mp.get_all_start_actions():
            start_method = "fork"
        with mp.get_context(start_method).Pool(self.num_envs) as pool:
            results = pool.map(_evaluate_sequence_worker, params)
        if return_details:
            values = torch.tensor([item["objectives"] for item in results], dtype=torch.float32)
            details = [item.get("metrics", {}) for item in results]
            return values, details
        if results and isinstance(results[0], dict):
            return torch.tensor([item["objectives"] for item in results], dtype=torch.float32)
        return torch.tensor(results, dtype=torch.float32)


class DesignSpaceProblem(MultiObjectiveProblem):
    def __init__(self, configs: dict):
        self.configs = configs
        self.mode = configs["mode"]
        self.rl_runner = None
        assert self.mode == "online", assert_error("only online RL-style mode is supported now.")
        self.rl_runner = RLSequenceRunner(self.configs["rl-bo"])
        self.x = self.rl_runner.sample_candidates(
            self.configs["rl-bo"]["candidate-pool"]
        )
        self.total_x = self.x.clone()
        self.n_dim = self.rl_runner.n_dim
        self.n_sample = self.x.shape[0]
        self.last_eval_details = []
        # derive per-dimension bounds from the underlying RL action space
        action_low = np.array(self.rl_runner.action_low).reshape(-1)
        action_high = np.array(self.rl_runner.action_high).reshape(-1)
        bounds = []
        for _ in range(self.rl_runner.sequence_length):
            for dim in range(self.rl_runner.action_dim):
                lo = float(action_low[dim] if action_low.size > 1 else action_low[0])
                hi = float(action_high[dim] if action_high.size > 1 else action_high[0])
                bounds.append((lo, hi))
        self._ref_point = torch.tensor(self.rl_runner.ref_point, dtype=torch.float32)
        self._bounds = torch.tensor(bounds)
        super().__init__()
        self.x = self.clip_to_bounds(self.x)

    def evaluate_true(self, x: torch.Tensor) -> torch.Tensor:
        evaluated = self.rl_runner.evaluate_sequences(x, return_details=True)
        if isinstance(evaluated, tuple) and len(evaluated) == 2:
            values, details = evaluated
            if isinstance(details, list):
                self.last_eval_details = details
            else:
                self.last_eval_details = [details]
            return values
        self.last_eval_details = []
        return evaluated

    def remove_sampled_data(self, x: torch.Tensor) -> NoReturn:
        if x.ndimension() == 1:
            _x = x.unsqueeze(0)
        else:
            _x = x
        sampled = torch.ones(self.x.size()[0], dtype=torch.bool)
        for i in range(self.x.size()[0]):
            for j in range(_x.size()[0]):
                if torch.equal(self.x[i], _x[j]):
                    sampled[i] = False
        self.x = self.x[sampled[:]]
        self.n_sample = self.x.shape[0]

    def clip_to_bounds(self, x: torch.Tensor) -> torch.Tensor:
        return self.rl_runner.clip_flat_sequences(x)

    def initial_samples(self):
        n_init = int(self.configs.get("rl-bo", {}).get("initial-samples", 0))
        if n_init > self.x.size()[0]:
            n_init = self.x.size()[0]
        if n_init <= 0:
            n_obj = int(self._ref_point.numel())
            return self.x[:0], torch.empty((0, n_obj), dtype=torch.float32)
        init_x = self.x[:n_init]
        init_y = self.evaluate_true(init_x)
        self.remove_sampled_data(init_x)
        return init_x, init_y

    def replenish_candidates(self):
        """
            Maintain a candidate pool for online BO by resampling random action
            sequences from the RL environment when the pool shrinks.
        """
        target_pool = int(self.configs["rl-bo"]["candidate-pool"])
        current = self.x.shape[0]
        if current >= target_pool:
            return
        needed = target_pool - current
        extra = self.rl_runner.sample_candidates(needed)
        self.x = torch.cat((self.x, extra), dim=0)
        self.x = self.clip_to_bounds(self.x)
        self.n_sample = self.x.shape[0]


def create_problem(configs: dict) -> DesignSpaceProblem:
    return DesignSpaceProblem(configs)
