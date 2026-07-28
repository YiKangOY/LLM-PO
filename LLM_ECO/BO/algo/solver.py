# Author: baichen318@gmail.com


import os
import json
import torch
import gpytorch
import numpy as np
from typing import Any, Dict, NoReturn
from model import initialize_dkl_gp
from problem import DesignSpaceProblem, OBJECTIVE_NAMES, _objectives_from_metrics
from utils import info, mkdir, write_txt
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from datetime import datetime


def _metric_value(container: Any, key: str) -> Any:
    if isinstance(container, dict):
        return container.get(key)
    if hasattr(container, key):
        return getattr(container, key)
    return None


def _coerce_normalized_ref_point(
    normalized_ref: torch.Tensor,
    normalized_y: torch.Tensor,
    margin: float = 1e-6,
) -> torch.Tensor:
    if normalized_y.ndimension() == 1:
        normalized_y = normalized_y.unsqueeze(0)
    if normalized_y.size(0) == 0:
        return normalized_ref
    return torch.minimum(
        normalized_ref,
        normalized_y.min(dim=0).values - margin,
    )


class BORunLogger(object):
    """
    RL-style JSONL logger: one structured event per line.
    """

    def __init__(self, log_dir: str, base_name: str = "bo_explorer_train"):
        mkdir(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, "{}_{}.log".format(base_name, timestamp))
        self._fh = open(self.log_path, "a", encoding="utf-8")

    def log_event(self, event: str, payload: dict) -> NoReturn:
        record = {
            "event": event,
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        record.update(payload)
        self._fh.write(json.dumps(record, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> NoReturn:
        if getattr(self, "_fh", None):
            self._fh.close()
            self._fh = None


class BOOMExplorerSolver(object):
    def __init__(self, problem: object):
        super(BOOMExplorerSolver, self).__init__()
        self.problem = problem
        self.objective_normalizer: Dict[str, torch.Tensor] = {}
        self.minimize_objectives = {"area", "power"}
        self.objective_weights = self._resolve_objective_weights()
        self.paths = self._resolve_paths()
        self.run_logger = BORunLogger(self.paths["logs_dir"], "bo_explorer_train")
        self.episode_counter = 0
        self.run_logger.log_event(
            "run_started",
            {
                "design": self.problem.rl_runner.configs.get("design-name"),
                "candidate_pool": int(self.problem.configs["rl-bo"]["candidate-pool"]),
                "initial_samples": int(
                    self.problem.configs.get("rl-bo", {}).get("initial-samples", 0)
                ),
                "max_bo_steps": int(self.problem.configs["bo"]["max-bo-steps"]),
                "objective_order": list(OBJECTIVE_NAMES),
                "ref_point": [float(v) for v in self.problem._ref_point.detach().cpu().tolist()],
                "objective_weights": {
                    name: float(self.objective_weights[idx].item())
                    for idx, name in enumerate(OBJECTIVE_NAMES)
                },
            },
        )

    @staticmethod
    def _as_float(val, default=0.0):
        try:
            return float(val)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _first_not_none(values, default=None):
        for value in values:
            if value is not None:
                return value
        return default

    def _extract_ppa_metrics(self, detail: dict, objective_values: np.ndarray = None) -> dict:
        detail = detail or {}
        timing = detail.get("timing", {}) if isinstance(detail, dict) else {}
        setup = timing.get("setup", {}) if isinstance(timing, dict) else {}
        hold = timing.get("hold", {}) if isinstance(timing, dict) else {}
        area = detail.get("area", {}) if isinstance(detail, dict) else {}
        power = detail.get("power", {}) if isinstance(detail, dict) else {}
        qor_metrics = detail.get("qor_metrics", {}) if isinstance(detail, dict) else {}
        power_metrics = detail.get("power_metrics", {}) if isinstance(detail, dict) else {}
        if objective_values is None:
            objective_values = _objectives_from_metrics(detail)
        area_obj = self._as_float(objective_values[0])
        hold_tns_obj = self._as_float(objective_values[1])
        power_obj = self._as_float(objective_values[2])
        setup_tns_obj = self._as_float(objective_values[3])

        total_cell_area = self._as_float(
            self._first_not_none(
                [
                    area.get("total_cell_area"),
                    _metric_value(qor_metrics, "total_cell_area"),
                    detail.get("total_cell_area"),
                ],
                area_obj,
            ),
            area_obj,
        )

        setup_wns = self._as_float(
            self._first_not_none(
                [
                    setup.get("wns"),
                    timing.get("setup_wns"),
                    timing.get("setup_critical_path_slack"),
                    _metric_value(qor_metrics, "setup_critical_path_slack"),
                ],
                0.0,
            )
        )
        hold_wns = self._as_float(
            self._first_not_none(
                [
                    hold.get("wns"),
                    timing.get("hold_wns"),
                    timing.get("hold_critical_path_slack"),
                    _metric_value(qor_metrics, "hold_critical_path_slack"),
                ],
                0.0,
            )
        )

        internal_power = self._as_float(
            self._first_not_none(
                [
                    power.get("internal"),
                    power.get("internal_power"),
                    _metric_value(power_metrics, "internal_power"),
                ],
                0.0,
            )
        )
        switching_power = self._as_float(
            self._first_not_none(
                [
                    power.get("switching"),
                    power.get("switching_power"),
                    power.get("dynamic"),
                    power.get("dynamic_power"),
                    _metric_value(power_metrics, "switching_power"),
                    _metric_value(power_metrics, "dynamic_power"),
                ],
                0.0,
            )
        )
        leakage_power = self._as_float(
            self._first_not_none(
                [
                    power.get("leakage"),
                    power.get("leakage_power"),
                    _metric_value(power_metrics, "leakage_power"),
                ],
                0.0,
            )
        )
        return {
            "timing": {
                "setup_tns": setup_tns_obj,
                "hold_tns": hold_tns_obj,
                "setup_wns": setup_wns,
                "hold_wns": hold_wns,
            },
            "area": {
                "design_area": area_obj,
                "total_cell_area": total_cell_area,
            },
            "power": {
                "total": power_obj,
                "internal": internal_power,
                "switching": switching_power,
                "leakage": leakage_power,
            },
        }

    def _extract_rl_final_metrics(self, objective_values: np.ndarray) -> dict:
        return {
            "area": self._as_float(objective_values[0]),
            "hold_tns": self._as_float(objective_values[1]),
            "power": self._as_float(objective_values[2]),
            "setup_tns": self._as_float(objective_values[3]),
        }

    def _log_episode_metrics(
        self,
        phase: str,
        objective_values: np.ndarray,
        detail: dict,
    ) -> NoReturn:
        self.episode_counter += 1
        metrics = self._extract_ppa_metrics(detail=detail, objective_values=objective_values)
        final = {
            name: float(objective_values[idx])
            for idx, name in enumerate(OBJECTIVE_NAMES)
        }
        payload = {
            "episode": int(self.episode_counter),
            "phase": phase,
            "objective_order": list(OBJECTIVE_NAMES),
            "objectives": final,
            "final": self._extract_rl_final_metrics(objective_values),
            "metrics": metrics,
        }
        self.run_logger.log_event("episode_metrics", payload)

    def _last_eval_details(self, expected_count: int):
        details = getattr(self.problem, "last_eval_details", None)
        if not isinstance(details, list):
            details = []
        if len(details) < expected_count:
            details = details + [{} for _ in range(expected_count - len(details))]
        return details[:expected_count]

    def _resolve_paths(self) -> dict:
        env_kwargs = self.problem.rl_runner.configs.get("env-kwargs") or {}
        base_path = env_kwargs.get("base_path")
        if not base_path:
            run_paths = env_kwargs.get("run_paths")
            if isinstance(run_paths, dict):
                base_path = (
                    run_paths.get("workspace")
                    or run_paths.get("work_root")
                    or run_paths.get("base_path")
                )
        if base_path:
            bo_root = os.path.join(base_path, "BO")
            return {
                "bo_root": bo_root,
                "results_dir": os.path.join(bo_root, "results"),
                "models_dir": os.path.join(bo_root, "models"),
                "logs_dir": os.path.join(bo_root, "logs"),
            }
        # Fallback path (no base_path): keep results/models behavior, still add logs subdir.
        root = self.problem.configs["report"]["path"]
        return {
            "bo_root": root,
            "results_dir": root,
            "models_dir": root,
            "logs_dir": os.path.join(root, "logs"),
        }

    def _resolve_objective_weights(self) -> torch.Tensor:
        defaults = {
            "area": 1.0,
            "hold_tns": 1.0,
            "power": 1.0,
            "setup_tns": 3.0,
        }
        env_cfg = getattr(self.problem.rl_runner, "env_config", {}) or {}
        reward_weights = env_cfg.get("reward_weights", {}) if isinstance(env_cfg, dict) else {}
        cfg = self.problem.configs.get("rl-bo", {}) if isinstance(self.problem.configs, dict) else {}
        configured = cfg.get("objective-weights", {}) if isinstance(cfg, dict) else {}
        weights = []
        for name in OBJECTIVE_NAMES:
            if isinstance(configured, dict) and name in configured:
                val = configured[name]
            elif isinstance(reward_weights, dict) and name in reward_weights:
                val = reward_weights[name]
            else:
                val = defaults.get(name, 1.0)
            try:
                weight = float(val)
            except (TypeError, ValueError):
                weight = defaults.get(name, 1.0)
            if weight <= 0:
                weight = defaults.get(name, 1.0)
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.float32)

    def _build_objective_normalizer(self, raw_y: torch.Tensor) -> Dict[str, torch.Tensor]:
        env_cfg = getattr(self.problem.rl_runner, "env_config", {}) or {}
        norm_cfg = env_cfg.get("normalization_ranges", {}) if isinstance(env_cfg, dict) else {}
        key_map = {
            "area": ("design_area", "total_cell_area", "area"),
            "hold_tns": ("hold_total_negative_slack", "hold_tns"),
            "power": ("total_power", "power"),
            "setup_tns": ("setup_total_negative_slack", "setup_tns"),
        }
        lowers = []
        uppers = []
        for idx, name in enumerate(OBJECTIVE_NAMES):
            lo = None
            hi = None
            for key in key_map.get(name, ()):
                cfg = norm_cfg.get(key)
                if isinstance(cfg, dict) and "min" in cfg and "max" in cfg:
                    try:
                        lo = float(cfg["min"])
                        hi = float(cfg["max"])
                        break
                    except (TypeError, ValueError):
                        pass
            if lo is None or hi is None:
                col = raw_y[:, idx]
                lo = float(torch.min(col).item())
                hi = float(torch.max(col).item())
            if hi - lo < 1e-12:
                hi = lo + 1.0
            lowers.append(lo)
            uppers.append(hi)
        return {
            "lower": torch.tensor(lowers, dtype=torch.float32),
            "upper": torch.tensor(uppers, dtype=torch.float32),
        }

    def _maximize_mask(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(
            [name not in self.minimize_objectives for name in OBJECTIVE_NAMES],
            dtype=dtype,
            device=device,
        )

    def _normalize_objectives(self, y: torch.Tensor) -> torch.Tensor:
        lower = self.objective_normalizer["lower"].to(y.device, dtype=y.dtype)
        upper = self.objective_normalizer["upper"].to(y.device, dtype=y.dtype)
        denom = (upper - lower).clamp_min(1e-12)
        maximize = self._maximize_mask(device=y.device, dtype=torch.bool)
        weights = self.objective_weights.to(y.device, dtype=y.dtype)
        normalized = torch.empty_like(y)
        normalized[:, maximize] = (y[:, maximize] - lower[maximize]) / denom[maximize]
        normalized[:, ~maximize] = (upper[~maximize] - y[:, ~maximize]) / denom[~maximize]
        return normalized * weights

    def _normalize_ref_point(self, ref_point: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        lower = self.objective_normalizer["lower"].to(device=device, dtype=dtype)
        upper = self.objective_normalizer["upper"].to(device=device, dtype=dtype)
        denom = (upper - lower).clamp_min(1e-12)
        maximize = self._maximize_mask(device=device, dtype=torch.bool)
        weights = self.objective_weights.to(device=device, dtype=dtype)
        ref = ref_point.to(device=device, dtype=dtype)
        normalized = torch.empty_like(ref)
        normalized[maximize] = (ref[maximize] - lower[maximize]) / denom[maximize]
        normalized[~maximize] = (upper[~maximize] - ref[~maximize]) / denom[~maximize]
        return normalized * weights

    def _to_utility_space(self, y: torch.Tensor) -> torch.Tensor:
        maximize = self._maximize_mask(device=y.device, dtype=torch.bool)
        utility = y.clone()
        utility[:, ~maximize] = -utility[:, ~maximize]
        return utility

    def initialize(self) -> NoReturn:
        self.visited_x, self.visited_y = self.problem.initial_samples()
        self.problem.total_y = self.visited_y.clone()
        vals = self.visited_y.detach().cpu().numpy().reshape(-1, self.visited_y.shape[-1])
        details = self._last_eval_details(vals.shape[0])
        for i, row in enumerate(vals):
            self._log_episode_metrics(
                phase="initial_sample",
                objective_values=row,
                detail=details[i],
            )
        self.run_logger.log_event(
            "init_completed",
            {
                "samples": int(vals.shape[0]),
            },
        )

    def set_optimizer(self) -> torch.optim.Adam:
        parameters = [
            {"params": self.model.mlp.parameters()},
            {"params": self.model.gp.covar_module.parameters()},
            {"params": self.model.gp.mean_module.parameters()},
            {"params": self.model.gp.likelihood.parameters()}
        ]
        return torch.optim.Adam(
            parameters, lr=self.problem.configs["dkl-gp"]["learning-rate"]
        )

    def fit_dkl_gp(self) -> NoReturn:
        if self.visited_x.size(0) == 0 or self.visited_y.size(0) == 0:
            self.model = None
            self.run_logger.log_event(
                "fit_skipped",
                {"reason": "no_observations"},
            )
            return
        dkl_cfg = self.problem.configs.get("dkl-gp", {})
        use_cuda = dkl_cfg.get("use-cuda", False) and torch.cuda.is_available()
        gpu_id = dkl_cfg.get("gpu-id", 0)
        device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")

        self.visited_x = self.visited_x.to(device)
        self.visited_y = self.visited_y.to(device)
        self.objective_normalizer = self._build_objective_normalizer(
            self.visited_y.detach().cpu()
        )
        normalized_y = self._normalize_objectives(self.visited_y)

        self.model = initialize_dkl_gp(
            self.visited_x,
            normalized_y,
            self.problem.configs["dkl-gp"]["mlp-output-dim"]
        )
        self.model.device = device
        self.model.set_train()
        optimizer = self.set_optimizer()

        iterator = range(self.problem.configs["dkl-gp"]["max-traininig-epoch"])
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.gp.likelihood,
            self.model.gp
        )
        y = self.model.transform_ylayout(normalized_y).squeeze(1)
        for i in iterator:
            optimizer.zero_grad()
            _y = self.model.train(self.visited_x)
            loss = -mll(_y, y)
            loss.backward()
            optimizer.step()
        # Keep feature scaling consistent between GP training inputs and
        # acquisition-time candidate evaluation.
        self.model.forward_mlp(self.visited_x, update_stats=True)
        self.model.set_eval()

    def eipv_suggest(self, batch: int = 1) -> NoReturn:
        if self.problem.x.size(0) == 0:
            self.problem.replenish_candidates()
        self.problem.x = self.problem.clip_to_bounds(self.problem.x)
        batch = min(int(batch), int(self.problem.x.size(0)))
        if batch <= 0:
            return

        if getattr(self, "model", None) is None or self.visited_y.size(0) == 0:
            indices = torch.randperm(self.problem.x.size(0))[:batch]
            new_x = self.problem.x[indices].to(torch.float32).reshape(-1, self.problem.n_dim)
        else:
            normalized_y = self._normalize_objectives(self.visited_y.to(self.model.device))
            normalized_ref = _coerce_normalized_ref_point(
                self._normalize_ref_point(
                    self.problem._ref_point,
                    dtype=normalized_y.dtype,
                    device=self.model.device,
                ),
                normalized_y,
            )
            partitioning = NondominatedPartitioning(
                ref_point=normalized_ref,
                Y=normalized_y,
            )

            acq_func = ExpectedHypervolumeImprovement(
                model=self.model.gp,
                ref_point=normalized_ref.detach().cpu().tolist(),
                partitioning=partitioning,
            ).to(self.model.device)

            acq_val = acq_func(
                self.model.forward_mlp(
                    self.problem.x.to(torch.float).to(self.model.device)
                ).unsqueeze(1).to(self.model.device)
            ).to(self.model.device)
            _, indices = torch.topk(acq_val, k=batch)
            new_x = self.problem.x[indices].to(torch.float32).reshape(-1, self.problem.n_dim)
        new_x = self.problem.clip_to_bounds(new_x)
        if self.problem.mode == "online":
            # Remove already-suggested sequences and replenish the pool to keep BO exploring.
            self.problem.remove_sampled_data(new_x)
            self.problem.replenish_candidates()
        self.visited_x = torch.cat((self.visited_x, new_x), 0)
        new_y = self.problem.evaluate_true(new_x)
        if new_y.ndimension() == 1:
            new_y = new_y.unsqueeze(0)
        self.visited_y = torch.cat((self.visited_y, new_y), 0)
        vals = new_y.detach().cpu().numpy().reshape(-1, new_y.shape[-1])
        details = self._last_eval_details(vals.shape[0])
        for offset, row in enumerate(vals):
            self._log_episode_metrics(
                phase="bo_step",
                objective_values=row,
                detail=details[offset],
            )

    def report(self):
        utility_y = self._to_utility_space(self.visited_y)
        pred = self.visited_y[is_non_dominated(utility_y)]
        info("pareto set: {}, size: {}".format(
                str(pred.detach().cpu()),
                len(pred)
            )
        )
        for d in (self.paths["results_dir"], self.paths["models_dir"], self.paths["logs_dir"]):
            mkdir(d)
        pareto_path = os.path.join(self.paths["results_dir"], "pareto-frontier.rpt")
        model_path = os.path.join(self.paths["models_dir"], "dkl-gp.mdl")

        write_txt(pareto_path, np.array(pred.detach().cpu()), fmt="%f")
        self.model.save(model_path)
        self.run_logger.log_event(
            "run_completed",
            {
                "pareto_size": int(len(pred)),
                "pareto_path": pareto_path,
                "model_path": model_path,
            },
        )
        self.run_logger.close()

    def __del__(self):
        if getattr(self, "run_logger", None) is not None:
            self.run_logger.close()


def create_solver(problem: DesignSpaceProblem) ->BOOMExplorerSolver:
    return BOOMExplorerSolver(problem)
