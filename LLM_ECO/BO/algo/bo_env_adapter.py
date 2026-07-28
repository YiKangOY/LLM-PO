# Author: baichen318@gmail.com

import importlib
import importlib.util
import copy
import shutil
import os
import sys
from typing import Any, Dict, Optional


def _bo_run_dir(workspace: str, env_id: int) -> str:
    return os.path.join(workspace, "BO", "run_dir", "run_{}".format(env_id))


class BOECOEnvAdapter(object):
    """
    Adapter that wires the BO ECO flow to the existing ECO environment
    implementation.

    Expected env-kwargs:
      - bo_root: path to the ECO environment package root
      - use_simulation: optional bool to toggle simulation
      - design_name: optional design name override
      - reports_loader_entrypoint: optional "path.to.module:function"
      - run_paths: optional RunPaths instance
    """

    def __init__(
        self,
        config: Dict[str, Any],
        bo_root: Optional[str] = None,
        use_simulation: Optional[bool] = None,
        design_name: Optional[str] = None,
        reports_loader_entrypoint: Optional[str] = None,
        run_paths: Optional[object] = None,
        **kwargs: Any,
    ):
        self._ensure_bo_imports(bo_root)

        from rl_config import load_config as load_bo_config

        design_name_kw = design_name or kwargs.get("design_name")
        base_path_kw = (
            kwargs.get("base_path")
            or kwargs.get("workspace")
            or config.get("env_kwargs", {}).get("base_path")
        )
        if isinstance(run_paths, dict):
            base_path_kw = (
                base_path_kw
                or run_paths.get("base_path")
                or run_paths.get("workspace")
                or run_paths.get("work_root")
            )
        base_path_kw = base_path_kw or bo_root

        overrides = self._build_bo_overrides(
            config,
            use_simulation,
            bo_root=bo_root,
            base_path=base_path_kw,
            design_name=design_name_kw,
        )
        if overrides:
            load_bo_config(config_overrides=overrides)

        reports_loader = None
        if reports_loader_entrypoint:
            reports_loader = self._load_entrypoint(reports_loader_entrypoint)

        from rl_environment import ECOEnvironment as BOBaseEnvironment
        from rl_command_executor import RunPaths

        run_paths_obj = run_paths
        if isinstance(run_paths, dict):
            env_id = run_paths.get("env_id", 0)
            workspace = (
                config.get("env_kwargs", {}).get("base_path")
                or run_paths.get("base_path")
                or run_paths.get("workspace")
                or run_paths.get("work_root")
                or config.get("env_kwargs", {}).get("bo_root")
            )
            run_dir = run_paths.get("run_dir")
            if not run_dir and workspace:
                run_dir = _bo_run_dir(workspace, int(env_id))
            run_paths_obj = RunPaths(
                workspace=workspace,
                run_dir=run_dir,
                env_id=env_id,
                reports_dir=run_paths.get("reports_dir"),
                logs_dir=run_paths.get("logs_dir"),
                scripts_dir=run_paths.get("scripts_dir"),
                session_prefix=run_paths.get("session_prefix"),
                server_host=run_paths.get("server_host"),
                server_port=run_paths.get("server_port"),
                use_pt_server=run_paths.get("use_pt_server", True),
            )
        if run_paths_obj is None:
            # Default to bo_root as workspace if nothing provided
            default_workspace = config.get("env_kwargs", {}).get("base_path") or bo_root
            default_run_dir = _bo_run_dir(default_workspace, 0)
            run_paths_obj = RunPaths(workspace=default_workspace, run_dir=default_run_dir)
        # Ensure per-env directories and baseline reports exist
        if hasattr(run_paths_obj, "prepare"):
            run_paths_obj.prepare(skip_if_exists=False)
        # Ensure expected baseline reports and session are present under run_dir
        source_reports = os.path.join(run_paths_obj.workspace, "reports")
        dest_reports = os.path.join(run_paths_obj.run_dir, "reports")
        if os.path.isdir(source_reports):
            os.makedirs(dest_reports, exist_ok=True)
            for name in ("report_qor_0.txt", "report_power_0.txt"):
                src = os.path.join(source_reports, name)
                dst = os.path.join(dest_reports, name)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copyfile(src, dst)
        # Point reports_dir to the per-run copy so BO/RL read/write in the same place.
        run_paths_obj.reports_dir = dest_reports
        source_session = os.path.join(run_paths_obj.workspace, "eco_session_0")
        dest_session = os.path.join(run_paths_obj.run_dir, "eco_session_0")
        if os.path.isdir(source_session) and not os.path.exists(dest_session):
            shutil.copytree(source_session, dest_session)

        # Wrap reports_loader to auto-seed missing iteration reports from iteration 0.
        def _reports_loader_with_fallback(**kwargs):
            iteration = kwargs.get("iteration", 0)
            reports_dir = kwargs.get("reports_dir") or run_paths_obj.reports_dir
            if kwargs.get("reports_base_path") is None:
                kwargs["reports_base_path"] = run_paths_obj.run_dir
            for stem in ("report_qor", "report_power"):
                dst = os.path.join(reports_dir, f"{stem}_{iteration}.txt")
                src = os.path.join(reports_dir, f"{stem}_0.txt")
                if iteration != 0 and not os.path.exists(dst) and os.path.exists(src):
                    shutil.copyfile(src, dst)
            reports = reports_loader(**kwargs) if reports_loader else None
            if isinstance(reports, dict):
                for key in (
                    "timing_unfix",
                    "power_unfix",
                    "area_unfix",
                    "unfix_timing_reports",
                    "unfix_power_reports",
                    "unfix_area_reports",
                ):
                    reports.pop(key, None)
            return reports

        self.env = BOBaseEnvironment(
            design_name=design_name,
            reports_loader=_reports_loader_with_fallback if reports_loader else None,
            use_simulation=use_simulation,
            run_paths=run_paths_obj,
            render_mode=None,
        )
        self.action_space = self.env.action_space

    def reset(self, *args: Any, **kwargs: Any):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

    def _ensure_bo_imports(self, bo_root: Optional[str]) -> None:
        if not bo_root:
            raise ValueError(
                "bo_root is required to import the ECO environment implementation"
            )
        if not os.path.isdir(bo_root):
            raise ValueError("bo_root does not exist: {}".format(bo_root))

        bo_root = os.path.abspath(bo_root)
        bo_parent = os.path.dirname(bo_root)
        if bo_parent not in sys.path:
            sys.path.insert(0, bo_parent)
        if bo_root not in sys.path:
            sys.path.insert(0, bo_root)
        agent_path = os.path.join(bo_parent, "Agent")
        if os.path.isdir(agent_path) and agent_path not in sys.path:
            sys.path.insert(0, agent_path)

    def _build_bo_overrides(
        self,
        config: Dict[str, Any],
        use_simulation: Optional[bool],
        bo_root: Optional[str] = None,
        base_path: Optional[str] = None,
        design_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {}
        bo_loop_overrides: Dict[str, Any] = {}
        if "max_iterations_per_episode" in config:
            bo_loop_overrides["max_iterations_per_episode"] = config["max_iterations_per_episode"]
        if bo_loop_overrides:
            overrides["rl"] = bo_loop_overrides

        if config.get("reward_weights"):
            overrides["reward_weights"] = config["reward_weights"]
        if config.get("action_spaces"):
            overrides["action_spaces"] = config["action_spaces"]
        normalization_ranges = config.get("normalization_ranges")
        if not normalization_ranges:
            normalization_ranges = self._load_rl_normalization_ranges(
                bo_root=bo_root,
                design_name=design_name,
            )
        if normalization_ranges:
            overrides["normalization_ranges"] = normalization_ranges
        if use_simulation is not None or base_path or design_name:
            env_cfg: Dict[str, Any] = {}
            if use_simulation is not None:
                env_cfg["use_real_execution"] = not use_simulation
            if base_path:
                env_cfg["base_path"] = base_path
            if design_name:
                env_cfg["design_name"] = design_name
            overrides["environment"] = env_cfg
        return overrides

    def _load_rl_normalization_ranges(
        self,
        bo_root: Optional[str],
        design_name: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if not bo_root or not design_name:
            return None
        bo_root_abs = os.path.abspath(bo_root)
        cfg_path = os.path.join(bo_root_abs, "design_configs.py")
        if not os.path.exists(cfg_path):
            return None
        spec = importlib.util.spec_from_file_location("bo_rl_design_configs", cfg_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        path_added = False
        if bo_root_abs not in sys.path:
            sys.path.insert(0, bo_root_abs)
            path_added = True
        try:
            spec.loader.exec_module(module)  # type: ignore[arg-type]
        finally:
            if path_added:
                try:
                    sys.path.remove(bo_root_abs)
                except ValueError:
                    pass
        design_overrides = getattr(module, "DESIGN_CONFIG_OVERRIDES", None)
        if not isinstance(design_overrides, dict):
            return None
        design_cfg = design_overrides.get(design_name)
        if not isinstance(design_cfg, dict):
            return None
        ranges = design_cfg.get("normalization_ranges")
        if not isinstance(ranges, dict):
            return None
        return copy.deepcopy(ranges)

    def _load_entrypoint(self, entrypoint: str):
        if ":" not in entrypoint:
            raise ValueError("reports_loader_entrypoint must be formatted as 'path.to.module:function'")
        module_path, func_name = entrypoint.split(":")
        # Avoid picking up BO's local utils when importing Agent utilities.
        if "utils" in sys.modules:
            mod = sys.modules["utils"]
            mod_file = getattr(mod, "__file__", "") or ""
            if os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "utils")) in mod_file:
                sys.modules.pop("utils")
        module = importlib.import_module(module_path)
        return getattr(module, func_name)
