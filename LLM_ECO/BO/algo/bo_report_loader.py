#!/usr/bin/env python3

import copy
import os
from typing import Any, Dict, Optional

from bo_report_parser import (
    build_power_metrics,
    build_qor_metrics,
    parse_area_log,
    parse_power_log,
    parse_power_report,
    parse_timing_log,
    parse_qor_report,
)

_RUN_CONTEXT: Dict[str, Any] = {
    "base_path": None,
    "episode_index": 0,
    "last_iteration": None,
}


def _read_report(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError("Report file not found: {}".format(path))
    with open(path, "r") as handle:
        return handle.read()


def load_reports_for_iteration(
    iteration: int,
    last_command_type: Any = None,
    last_reports: Optional[Dict[str, Any]] = None,
    last_executed_command: str = "",
    tool_using: bool = False,
    reports_base_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    BO-local report loader used by BO env adapter.

    Notes:
      - Keeps the same callable signature as Agent.eco_ppa_agent loader.
      - Intentionally does not load or pack any unfix_* reports.
      - Returns structured qor_metrics/power_metrics for BO state encoding.
    """
    del last_command_type  # kept for signature compatibility
    del tool_using         # kept for signature compatibility

    last_reports = last_reports or {}
    path_base = reports_base_path or os.getcwd()
    reports_root = os.path.join(path_base, "reports")
    if _RUN_CONTEXT["base_path"] != path_base:
        _RUN_CONTEXT["base_path"] = path_base
        _RUN_CONTEXT["episode_index"] = 0
        _RUN_CONTEXT["last_iteration"] = None

    if iteration == 0 and _RUN_CONTEXT["last_iteration"] not in (None, 0):
        _RUN_CONTEXT["episode_index"] += 1
    _RUN_CONTEXT["last_iteration"] = iteration

    # Episode-0 warmup: do not load step reports from run logs that do not exist yet.
    # Keep the state from the previous step inside this first episode.
    if _RUN_CONTEXT["episode_index"] == 0 and iteration > 0:
        cached = copy.deepcopy(last_reports) if isinstance(last_reports, dict) else {}
        for key in (
            "timing_fix",
            "area_fix",
            "power_fix",
            "timing_unfix",
            "area_unfix",
            "power_unfix",
        ):
            cached.pop(key, None)
        return cached

    qor_path = os.path.join(reports_root, "report_qor_{}.txt".format(iteration))
    power_path = os.path.join(reports_root, "report_power_{}.txt".format(iteration))

    qor_content = _read_report(qor_path)
    power_content = _read_report(power_path)

    qor_data = parse_qor_report(qor_content)
    power_data = parse_power_report(power_content)
    qor_metrics = build_qor_metrics(qor_data)
    power_metrics = build_power_metrics(power_data)

    parsed_reports: Dict[str, Any] = {
        "timing": {
            "setup_violating_paths": qor_data["setup_violating_paths"],
            "hold_violating_paths": qor_data["hold_violating_paths"],
            "setup_critical_path_slack": qor_data["setup_critical_path_slack"],
            "hold_critical_path_slack": qor_data["hold_critical_path_slack"],
            "setup_total_negative_slack": qor_data["setup_total_negative_slack"],
            "hold_total_negative_slack": qor_data["hold_total_negative_slack"],
            "clock_period": qor_data["clock_period"],
        },
        "area": {
            "design_area": qor_data["design_area"],
        },
        "power": power_data,
        "qor_metrics": qor_metrics,
        "power_metrics": power_metrics,
    }

    if iteration <= 0:
        return parsed_reports

    prev_iteration = iteration - 1
    command_lower = (last_executed_command or "").lower()

    if "opt_timing" in command_lower:
        timing_fix_path = os.path.join(reports_root, "fix_timing_{}.txt".format(prev_iteration))
        if os.path.exists(timing_fix_path):
            parsed_reports["timing_fix"] = parse_timing_log(_read_report(timing_fix_path))
    elif "opt_area" in command_lower:
        area_fix_path = os.path.join(reports_root, "fix_area_{}.txt".format(prev_iteration))
        if os.path.exists(area_fix_path):
            parsed_reports["area_fix"] = parse_area_log(_read_report(area_fix_path))
    elif "opt_power" in command_lower:
        power_fix_path = os.path.join(reports_root, "fix_power_{}.txt".format(prev_iteration))
        if os.path.exists(power_fix_path):
            power_fix = parse_power_log(_read_report(power_fix_path))
            last_power_data = copy.deepcopy(last_reports.get("power", {}))
            last_total = float(last_power_data.get("total_power", 0.0))
            last_leakage = float(last_power_data.get("leakage_power", 0.0))
            curr_total = float(parsed_reports["power"].get("total_power", 0.0))
            curr_leakage = float(parsed_reports["power"].get("leakage_power", 0.0))
            parsed_reports["power_fix"] = {
                "fix_type": "power",
                "report_format": power_fix.get("report_format"),
                "elapsed_time_seconds": power_fix.get("elapsed_time_seconds", 0),
                "total_power_decreased": round(last_total - curr_total, 4),
                "leakage_power_decreased": round(last_leakage - curr_leakage, 4),
                "dynamic_power_decreased": round((last_total - last_leakage) - (curr_total - curr_leakage), 4),
                "total_area_decreased": power_fix.get("total_area_decreased", 0.0),
                "percentage_area_decreased": power_fix.get("percentage_area_decreased", 0.0),
            }
            buffer_removal_used = "buffer_removal" in command_lower
            if buffer_removal_used and power_fix.get("report_format") == "fixing_summary":
                parsed_reports["power_fix"]["buffers_removed"] = power_fix.get("buffers_removed", 0)
                parsed_reports["power_fix"]["percentage_buffers_removed"] = power_fix.get("percentage_buffers_removed", 0.0)
                parsed_reports["power_fix"]["percentage_cells_removed"] = power_fix.get("percentage_cells_removed", 0.0)

    return parsed_reports
