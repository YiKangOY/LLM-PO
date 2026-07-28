#!/usr/bin/env python3

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class PowerMetrics:
    total_power: float = 0.0
    internal_power: float = 0.0
    switching_power: float = 0.0
    leakage_power: float = 0.0
    clock_tree_power: float = 0.0
    register_power: float = 0.0
    combinational_power: float = 0.0
    internal_power_pct: float = 0.0
    switching_power_pct: float = 0.0
    leakage_power_pct: float = 0.0
    power_groups: Dict[str, Dict[str, float]] = None

    def __post_init__(self):
        if self.power_groups is None:
            self.power_groups = {}


@dataclass
class QoRMetrics:
    setup_critical_path_slack: float = 0.0
    setup_total_negative_slack: float = 0.0
    setup_violating_paths: int = 0
    setup_levels_of_logic: int = 0
    hold_critical_path_slack: float = 0.0
    hold_total_negative_slack: float = 0.0
    hold_violating_paths: int = 0
    hold_levels_of_logic: int = 0
    total_cell_area: float = 0.0
    design_area: float = 0.0
    min_capacitance_count: int = 0
    max_transition_count: int = 0


def parse_power_report(report_content: str) -> Dict[str, Any]:
    power_data: Dict[str, Any] = {
        "design_name": None,
        "total_power": 0.0,
        "clock_tree_power": 0.0,
        "register_power": 0.0,
        "combinational_power": 0.0,
        "leakage_power": 0.0,
        "net_switching_power": 0.0,
        "cell_internal_power": 0.0,
        "power_breakdown": {},
        "raw_content": report_content,
    }

    lines = report_content.strip().split("\n")
    for line in lines:
        if line.startswith("Design :"):
            power_data["design_name"] = line.split(":", 1)[1].strip()
            break

    in_power_table = False
    for line in lines:
        line = line.strip()
        if "Internal  Switching  Leakage    Total" in line:
            in_power_table = True
            continue
        if in_power_table and line.startswith("----"):
            continue
        if in_power_table and (line == "" or line.startswith("Net Switching")):
            in_power_table = False
            continue

        if in_power_table and line and not line.startswith("Power Group"):
            parts = re.split(r"\s+", line)
            if len(parts) < 5:
                continue
            group_name = parts[0]
            try:
                total_power = float(parts[4])
            except ValueError:
                continue
            power_data["power_breakdown"][group_name] = total_power
            if group_name == "clock_tree":
                power_data["clock_tree_power"] = total_power
            elif group_name == "register":
                power_data["register_power"] = total_power
            elif group_name == "combinational":
                power_data["combinational_power"] = total_power

    for line in lines:
        line = line.strip()
        if line.startswith("Net Switching Power"):
            match = re.search(r"=\s+([\d.eE+-]+)", line)
            if match:
                power_data["net_switching_power"] = float(match.group(1))
        elif line.startswith("Cell Internal Power"):
            match = re.search(r"=\s+([\d.eE+-]+)", line)
            if match:
                power_data["cell_internal_power"] = float(match.group(1))
        elif line.startswith("Cell Leakage Power"):
            match = re.search(r"=\s+([\d.eE+-]+)", line)
            if match:
                power_data["leakage_power"] = float(match.group(1))
        elif line.startswith("Total Power"):
            match = re.search(r"=\s+([\d.eE+-]+)", line)
            if match:
                power_data["total_power"] = float(match.group(1))

    return power_data


def parse_qor_report(report_content: str) -> Dict[str, Any]:
    qor_data: Dict[str, Any] = {
        "design_name": None,
        "clock_period": 0.0,
        "setup_violating_paths": 0,
        "hold_violating_paths": 0,
        "setup_critical_path_slack": 0.0,
        "hold_critical_path_slack": 0.0,
        "setup_total_negative_slack": 0.0,
        "hold_total_negative_slack": 0.0,
        "setup_levels_of_logic": 0,
        "hold_levels_of_logic": 0,
        "total_cell_area": 0.0,
        "design_area": 0.0,
        "leaf_cell_count": 0,
        "pin_count": 0,
        "min_capacitance_count": 0,
        "max_transition_count": 0,
        "total_drc_cost": 0.0,
        "raw_content": report_content,
    }

    lines = report_content.strip().split("\n")
    for line in lines:
        if line.startswith("Design :"):
            qor_data["design_name"] = line.split(":", 1)[1].strip()
            break

    clock_period_patterns = [
        r"Clock Period\s*\(ns\)\s*:\s*([-\d.]+)",
        r"Clock Period\s*:\s*([-\d.]+)",
        r"Target Clock Period\s*:\s*([-\d.]+)",
        r"Target Period\s*:\s*([-\d.]+)",
        r"Clock Period\s*=\s*([-\d.]+)",
    ]
    for line in lines:
        for pattern in clock_period_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                qor_data["clock_period"] = float(match.group(1))
                break
        if qor_data["clock_period"] != 0.0:
            break

    current_section = None
    for line in lines:
        line = line.strip()
        if "max_delay/setup" in line:
            current_section = "setup"
            continue
        if "min_delay/hold" in line:
            current_section = "hold"
            continue
        if line == "Area":
            current_section = "area"
            continue
        if "Cell & Pin Count" in line:
            current_section = "cell_count"
            continue
        if "Design Rule Violations" in line:
            current_section = "drc"
            continue

        if current_section == "setup":
            if "Critical Path Slack:" in line:
                match = re.search(r":\s*([-\d.]+)", line)
                if match:
                    qor_data["setup_critical_path_slack"] = float(match.group(1))
            elif "Total Negative Slack:" in line:
                match = re.search(r":\s*([-\d.]+)", line)
                if match:
                    qor_data["setup_total_negative_slack"] = float(match.group(1))
            elif "No. of Violating Paths:" in line:
                match = re.search(r":\s*(\d+)", line)
                if match:
                    qor_data["setup_violating_paths"] = int(match.group(1))
            elif "Levels of Logic:" in line:
                match = re.search(r":\s*(\d+)", line)
                if match:
                    qor_data["setup_levels_of_logic"] = int(match.group(1))

        elif current_section == "hold":
            if "Critical Path Slack:" in line:
                match = re.search(r":\s*([-\d.]+)", line)
                if match:
                    qor_data["hold_critical_path_slack"] = float(match.group(1))
            elif "Total Negative Slack:" in line:
                match = re.search(r":\s*([-\d.]+)", line)
                if match:
                    qor_data["hold_total_negative_slack"] = float(match.group(1))
            elif "No. of Violating Paths:" in line:
                match = re.search(r":\s*(\d+)", line)
                if match:
                    qor_data["hold_violating_paths"] = int(match.group(1))
            elif "Levels of Logic:" in line:
                match = re.search(r":\s*(\d+)", line)
                if match:
                    qor_data["hold_levels_of_logic"] = int(match.group(1))

        elif current_section == "area":
            if "Total cell area:" in line:
                match = re.search(r":\s*([\d.]+)", line)
                if match:
                    qor_data["total_cell_area"] = float(match.group(1))
            elif "Design Area:" in line:
                match = re.search(r":\s*([\d.]+)", line)
                if match:
                    qor_data["design_area"] = float(match.group(1))

        elif current_section == "cell_count":
            if "Pin Count:" in line:
                match = re.search(r":\s*(\d+)", line)
                if match:
                    qor_data["pin_count"] = int(match.group(1))
            elif "Leaf Cell Count:" in line:
                match = re.search(r":\s*(\d+)", line)
                if match:
                    qor_data["leaf_cell_count"] = int(match.group(1))

        elif current_section == "drc":
            if "min_capacitance Count:" in line:
                match = re.search(r":\s*(\d+)", line)
                if match:
                    qor_data["min_capacitance_count"] = int(match.group(1))
            elif "max_transition Count:" in line:
                match = re.search(r":\s*(\d+)", line)
                if match:
                    qor_data["max_transition_count"] = int(match.group(1))
            elif "Total DRC Cost:" in line:
                match = re.search(r":\s*([\d.]+)", line)
                if match:
                    qor_data["total_drc_cost"] = float(match.group(1))

    return qor_data


def parse_timing_log(log_content: str) -> Dict[str, Any]:
    timing_data = {
        "fix_type": "timing",
        "elapsed_time_seconds": 0,
        "gate_sizing_commands": 0,
        "total_commands": 0,
        "area_increased": 0.0,
        "total_violating_endpoints_found": 0,
        "total_violating_endpoints_fixed": 0,
        "total_violating_endpoints_remaining": 0,
        "percentage_violations_fixed": 0.0,
        "raw_content": log_content,
    }

    lines = log_content.strip().split("\n")
    for line in lines:
        if "Information: Elapsed time" in line:
            match = re.search(r"\[\s*(\d+)\s+seconds\s*\]", line)
            if match:
                timing_data["elapsed_time_seconds"] = int(match.group(1))
                break

    in_final_summary = False
    for line in lines:
        line = line.strip()
        if "Final ECO Summary:" in line:
            in_final_summary = True
            continue
        if in_final_summary and line.startswith("Number of gate_sizing commands"):
            match = re.search(r"(\d+)$", line)
            if match:
                timing_data["gate_sizing_commands"] = int(match.group(1))
        elif in_final_summary and line.startswith("Total number of commands"):
            match = re.search(r"(\d+)$", line)
            if match:
                timing_data["total_commands"] = int(match.group(1))
        elif in_final_summary and line.startswith("Area increased by cell sizing"):
            match = re.search(r"([\d.]+)$", line)
            if match:
                timing_data["area_increased"] = float(match.group(1))
        elif in_final_summary and line.startswith("Fixing Summary:"):
            in_final_summary = False
            break

    in_fixing_summary = False
    for line in lines:
        line = line.strip()
        if "Fixing Summary:" in line:
            in_fixing_summary = True
            continue
        if in_fixing_summary and line.startswith("Total violating endpoints found"):
            match = re.search(r"(\d+)$", line)
            if match:
                timing_data["total_violating_endpoints_found"] = int(match.group(1))
        elif in_fixing_summary and line.startswith("Total violating endpoints fixed"):
            match = re.search(r"(\d+)$", line)
            if match:
                timing_data["total_violating_endpoints_fixed"] = int(match.group(1))
        elif in_fixing_summary and line.startswith("Total violating endpoints remaining"):
            match = re.search(r"(\d+)$", line)
            if match:
                timing_data["total_violating_endpoints_remaining"] = int(match.group(1))
        elif in_fixing_summary and line.startswith("Total percentage of violations fixed"):
            match = re.search(r"([\d.]+)%", line)
            if match:
                timing_data["percentage_violations_fixed"] = float(match.group(1))
        elif in_fixing_summary and "Information: Elapsed time" in line:
            break

    return timing_data


def parse_power_log(log_content: str) -> Dict[str, Any]:
    power_data = {
        "fix_type": "power",
        "report_format": None,
        "elapsed_time_seconds": 0,
        "initial_total_area": 0.0,
        "final_total_area": 0.0,
        "total_area_decreased": 0.0,
        "percentage_area_decreased": 0.0,
        "percentage_datapath_area_decreased": 0.0,
        "buffers_removed": 0,
        "percentage_buffers_removed": 0.0,
        "percentage_cells_removed": 0.0,
        "raw_content": log_content,
    }
    lines = log_content.strip().split("\n")

    for line in lines:
        if "Information: Elapsed time" in line:
            match = re.search(r"\[\s*(\d+)\s+seconds\s*\]", line)
            if match:
                power_data["elapsed_time_seconds"] = int(match.group(1))
                break

    has_final_eco_summary = any("Final ECO Summary:" in line for line in lines)
    has_fixing_summary = any("Fixing Summary:" in line for line in lines)

    if has_final_eco_summary:
        power_data["report_format"] = "eco_summary"
        in_summary = False
        for line in lines:
            line = line.strip()
            if "Final ECO Summary:" in line:
                in_summary = True
                continue
            if in_summary and line.startswith("Initial total cell area"):
                match = re.search(r"([\d.]+)$", line)
                if match:
                    power_data["initial_total_area"] = float(match.group(1))
            elif in_summary and line.startswith("Final total cell area"):
                match = re.search(r"([\d.]+)$", line)
                if match:
                    power_data["final_total_area"] = float(match.group(1))
            elif in_summary and line.startswith("Total cell area decreased"):
                match = re.search(r"([\d.]+)$", line)
                if match:
                    power_data["total_area_decreased"] = float(match.group(1))
            elif in_summary and line.startswith("Percentage of total cell area decreased"):
                match = re.search(r"([\d.]+)%", line)
                if match:
                    power_data["percentage_area_decreased"] = float(match.group(1))
            elif in_summary and line.startswith("Percentage of datapath cell area decreased"):
                match = re.search(r"([\d.]+)%", line)
                if match:
                    power_data["percentage_datapath_area_decreased"] = float(match.group(1))
            elif in_summary and "Information: Elapsed time" in line:
                break
    elif has_fixing_summary:
        power_data["report_format"] = "fixing_summary"
        in_fixing_summary = False
        for line in lines:
            line = line.strip()
            if "Fixing Summary:" in line:
                in_fixing_summary = True
                continue
            if in_fixing_summary and line.startswith("Total number of buffers removed"):
                match = re.search(r"(\d+)$", line)
                if match:
                    power_data["buffers_removed"] = int(match.group(1))
            elif in_fixing_summary and line.startswith("Percentage of buffers removed"):
                match = re.search(r"([\d.]+)%", line)
                if match:
                    power_data["percentage_buffers_removed"] = float(match.group(1))
            elif in_fixing_summary and line.startswith("Percentage of cells removed"):
                match = re.search(r"([\d.]+)%", line)
                if match:
                    power_data["percentage_cells_removed"] = float(match.group(1))
            elif in_fixing_summary and line.startswith("Information:"):
                break

    return power_data


def parse_area_log(log_content: str) -> Dict[str, Any]:
    area_data = {
        "fix_type": "area",
        "report_format": None,
        "elapsed_time_seconds": 0,
        "initial_total_area": 0.0,
        "final_total_area": 0.0,
        "total_area_decreased": 0.0,
        "percentage_area_decreased": 0.0,
        "percentage_datapath_area_decreased": 0.0,
        "buffers_removed": 0,
        "percentage_buffers_removed": 0.0,
        "percentage_cells_removed": 0.0,
        "raw_content": log_content,
    }
    lines = log_content.strip().split("\n")

    for line in lines:
        if "Information: Elapsed time" in line:
            match = re.search(r"\[\s*(\d+)\s+seconds\s*\]", line)
            if match:
                area_data["elapsed_time_seconds"] = int(match.group(1))
                break

    has_final_eco_summary = any("Final ECO Summary:" in line for line in lines)
    has_fixing_summary = any("Fixing Summary:" in line for line in lines)

    if has_final_eco_summary:
        area_data["report_format"] = "eco_summary"
        in_summary = False
        for line in lines:
            line = line.strip()
            if "Final ECO Summary:" in line:
                in_summary = True
                continue
            if in_summary and line.startswith("Initial total cell area"):
                match = re.search(r"([\d.]+)$", line)
                if match:
                    area_data["initial_total_area"] = float(match.group(1))
            elif in_summary and line.startswith("Final total cell area"):
                match = re.search(r"([\d.]+)$", line)
                if match:
                    area_data["final_total_area"] = float(match.group(1))
            elif in_summary and line.startswith("Total cell area decreased"):
                match = re.search(r"([\d.]+)$", line)
                if match:
                    area_data["total_area_decreased"] = float(match.group(1))
            elif in_summary and line.startswith("Percentage of total cell area decreased"):
                match = re.search(r"([\d.]+)%", line)
                if match:
                    area_data["percentage_area_decreased"] = float(match.group(1))
            elif in_summary and line.startswith("Percentage of datapath cell area decreased"):
                match = re.search(r"([\d.]+)%", line)
                if match:
                    area_data["percentage_datapath_area_decreased"] = float(match.group(1))
            elif in_summary and "Information: Elapsed time" in line:
                break
    elif has_fixing_summary:
        area_data["report_format"] = "fixing_summary"
        in_fixing_summary = False
        for line in lines:
            line = line.strip()
            if "Fixing Summary:" in line:
                in_fixing_summary = True
                continue
            if in_fixing_summary and line.startswith("Total number of buffers removed"):
                match = re.search(r"(\d+)$", line)
                if match:
                    area_data["buffers_removed"] = int(match.group(1))
            elif in_fixing_summary and line.startswith("Percentage of buffers removed"):
                match = re.search(r"([\d.]+)%", line)
                if match:
                    area_data["percentage_buffers_removed"] = float(match.group(1))
            elif in_fixing_summary and line.startswith("Percentage of cells removed"):
                match = re.search(r"([\d.]+)%", line)
                if match:
                    area_data["percentage_cells_removed"] = float(match.group(1))
            elif in_fixing_summary and line.startswith("Information:"):
                break

    return area_data


def build_qor_metrics(qor_data: Dict[str, Any]) -> QoRMetrics:
    return QoRMetrics(
        setup_critical_path_slack=float(qor_data.get("setup_critical_path_slack", 0.0)),
        setup_total_negative_slack=float(qor_data.get("setup_total_negative_slack", 0.0)),
        setup_violating_paths=int(qor_data.get("setup_violating_paths", 0)),
        setup_levels_of_logic=int(qor_data.get("setup_levels_of_logic", 0)),
        hold_critical_path_slack=float(qor_data.get("hold_critical_path_slack", 0.0)),
        hold_total_negative_slack=float(qor_data.get("hold_total_negative_slack", 0.0)),
        hold_violating_paths=int(qor_data.get("hold_violating_paths", 0)),
        hold_levels_of_logic=int(qor_data.get("hold_levels_of_logic", 0)),
        total_cell_area=float(qor_data.get("total_cell_area", 0.0)),
        design_area=float(qor_data.get("design_area", 0.0)),
        min_capacitance_count=int(qor_data.get("min_capacitance_count", 0)),
        max_transition_count=int(qor_data.get("max_transition_count", 0)),
    )


def build_power_metrics(power_data: Dict[str, Any]) -> PowerMetrics:
    total = float(power_data.get("total_power", 0.0))
    internal = float(power_data.get("cell_internal_power", 0.0))
    switching = float(power_data.get("net_switching_power", 0.0))
    leakage = float(power_data.get("leakage_power", 0.0))

    if total > 0:
        internal_pct = (internal / total) * 100.0
        switching_pct = (switching / total) * 100.0
        leakage_pct = (leakage / total) * 100.0
    else:
        internal_pct = 0.0
        switching_pct = 0.0
        leakage_pct = 0.0

    return PowerMetrics(
        total_power=total,
        internal_power=internal,
        switching_power=switching,
        leakage_power=leakage,
        clock_tree_power=float(power_data.get("clock_tree_power", 0.0)),
        register_power=float(power_data.get("register_power", 0.0)),
        combinational_power=float(power_data.get("combinational_power", 0.0)),
        internal_power_pct=internal_pct,
        switching_power_pct=switching_pct,
        leakage_power_pct=leakage_pct,
        power_groups={
            key: {"total": float(val)}
            for key, val in (power_data.get("power_breakdown", {}) or {}).items()
        },
    )


def parse_qor_report_file(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, "r") as handle:
            content = handle.read()
        qor_data = parse_qor_report(content)
        metrics = build_qor_metrics(qor_data)
        return {
            "parsing_successful": True,
            "parsing_errors": [],
            "metrics": asdict(metrics),
        }
    except Exception as exc:
        return {
            "parsing_successful": False,
            "parsing_errors": [str(exc)],
            "metrics": {},
        }


def parse_power_report_file(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, "r") as handle:
            content = handle.read()
        power_data = parse_power_report(content)
        metrics = build_power_metrics(power_data)
        return {
            "parsing_successful": True,
            "parsing_errors": [],
            "metrics": asdict(metrics),
        }
    except Exception as exc:
        return {
            "parsing_successful": False,
            "parsing_errors": [str(exc)],
            "metrics": {},
        }


def parse_elapsed_time(report_content: str) -> float:
    if not report_content:
        return 0.0
    elapsed_match = re.search(r"Elapsed time\s*\[\s*(\d+)\s+seconds\s*\]", report_content)
    if not elapsed_match:
        return 0.0
    return float(elapsed_match.group(1))
