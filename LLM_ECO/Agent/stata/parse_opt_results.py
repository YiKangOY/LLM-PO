#!/usr/bin/env python3
"""Parse ECO optimization reports into a CSV summary."""

import csv
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from configs import design_max_iterations_per_trace
from eco_database import ReportParser
from eco_ppa_agent import load_reports_for_iteration

DESIGN_BASE_PATH = os.environ.get(
    "LLM_ECO_RUNS_DIR",
    os.path.join(PROJECT_ROOT, "data"),
)


def load_final_reports(design_dir, iteration):
    """Load reports for a specific iteration in a design directory."""
    last_reports = {}
    if iteration > 0:
        power_path = os.path.join(design_dir, "reports", f"report_power_{iteration}.txt")
        with open(power_path, "r") as power_file:
            power_content = power_file.read()
        last_reports["power"] = ReportParser.parse_power_report(power_content)
    return load_reports_for_iteration(
        iteration,
        last_command_type=None,
        last_reports=last_reports,
        last_executed_command="",
        tool_using=False,
        reports_base_path=design_dir,
    )


def format_design_name(design_name):
    """Format design name for summary output."""
    return design_name.replace("_", " ")


def main():
    design_names = list(design_max_iterations_per_trace.keys())
    output_csv = os.path.join(os.path.dirname(__file__), "optimization_results.csv")

    with open(output_csv, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "Design",
            "Area",
            "clock period ns",
            "Setup TNS (ns)",
            "Setup WNS (ns)",
            "Hold TNS(ns)",
            "Hold WNS(ns)",
            "Total Power (W)",
            "Leakage Power (W)",
            "Dynamic Power (W)",
        ])

        for design_name in design_names:
            design_dir = os.path.join(DESIGN_BASE_PATH, design_name)
            max_iteration = design_max_iterations_per_trace[design_name]
            reports = load_final_reports(design_dir, max_iteration)
            timing = reports["timing"]
            area = reports["area"]
            power = reports["power"]
            dynamic_power = power["total_power"] - power["leakage_power"]
            writer.writerow([
                format_design_name(design_name),
                area["design_area"],
                timing["clock_period"],
                timing["setup_total_negative_slack"] / 1000.0,
                timing["setup_critical_path_slack"] / 1000.0,
                timing["hold_total_negative_slack"] / 1000.0,
                timing["hold_critical_path_slack"] / 1000.0,
                power["total_power"],
                power["leakage_power"],
                dynamic_power,
            ])


if __name__ == "__main__":
    main()
