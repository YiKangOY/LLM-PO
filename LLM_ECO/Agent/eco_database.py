#!/usr/bin/env python3
"""
ECO Database - Simple storage system for ECO optimization iterations
Stores iteration data with extensible structure
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class IterationData:
    """Complete iteration data structure"""
    iteration: int
    timestamp: str
    
    # Always present: QoR and Power reports
    qor_report: Dict[str, Any]
    power_report: Dict[str, Any]
    
    # Optional: ECO execution data (empty for iteration 0)
    last_eco_option: Optional[str] = None  # "drc", "timing", "power", or None
    fixing_report: Optional[Dict[str, Any]] = None
    unfixable_reasons: Optional[List[str]] = None  # None for power ECO
    
    # Extensible: Additional metrics
    metrics: Optional[Dict[str, Any]] = None


class ReportParser:
    """Parser for QoR and Power reports"""
    
    @staticmethod
    def parse_power_report(report_content: str) -> Dict[str, Any]:
        """Parse power report and extract key metrics"""
        power_data = {
            "design_name": None,
            "total_power": 0.0,
            "clock_tree_power": 0.0,
            "register_power": 0.0,
            "combinational_power": 0.0,
            "leakage_power": 0.0,
            "net_switching_power": 0.0,
            "cell_internal_power": 0.0,
            "power_breakdown": {},
            "raw_content": report_content
        }
        
        lines = report_content.strip().split('\n')
        
        # Extract design name
        for line in lines:
            if line.startswith("Design :"):
                power_data["design_name"] = line.split(":")[1].strip()
                break
        scale = 1  # Keep power values in W (not converting to mW)
        # Parse power breakdown table
        in_power_table = False
        for line in lines:
            line = line.strip()
            
            if "Internal  Switching  Leakage    Total" in line:
                in_power_table = True
                continue
            elif in_power_table and line.startswith("----"):
                continue
            elif in_power_table and (line == "" or line.startswith("Net Switching")):
                in_power_table = False
                continue
            
            if in_power_table and line and not line.startswith("Power Group"):
                parts = re.split(r'\s+', line)
                if len(parts) >= 5:
                    group_name = parts[0]
                    try:
                        total_power = float(parts[4])  * scale  # Convert to mW if needed
                        power_data["power_breakdown"][group_name] = total_power
                        
                        if group_name == "clock_tree":
                            power_data["clock_tree_power"] = total_power
                        elif group_name == "register":
                            power_data["register_power"] = total_power
                        elif group_name == "combinational":
                            power_data["combinational_power"] = total_power
                            
                    except ValueError:
                        continue
        
        # Extract summary values
        for line in lines:
            line = line.strip()
            if line.startswith("Net Switching Power"):
                match = re.search(r'=\s+([\d.e-]+)', line)
                if match:
                    power_data["net_switching_power"] = float(match.group(1)) * scale
            elif line.startswith("Cell Internal Power"):
                match = re.search(r'=\s+([\d.e-]+)', line)
                if match:
                    power_data["cell_internal_power"] = float(match.group(1)) * scale
            elif line.startswith("Cell Leakage Power"):
                match = re.search(r'=\s+([\d.e-]+)', line)
                if match:
                    power_data["leakage_power"] = float(match.group(1)) * scale
            elif line.startswith("Total Power"):
                match = re.search(r'=\s+([\d.e-]+)', line)
                if match:
                    power_data["total_power"] = float(match.group(1)) * scale
        
        return power_data
    
    @staticmethod
    def parse_qor_report(report_content: str) -> Dict[str, Any]:
        """Parse QoR report and extract key metrics"""
        qor_data = {
            "design_name": None,
            "clock_period": 0.0,
            "setup_violating_paths": 0,
            "hold_violating_paths": 0,
            "setup_critical_path_slack": 0.0,
            "hold_critical_path_slack": 0.0,
            "setup_total_negative_slack": 0.0,
            "hold_total_negative_slack": 0.0,
            "total_cell_area": 0.0,
            "design_area": 0.0,
            "leaf_cell_count": 0,
            "pin_count": 0,
            "min_capacitance_count": 0,
            "max_transition_count": 0,
            "total_drc_cost": 0.0,
            "raw_content": report_content
        }
        
        lines = report_content.strip().split('\n')
        
        # Extract design name
        for line in lines:
            if line.startswith("Design :"):
                qor_data["design_name"] = line.split(":")[1].strip()
                break

        clock_period_patterns = [
            r'Clock Period\s*\(ns\)\s*:\s*([-\d.]+)',
            r'Clock Period\s*:\s*([-\d.]+)',
            r'Target Clock Period\s*:\s*([-\d.]+)',
            r'Target Period\s*:\s*([-\d.]+)',
            r'Clock Period\s*=\s*([-\d.]+)'
        ]
        for line in lines:
            for pattern in clock_period_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    qor_data["clock_period"] = float(match.group(1))
                    break
            if qor_data["clock_period"] != 0.0:
                break
        
        # Parse sections
        # Timing sections may appear twice; later values overwrite earlier ones so the second group is used.
        current_section = None
        for line in lines:
            line = line.strip()
            
            # Detect sections
            if "max_delay/setup" in line:
                current_section = "setup"
                continue
            elif "min_delay/hold" in line:
                current_section = "hold"
                continue
            elif line.strip() == "Area":
                current_section = "area"
                continue
            elif "Cell & Pin Count" in line:
                current_section = "cell_count"
                continue
            elif "Design Rule Violations" in line:
                current_section = "drc"
                continue
            
            # Parse setup timing
            if current_section == "setup":
                if "Critical Path Slack:" in line:
                    match = re.search(r':\s*([-\d.]+)', line)
                    if match:
                        qor_data["setup_critical_path_slack"] = float(match.group(1))
                elif "Total Negative Slack:" in line:
                    match = re.search(r':\s*([-\d.]+)', line)
                    if match:
                        qor_data["setup_total_negative_slack"] = float(match.group(1))
                elif "No. of Violating Paths:" in line:
                    match = re.search(r':\s*(\d+)', line)
                    if match:
                        qor_data["setup_violating_paths"] = int(match.group(1))
            
            # Parse hold timing
            elif current_section == "hold":
                if "Critical Path Slack:" in line:
                    match = re.search(r':\s*([-\d.]+)', line)
                    if match:
                        qor_data["hold_critical_path_slack"] = float(match.group(1))
                elif "Total Negative Slack:" in line:
                    match = re.search(r':\s*([-\d.]+)', line)
                    if match:
                        qor_data["hold_total_negative_slack"] = float(match.group(1))
                elif "No. of Violating Paths:" in line:
                    match = re.search(r':\s*(\d+)', line)
                    if match:
                        qor_data["hold_violating_paths"] = int(match.group(1))
            
            # Parse area
            elif current_section == "area":
                if "Total cell area:" in line:
                    match = re.search(r':\s*([\d.]+)', line)
                    if match:
                        qor_data["total_cell_area"] = float(match.group(1))
                elif "Design Area:" in line:
                    match = re.search(r':\s*([\d.]+)', line)
                    if match:
                        qor_data["design_area"] = float(match.group(1))
            
            # Parse cell count
            elif current_section == "cell_count":
                if "Pin Count:" in line:
                    match = re.search(r':\s*(\d+)', line)
                    if match:
                        qor_data["pin_count"] = int(match.group(1))
                elif "Leaf Cell Count:" in line:
                    match = re.search(r':\s*(\d+)', line)
                    if match:
                        qor_data["leaf_cell_count"] = int(match.group(1))
            
            # Parse DRC violations
            elif current_section == "drc":
                if "min_capacitance Count:" in line:
                    match = re.search(r':\s*(\d+)', line)
                    if match:
                        qor_data["min_capacitance_count"] = int(match.group(1))
                elif "max_transition Count:" in line:
                    match = re.search(r':\s*(\d+)', line)
                    if match:
                        qor_data["max_transition_count"] = int(match.group(1))
                elif "Total DRC Cost:" in line:
                    match = re.search(r':\s*([\d.]+)', line)
                    if match:
                        qor_data["total_drc_cost"] = float(match.group(1))
        
        return qor_data
    
    @staticmethod
    def parse_drc_log(log_content: str) -> Dict[str, Any]:
        """Parse DRC fixing log and extract key metrics"""
        drc_data = {
            'fix_type': 'drc',
            "detected_violations": 0,
            "remaining_violations": 0,
            "elapsed_time_seconds": 0,
            "gate_sizing_commands": 0,
            "total_commands": 0,
            "area_increased": 0.0,
            "raw_content": log_content
        }
        
        lines = log_content.strip().split('\n')
        
        # Extract detected violations (first occurrence)
        for line in lines:
            if "Information: Detected" in line and "violations" in line:
                match = re.search(r'Detected (\d+)', line)
                if match and drc_data["detected_violations"] == 0:
                    drc_data["detected_violations"] = int(match.group(1))
                    break
        
        # Extract remaining violations
        for line in lines:
            if "Total remaining violations" in line:
                match = re.search(r'violations\s+(\d+)', line)
                if match:
                    drc_data["remaining_violations"] = int(match.group(1))
                    break
        
        # Extract elapsed time
        for line in lines:
            if "Information: Elapsed time" in line:
                match = re.search(r'\[\s*(\d+)\s+seconds\s*\]', line)
                if match:
                    drc_data["elapsed_time_seconds"] = int(match.group(1))
                    break
        
        # Extract Final ECO Summary data
        in_summary = False
        for line in lines:
            line = line.strip()
            
            if "Final ECO Summary:" in line:
                in_summary = True
                continue
            elif in_summary and line.startswith("Number of gate_sizing commands"):
                match = re.search(r'(\d+)$', line)
                if match:
                    drc_data["gate_sizing_commands"] = int(match.group(1))
            elif in_summary and line.startswith("Total number of commands"):
                match = re.search(r'(\d+)$', line)
                if match:
                    drc_data["total_commands"] = int(match.group(1))
            elif in_summary and line.startswith("Area increased by cell sizing"):
                match = re.search(r'([\d.]+)$', line)
                if match:
                    drc_data["area_increased"] = float(match.group(1))
            elif in_summary and "Information: Elapsed time" in line:
                break
        
        return drc_data
    
    @staticmethod
    def parse_drc_unfixing(log_content: str) -> Dict[str, Any]:
        """Parse DRC unfixable violations log and group by reasons"""
        unfixing_data = {
            "fix_type": "drc",
            "reason_definitions": {},
            "violations_by_reason": {},
            "total_unfixable_violations": 0,
            "raw_content": log_content
        }
        
        lines = log_content.strip().split('\n')
        
        # Parse reason definitions
        in_definitions = False
        for line in lines:
            line = line.strip()
            
            if line == "Unfixable violations:":
                in_definitions = True
                continue
            elif in_definitions and line.startswith("Violation"):
                in_definitions = False
                break
            elif in_definitions and line and not line.startswith("Unfixable"):
                # Parse reason definition: "  A - There are available lib cells outside area limit"
                match = re.match(r'^([A-Z])\s*-\s*(.+)$', line)
                if match:
                    reason_code = match.group(1)
                    reason_description = match.group(2)
                    unfixing_data["reason_definitions"][reason_code] = reason_description
                    unfixing_data["violations_by_reason"][reason_code] = []
        
        # Parse violations table
        in_violations = False
        for line in lines:
            line = line.strip()
            
            if line.startswith("Violation") and "Reasons" in line:
                in_violations = True
                continue
            elif line.startswith("---"):
                continue
            elif in_violations and line:
                # Parse violation line: "fdct_zigzag/dct_mod/.../result_reg[21]/E  P"
                parts = line.rsplit(None, 1)  # Split from right, max 1 split
                if len(parts) == 2:
                    violation_path = parts[0].strip()
                    reasons = parts[1].strip()
                    
                    # Group by each reason code
                    for reason_code in reasons:
                        if reason_code in unfixing_data["violations_by_reason"]:
                            unfixing_data["violations_by_reason"][reason_code].append(violation_path)
                            unfixing_data["total_unfixable_violations"] += 1
        
        # Calculate counts per reason
        for reason_code in unfixing_data["violations_by_reason"]:
            count = len(unfixing_data["violations_by_reason"][reason_code])
            unfixing_data["violations_by_reason"][reason_code] = {
                "count": count,
                "violations": unfixing_data["violations_by_reason"][reason_code]
            }
        
        return unfixing_data
    
    @staticmethod
    def parse_timing_log(log_content: str) -> Dict[str, Any]:
        """Parse timing fixing log and extract key metrics"""
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
            "raw_content": log_content
        }
        
        lines = log_content.strip().split('\n')
        
        # Extract elapsed time
        for line in lines:
            if "Information: Elapsed time" in line:
                match = re.search(r'\[\s*(\d+)\s+seconds\s*\]', line)
                if match:
                    timing_data["elapsed_time_seconds"] = int(match.group(1))
                    break
        
        # Extract Final ECO Summary data
        in_final_summary = False
        for line in lines:
            line = line.strip()
            
            if "Final ECO Summary:" in line:
                in_final_summary = True
                continue
            elif in_final_summary and line.startswith("Number of gate_sizing commands"):
                match = re.search(r'(\d+)$', line)
                if match:
                    timing_data["gate_sizing_commands"] = int(match.group(1))
            elif in_final_summary and line.startswith("Total number of commands"):
                match = re.search(r'(\d+)$', line)
                if match:
                    timing_data["total_commands"] = int(match.group(1))
            elif in_final_summary and line.startswith("Area increased by cell sizing"):
                match = re.search(r'([\d.]+)$', line)
                if match:
                    timing_data["area_increased"] = float(match.group(1))
            elif in_final_summary and line.startswith("Fixing Summary:"):
                in_final_summary = False
                break
        
        # Extract Fixing Summary data
        in_fixing_summary = False
        for line in lines:
            line = line.strip()
            
            if "Fixing Summary:" in line:
                in_fixing_summary = True
                continue
            elif in_fixing_summary and line.startswith("Total violating endpoints found"):
                match = re.search(r'(\d+)$', line)
                if match:
                    timing_data["total_violating_endpoints_found"] = int(match.group(1))
            elif in_fixing_summary and line.startswith("Total violating endpoints fixed"):
                match = re.search(r'(\d+)$', line)
                if match:
                    timing_data["total_violating_endpoints_fixed"] = int(match.group(1))
            elif in_fixing_summary and line.startswith("Total violating endpoints remaining"):
                match = re.search(r'(\d+)$', line)
                if match:
                    timing_data["total_violating_endpoints_remaining"] = int(match.group(1))
            elif in_fixing_summary and line.startswith("Total percentage of violations fixed"):
                match = re.search(r'([\d.]+)%', line)
                if match:
                    timing_data["percentage_violations_fixed"] = float(match.group(1))
            elif in_fixing_summary and "Information: Elapsed time" in line:
                break
        
        return timing_data
    
    @staticmethod
    def parse_power_log(log_content: str) -> Dict[str, Any]:
        """Parse power fixing log and extract key metrics"""
        power_data = {
            "fix_type": "power",
            "report_format": None,  # Will be set to "eco_summary" or "fixing_summary"
            "elapsed_time_seconds": 0,
            "initial_total_area": 0.0,
            "final_total_area": 0.0,
            "total_area_decreased": 0.0,
            "percentage_area_decreased": 0.0,
            "percentage_datapath_area_decreased": 0.0,
            "buffers_removed": 0,
            "percentage_buffers_removed": 0.0,
            "percentage_cells_removed": 0.0,
            "raw_content": log_content
        }

        lines = log_content.strip().split('\n')

        # Extract elapsed time
        for line in lines:
            if "Information: Elapsed time" in line:
                match = re.search(r'\[\s*(\d+)\s+seconds\s*\]', line)
                if match:
                    power_data["elapsed_time_seconds"] = int(match.group(1))
                    break

        # Check which summary section exists (mutually exclusive)
        has_final_eco_summary = any("Final ECO Summary:" in line for line in lines)
        has_fixing_summary = any("Fixing Summary:" in line for line in lines)

        if has_final_eco_summary:
            power_data["report_format"] = "eco_summary"
            # Extract Final ECO Summary data
            in_summary = False
            for line in lines:
                line = line.strip()

                if "Final ECO Summary:" in line:
                    in_summary = True
                    continue
                elif in_summary and line.startswith("Number of gate_sizing commands"):
                    match = re.search(r'(\d+)$', line)
                    if match:
                        power_data["gate_sizing_commands"] = int(match.group(1))
                elif in_summary and line.startswith("Initial total cell area"):
                    match = re.search(r'([\d.]+)$', line)
                    if match:
                        power_data["initial_total_area"] = float(match.group(1))
                elif in_summary and line.startswith("Final total cell area"):
                    match = re.search(r'([\d.]+)$', line)
                    if match:
                        power_data["final_total_area"] = float(match.group(1))
                elif in_summary and line.startswith("Total cell area decreased"):
                    match = re.search(r'([\d.]+)$', line)
                    if match:
                        power_data["total_area_decreased"] = float(match.group(1))
                elif in_summary and line.startswith("Percentage of total cell area decreased"):
                    match = re.search(r'([\d.]+)%', line)
                    if match:
                        power_data["percentage_area_decreased"] = float(match.group(1))
                elif in_summary and line.startswith("Percentage of datapath cell area decreased"):
                    match = re.search(r'([\d.]+)%', line)
                    if match:
                        power_data["percentage_datapath_area_decreased"] = float(match.group(1))
                elif in_summary and "Information: Elapsed time" in line:
                    break

        elif has_fixing_summary:
            power_data["report_format"] = "fixing_summary"
            # Extract Fixing Summary data (alternative format)
            in_fixing_summary = False
            for line in lines:
                line = line.strip()

                if "Fixing Summary:" in line:
                    in_fixing_summary = True
                    continue
                elif in_fixing_summary and line.startswith("Total number of buffers removed"):
                    match = re.search(r'(\d+)$', line)
                    if match:
                        power_data["buffers_removed"] = int(match.group(1))
                elif in_fixing_summary and line.startswith("Percentage of buffers removed"):
                    match = re.search(r'([\d.]+)%', line)
                    if match:
                        power_data["percentage_buffers_removed"] = float(match.group(1))
                elif in_fixing_summary and line.startswith("Percentage of cells removed"):
                    match = re.search(r'([\d.]+)%', line)
                    if match:
                        power_data["percentage_cells_removed"] = float(match.group(1))
                elif in_fixing_summary and line.startswith("Information:"):
                    break

        return power_data
    
    @staticmethod
    def parse_area_log(log_content: str) -> Dict[str, Any]:
        """Parse area fixing log and extract key metrics"""
        area_data = {
            "fix_type": "area",
            "report_format": None,  # Will be set to "eco_summary" or "fixing_summary"
            "elapsed_time_seconds": 0,
            "initial_total_area": 0.0,
            "final_total_area": 0.0,
            "total_area_decreased": 0.0,
            "percentage_area_decreased": 0.0,
            "percentage_datapath_area_decreased": 0.0,
            "buffers_removed": 0,
            "percentage_buffers_removed": 0.0,
            "percentage_cells_removed": 0.0,
            "raw_content": log_content
        }

        lines = log_content.strip().split('\n')

        # Extract elapsed time
        for line in lines:
            if "Information: Elapsed time" in line:
                match = re.search(r'\[\s*(\d+)\s+seconds\s*\]', line)
                if match:
                    area_data["elapsed_time_seconds"] = int(match.group(1))
                    break

        # Check which summary section exists (mutually exclusive)
        has_final_eco_summary = any("Final ECO Summary:" in line for line in lines)
        has_fixing_summary = any("Fixing Summary:" in line for line in lines)

        if has_final_eco_summary:
            area_data["report_format"] = "eco_summary"
            # Extract Final ECO Summary data
            in_summary = False
            for line in lines:
                line = line.strip()

                if "Final ECO Summary:" in line:
                    in_summary = True
                    continue
                elif in_summary and line.startswith("Number of gate_sizing commands"):
                    match = re.search(r'(\d+)$', line)
                    if match:
                        area_data["gate_sizing_commands"] = int(match.group(1))
                elif in_summary and line.startswith("Initial total cell area"):
                    match = re.search(r'([\d.]+)$', line)
                    if match:
                        area_data["initial_total_area"] = float(match.group(1))
                elif in_summary and line.startswith("Final total cell area"):
                    match = re.search(r'([\d.]+)$', line)
                    if match:
                        area_data["final_total_area"] = float(match.group(1))
                elif in_summary and line.startswith("Total cell area decreased"):
                    match = re.search(r'([\d.]+)$', line)
                    if match:
                        area_data["total_area_decreased"] = float(match.group(1))
                elif in_summary and line.startswith("Percentage of total cell area decreased"):
                    match = re.search(r'([\d.]+)%', line)
                    if match:
                        area_data["percentage_area_decreased"] = float(match.group(1))
                elif in_summary and line.startswith("Percentage of datapath cell area decreased"):
                    match = re.search(r'([\d.]+)%', line)
                    if match:
                        area_data["percentage_datapath_area_decreased"] = float(match.group(1))
                elif in_summary and "Information: Elapsed time" in line:
                    break

        elif has_fixing_summary:
            area_data["report_format"] = "fixing_summary"
            # Extract Fixing Summary data (alternative format)
            in_fixing_summary = False
            for line in lines:
                line = line.strip()

                if "Fixing Summary:" in line:
                    in_fixing_summary = True
                    continue
                elif in_fixing_summary and line.startswith("Total number of buffers removed"):
                    match = re.search(r'(\d+)$', line)
                    if match:
                        area_data["buffers_removed"] = int(match.group(1))
                elif in_fixing_summary and line.startswith("Percentage of buffers removed"):
                    match = re.search(r'([\d.]+)%', line)
                    if match:
                        area_data["percentage_buffers_removed"] = float(match.group(1))
                elif in_fixing_summary and line.startswith("Percentage of cells removed"):
                    match = re.search(r'([\d.]+)%', line)
                    if match:
                        area_data["percentage_cells_removed"] = float(match.group(1))
                elif in_fixing_summary and line.startswith("Information:"):
                    break

        return area_data
    
    @staticmethod
    def parse_timing_unfixing(log_content: str, tool_using: bool) -> Dict[str, Any]:
        """Parse timing unfixable violations log and extract timing paths"""
        unfixing_data = {
            "fix_type": "timing",
            "reason_definitions": {},
            "timing_paths": [],
            "paths_sorted_by_slack": [],
            "summary": {
                "total_paths": 0,
                "unfix_reason_distribution": {}
            },
            "raw_content": log_content
        }
        
        lines = log_content.strip().split('\n')
        
        # Parse reason definitions
        in_definitions = False
        for line in lines:
            line = line.strip()
            
            if line == "Unfixable violations:":
                in_definitions = True
                continue
            elif in_definitions and line.startswith("Violation"):
                in_definitions = False
                break
            elif in_definitions and line and not line.startswith("Unfixable"):
                # Parse reason definition: "  A - There are available library cells outside area limit"
                match = re.match(r'^([A-Z])\s*-\s*(.+)$', line)
                if match:
                    reason_code = match.group(1)
                    reason_description = match.group(2)
                    unfixing_data["reason_definitions"][reason_code] = reason_description
        
        # Parse timing paths
        current_path = None
        in_violations = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Violation") and "Reasons" in line:
                in_violations = True
                continue
            elif line.startswith("---"):
                continue
            elif not in_violations:
                continue
                
            # Parse S: (startpoint) - can be indented
            if "S:" in line and line.strip().startswith("S:"):
                if current_path:
                    # Finalize previous path
                    unfixing_data["timing_paths"].append(current_path)
                
                startpoint = line.strip()[2:].strip()
                current_path = {
                    "startpoint": startpoint,
                    "endpoint": None,
                    "slack": 0.0,
                    "intermediate_nodes": [],
                    "accumulated_priority": 0,
                    "unfixable_reason_counts": {},
                    "total_unfixable_occurrences": 0
                }
            
            # Parse E: (endpoint) - can be indented
            elif "E:" in line and line.strip().startswith("E:"):
                if current_path:
                    # Extract endpoint and slack - handle indented lines and multiple spaces
                    stripped_line = line.strip()[2:]  # Remove E: and leading spaces
                    
                    # Try to split from right to get slack value
                    parts = stripped_line.rsplit(None, 1)  # Split from right, get last part
                    if len(parts) == 2:
                        try:
                            slack_val = float(parts[1])
                            current_path["endpoint"] = parts[0].strip()
                            current_path["slack"] = slack_val
                        except ValueError:
                            current_path["endpoint"] = stripped_line.strip()
                    else:
                        current_path["endpoint"] = stripped_line.strip()
            
            # Parse C: (continuation) - can be indented
            elif "C:" in line and line.strip().startswith("C:"):
                # C: indicates a continuation from previous path - this starts a new path
                if current_path:
                    # Finalize previous path first
                    unfixing_data["timing_paths"].append(current_path)
                
                continuation_point = line.strip()[2:].strip()
                current_path = {
                    "startpoint": f"[Continuation from] {continuation_point}",
                    "endpoint": None,
                    "slack": 0.0,
                    "intermediate_nodes": [{
                        "type": "continuation",
                        "name": continuation_point,
                        "reasons": [],
                        "priority": None
                    }],
                    "accumulated_priority": 0,
                    "unfixable_reason_counts": {},
                    "total_unfixable_occurrences": 0
                }
            
            # Parse intermediate nodes (indented lines with reason codes)
            elif line and not line.startswith(("S:", "E:", "C:")):
                if current_path:
                    # Parse intermediate node with reasons and priority
                    # Format: "    path/name    A W    P6"
                    parts = line.rsplit(None, 1)  # Split from right to get priority
                    if len(parts) == 2 and parts[1].startswith("P"):
                        node_and_reasons = parts[0].strip()
                        priority_str = parts[1]
                        
                        # Extract priority number
                        priority_match = re.match(r'P(\d+)', priority_str)
                        priority = int(priority_match.group(1)) if priority_match else 0
                        
                        # Split node path and reason codes
                        node_parts = node_and_reasons.rsplit(None, 1)
                        if len(node_parts) == 2:
                            node_path = node_parts[0].strip()
                            reasons = list(node_parts[1].strip())
                        else:
                            node_path = node_and_reasons
                            reasons = []
                    else:
                        # No priority, just node path and possibly reasons
                        node_parts = line.rsplit(None, 1)
                        if len(node_parts) == 2 and all(c.isalpha() and c.isupper() for c in node_parts[1]):
                            node_path = node_parts[0].strip()
                            reasons = list(node_parts[1].strip())
                        else:
                            node_path = line.strip()
                            reasons = []
                        priority = 0
                    
                    # Add intermediate node
                    current_path["intermediate_nodes"].append({
                        "type": "intermediate", 
                        "name": node_path,
                        "reasons": reasons,
                        "priority": priority
                    })
                    
                    # Update path statistics
                    current_path["accumulated_priority"] += priority
                    for reason in reasons:
                        current_path["unfixable_reason_counts"][reason] = \
                            current_path["unfixable_reason_counts"].get(reason, 0) + 1
                        current_path["total_unfixable_occurrences"] += 1
        
        # Finalize last path
        if current_path:
            unfixing_data["timing_paths"].append(current_path)
        
        # Sort paths by slack (most negative first)
        unfixing_data["paths_sorted_by_slack"] = sorted(
            unfixing_data["timing_paths"], 
            key=lambda x: x["slack"]
        )
        
        # Calculate summary statistics
        if unfixing_data["timing_paths"]:
            unfixing_data["summary"]["total_paths"] = len(unfixing_data["timing_paths"])

            # Calculate reason distribution across all paths
            if tool_using:
                # When tool_using is True, compute slack-weighted reason distribution
                # For each endpoint, weight reasons by slack: weight = reason_count * slack
                # More negative slack = higher weight (more critical)
                weighted_reason_distribution = {}
                for path in unfixing_data["timing_paths"]:
                    path_slack = path["slack"]
                    for reason, count in path["unfixable_reason_counts"].items():
                        # Weight = count * slack (both count and result will be negative for violations)
                        weight = count * path_slack
                        if reason in weighted_reason_distribution:
                            weighted_reason_distribution[reason] += weight
                        else:
                            weighted_reason_distribution[reason] = weight
                # Round all weights to 4 decimal places
                weighted_reason_distribution = {k: round(v, 4) for k, v in weighted_reason_distribution.items()}
                unfixing_data["summary"]["unfix_reason_distribution"] = weighted_reason_distribution
            else:
                # Original unweighted count
                for path in unfixing_data["timing_paths"]:
                    for reason, count in path["unfixable_reason_counts"].items():
                        unfixing_data["summary"]["unfix_reason_distribution"][reason] = \
                            unfixing_data["summary"]["unfix_reason_distribution"].get(reason, 0) + count

            # Add slack distribution if tool_using is True
            if tool_using:
                # Filter paths with negative slack
                negative_slack_paths = [path for path in unfixing_data["timing_paths"] if path["slack"] < 0]

                if negative_slack_paths:
                    # Get the most negative slack
                    most_negative_slack = min(path["slack"] for path in negative_slack_paths)

                    # Create 10 uniform bins from 0 to most_negative_slack
                    # Generate 11 bin edges (boundaries for 10 bins)
                    num_bins = 10
                    slack_bin_ranges = []
                    bin_width = (0 - most_negative_slack) / num_bins

                    for i in range(num_bins + 1):
                        bin_edge = 0 - (i * bin_width)
                        slack_bin_ranges.append(round(bin_edge, 4))

                    # Count paths in each bin
                    number_of_paths = [0] * num_bins

                    for path in negative_slack_paths:
                        path_slack = path["slack"]
                        # Find which bin this path belongs to
                        for bin_idx in range(num_bins):
                            if bin_idx == num_bins - 1:
                                # Last bin includes the most negative value
                                if slack_bin_ranges[bin_idx] >= path_slack >= slack_bin_ranges[bin_idx + 1]:
                                    number_of_paths[bin_idx] += 1
                                    break
                            else:
                                # Other bins: left_edge >= slack > right_edge
                                if slack_bin_ranges[bin_idx] >= path_slack > slack_bin_ranges[bin_idx + 1]:
                                    number_of_paths[bin_idx] += 1
                                    break

                    unfixing_data["summary"]["slack_distribution"] = {
                        "number_of_paths": number_of_paths,
                        "slack_bin_ranges": slack_bin_ranges
                    }
                else:
                    # No negative slack paths found
                    unfixing_data["summary"]["slack_distribution"] = {
                        "number_of_paths": [0] * 10,
                        "slack_bin_ranges": [0.0] * 11
                    }

        return unfixing_data

    @staticmethod
    def parse_area_power_unfixing(log_content: str) -> Dict[str, Any]:
        """Parse area/power unfixable cells log and extract unusable cells summary"""
        unfixing_data = {
            "fix_type": "area_power",
            "unusable_cells_summary": {},
            "reason_definitions": {
                "B": "Benefit from sizing cell is too small",
                "L": "Physical constraints restrict sizing",
                "S": "Cell has no alternate library cell with better power",
                "T": "Sizing cell might degrade timing",
                "U": "UPF restrictions prevent sizing",
                "V": "Cell is invalid or has dont_touch attribute",
                "W": "Sizing cell might degrade DRC",
                "X": "Cell is unusable for ECO",
                "Z": "Cell is sized"
            },
            "raw_content": log_content
        }
        
        lines = log_content.strip().split('\n')
        
        # Look for "Unusable cells summary:" section
        in_summary = False
        for line in lines:
            line = line.strip()
            
            if line == "Unusable cells summary:":
                in_summary = True
                continue
            elif in_summary and line.startswith("-------"):
                continue
            elif in_summary and line.startswith("Total number of cells checked"):
                match = re.search(r':\s*(\d+)', line)
                if match:
                    unfixing_data["total_cells_checked"] = int(match.group(1))
            elif in_summary and line.startswith("Total number of cells with reason code"):
                match = re.search(r'reason code ([A-Z])\s*:\s*(\d+)', line)
                if match:
                    reason_code = match.group(1)
                    count = int(match.group(2))
                    unfixing_data["unusable_cells_summary"][reason_code] = count
            elif in_summary and line.startswith("Information: Reported unusable reasons"):
                # End of summary section
                break
        if not in_summary:
            return None
        return unfixing_data


class ECODatabase:
    """
    Simple ECO Database for storing optimization iterations
    
    Features:
    - JSON-based storage for simplicity
    - Easy to extend with new fields
    - Handles iteration 0 baseline correctly
    - Dummy implementations for rapid prototyping
    """
    
    def __init__(self, db_file: str = "eco_database.json"):
        self.db_file = db_file
        self.data = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "version": "1.0",
                "description": "ECO optimization iteration database"
            },
            "iterations": {}
        }
        self._load_database()
        print(f"📁 ECO Database initialized: {self.db_file}")
    
    def _load_database(self):
        """Load existing database or create new one"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    self.data = json.load(f)
                print(f"✅ Loaded existing database with {len(self.data['iterations'])} iterations")
            except Exception as e:
                print(f"⚠️  Could not load database: {e}, creating new one")
        else:
            print("📄 Creating new database")
    
    def _save_database(self):
        """Save database to file"""
        try:
            with open(self.db_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"❌ Failed to save database: {e}")
    
    def store_iteration(self, iteration_data: IterationData):
        """Store complete iteration data"""
        iteration_key = str(iteration_data.iteration)
        self.data["iterations"][iteration_key] = asdict(iteration_data)
        self._save_database()
        
        if iteration_data.iteration == 0:
            print(f"✅ Baseline iteration {iteration_data.iteration} stored (QoR + Power only)")
        else:
            eco_type = iteration_data.last_eco_option or "none"
            print(f"✅ Iteration {iteration_data.iteration} stored (ECO: {eco_type})")
    
    def get_iteration(self, iteration: int) -> Optional[IterationData]:
        """Retrieve iteration data"""
        iteration_key = str(iteration)
        if iteration_key not in self.data["iterations"]:
            return None
        
        data = self.data["iterations"][iteration_key]
        return IterationData(**data)
    
    def get_all_iterations(self) -> List[IterationData]:
        """Get all stored iterations"""
        iterations = []
        for iteration_key in sorted(self.data["iterations"].keys(), key=int):
            data = self.data["iterations"][iteration_key]
            iterations.append(IterationData(**data))
        return iterations
    
    def get_eco_history(self) -> List[Dict[str, Any]]:
        """Get ECO command history (excluding iteration 0)"""
        history = []
        for iteration_data in self.get_all_iterations():
            if iteration_data.iteration > 0 and iteration_data.last_eco_option:
                history.append({
                    "iteration": iteration_data.iteration,
                    "eco_option": iteration_data.last_eco_option,
                    "timestamp": iteration_data.timestamp,
                    "has_fixing_report": iteration_data.fixing_report is not None,
                    "has_unfixable_reasons": iteration_data.unfixable_reasons is not None
                })
        return history
    
    def get_optimization_trend(self) -> Dict[str, List]:
        """Get optimization trends across iterations"""
        iterations = self.get_all_iterations()
        trends = {
            "iterations": [],
            "total_violations": [],
            "total_power": [],
            "eco_commands": []
        }
        
        for iter_data in iterations:
            trends["iterations"].append(iter_data.iteration)
            
            # Extract violations from QoR report (dummy implementation)
            qor = iter_data.qor_report
            violations = qor.get("setup_violations", 0) + qor.get("hold_violations", 0) + qor.get("drc_violations", 0)
            trends["total_violations"].append(violations)
            
            # Extract power from power report
            power = iter_data.power_report.get("total_power", 0.0)
            trends["total_power"].append(power)
            
            # ECO command (None for iteration 0)
            trends["eco_commands"].append(iter_data.last_eco_option)
        
        return trends
    
    def export_summary(self) -> Dict[str, Any]:
        """Export database summary"""
        iterations = self.get_all_iterations()
        return {
            "total_iterations": len(iterations),
            "eco_history": self.get_eco_history(),
            "optimization_trends": self.get_optimization_trend(),
            "latest_iteration": max([i.iteration for i in iterations]) if iterations else -1
        }
    
    def parse_and_store_reports(self, iteration: int, qor_file_path: str, power_file_path: str,
                               eco_option: str = None, fixing_report_data: Dict = None, 
                               unfixable_reasons: List[str] = None):
        """Parse report files and store iteration data"""
        try:
            # Read and parse QoR report
            with open(qor_file_path, 'r') as f:
                qor_content = f.read()
            qor_data = ReportParser.parse_qor_report(qor_content)
            
            # Read and parse Power report
            with open(power_file_path, 'r') as f:
                power_content = f.read()
            power_data = ReportParser.parse_power_report(power_content)
            
            # Create iteration data
            iteration_data = IterationData(
                iteration=iteration,
                timestamp=datetime.now().isoformat(),
                qor_report=qor_data,
                power_report=power_data,
                last_eco_option=eco_option,
                fixing_report=fixing_report_data,
                unfixable_reasons=unfixable_reasons if eco_option in ["drc", "timing"] else None,
                metrics={
                    "total_violations": qor_data["setup_violating_paths"] + qor_data["hold_violating_paths"] + 
                                      qor_data["min_capacitance_count"] + qor_data["max_transition_count"],
                    "parsing_success": True
                }
            )
            
            # Store in database
            self.store_iteration(iteration_data)
            
            print(f"✅ Parsed and stored iteration {iteration}")
            print(f"   QoR: {qor_data['setup_violating_paths']} setup, {qor_data['hold_violating_paths']} hold, "
                  f"{qor_data['min_capacitance_count'] + qor_data['max_transition_count']} DRC violations")
            print(f"   Power: {power_data['total_power']:.3e}W total")
            
            return iteration_data
            
        except FileNotFoundError as e:
            print(f"❌ Report file not found: {e}")
            return None
        except Exception as e:
            print(f"❌ Failed to parse reports for iteration {iteration}: {e}")
            return None
    
    def parse_baseline_reports(self, qor_file_path: str = "reports/report_qor_0.txt", 
                              power_file_path: str = "reports/report_power_0.txt"):
        """Parse and store baseline reports (iteration 0)"""
        return self.parse_and_store_reports(
            iteration=0,
            qor_file_path=qor_file_path,
            power_file_path=power_file_path,
            eco_option=None,
            fixing_report_data=None,
            unfixable_reasons=None
        )
    
    def print_status(self):
        """Print database status"""
        iterations = self.get_all_iterations()
        print(f"\n📊 ECO Database Status")
        print(f"Total iterations: {len(iterations)}")
        
        if iterations:
            print(f"Iteration range: {min(i.iteration for i in iterations)} - {max(i.iteration for i in iterations)}")
            
            eco_counts = {}
            for iter_data in iterations:
                if iter_data.iteration > 0 and iter_data.last_eco_option:
                    eco_type = iter_data.last_eco_option
                    eco_counts[eco_type] = eco_counts.get(eco_type, 0) + 1
            
            if eco_counts:
                print(f"ECO command distribution: {eco_counts}")


# Helper functions for creating iteration data
def create_baseline_iteration(iteration: int = 0) -> IterationData:
    """Create baseline iteration (iteration 0) with only QoR and Power reports"""
    return IterationData(
        iteration=iteration,
        timestamp=datetime.now().isoformat(),
        qor_report={
            "setup_violations": 100,
            "hold_violations": 50,
            "drc_violations": 30,
            "total_area": 1000.0,
            "raw_content": f"Dummy QoR report for iteration {iteration}"
        },
        power_report={
            "total_power": 1000.0,
            "clock_power": 300.0,
            "register_power": 250.0,
            "combinational_power": 350.0,
            "leakage_power": 100.0,
            "raw_content": f"Dummy Power report for iteration {iteration}"
        },
        # No ECO data for baseline
        last_eco_option=None,
        fixing_report=None,
        unfixable_reasons=None,
        metrics={"baseline": True}
    )


def create_eco_iteration(iteration: int, eco_option: str, violations_fixed: int = 10) -> IterationData:
    """Create ECO iteration with fixing results"""
    
    # Simulate improvement
    base_violations = max(0, 100 - iteration * 10)
    base_power = max(500.0, 1000.0 - iteration * 50)
    
    return IterationData(
        iteration=iteration,
        timestamp=datetime.now().isoformat(),
        qor_report={
            "setup_violations": max(0, base_violations - violations_fixed),
            "hold_violations": max(0, 30 - iteration * 3),
            "drc_violations": max(0, 20 - iteration * 2),
            "total_area": 1000.0 + iteration * 10,
            "raw_content": f"Dummy QoR report for iteration {iteration}"
        },
        power_report={
            "total_power": base_power,
            "clock_power": base_power * 0.3,
            "register_power": base_power * 0.25,
            "combinational_power": base_power * 0.35,
            "leakage_power": base_power * 0.1,
            "raw_content": f"Dummy Power report for iteration {iteration}"
        },
        last_eco_option=eco_option,
        fixing_report={
            "command": f"opt_{eco_option} -actions gate_sizing -timeout 300",
            "execution_time": 250.0,
            "success": True,
            "violations_fixed": violations_fixed,
            "raw_output": f"Dummy {eco_option} fixing output for iteration {iteration}"
        },
        # Unfixable reasons only for DRC and Timing
        unfixable_reasons=[f"Cannot fix constraint conflict in {eco_option}"] if eco_option in ["drc", "timing"] else None,
        metrics={
            "improvement_rate": violations_fixed / base_violations * 100 if base_violations > 0 else 0,
            "convergence": "improving" if iteration < 3 else "converging"
        }
    )


def demo_database():
    """Demonstrate database usage with real report parsing"""
    print("🚀 ECO Database Demo with Real Report Parsing")
    print("=" * 60)
    
    # Initialize database
    db = ECODatabase("demo_eco_parsed.json")
    
    # Parse and store baseline reports (iteration 0)
    print(f"\n📊 Parsing baseline reports...")
    baseline_data = db.parse_baseline_reports()
    
    if baseline_data:
        print(f"\n📈 Parsed Baseline Data:")
        qor = baseline_data.qor_report
        power = baseline_data.power_report
        print(f"  Design: {qor['design_name']}")
        print(f"  Setup violations: {qor['setup_violating_paths']}")
        print(f"  Hold violations: {qor['hold_violating_paths']}")
        print(f"  DRC violations: {qor['min_capacitance_count'] + qor['max_transition_count']}")
        print(f"  Total power: {power['total_power']:.3e}W")
        print(f"  Clock power: {power['clock_tree_power']:.3e}W")
        print(f"  Register power: {power['register_power']:.3e}W")
        print(f"  Combinational power: {power['combinational_power']:.3e}W")
        print(f"  Cell area: {qor['total_cell_area']:.1f}")
        print(f"  Leaf cells: {qor['leaf_cell_count']}")
    
    # Simulate storing additional ECO iterations with dummy fixing data
    eco_options = ["power", "drc", "timing"]
    for i in range(1, 4):
        print(f"\n📊 Simulating ECO iteration {i}...")
        eco_option = eco_options[(i-1) % len(eco_options)]
        
        # Create dummy fixing report
        fixing_report = {
            "command": f"opt_{eco_option} -actions gate_sizing -timeout 300",
            "execution_time": 250.0,
            "success": True,
            "violations_fixed": 10,
            "raw_output": f"Dummy {eco_option} fixing output for iteration {i}"
        }
        
        # Create dummy unfixable reasons for DRC/Timing
        unfixable_reasons = None
        if eco_option in ["drc", "timing"]:
            unfixable_reasons = [f"Cannot fix constraint conflict in {eco_option}", "Design limitation"]
        
        # Store with dummy data (since we don't have iteration 1+ report files)
        dummy_iter = create_eco_iteration(i, eco_option, violations_fixed=15-i*3)
        dummy_iter.fixing_report = fixing_report
        dummy_iter.unfixable_reasons = unfixable_reasons
        db.store_iteration(dummy_iter)
    
    # Print database status
    db.print_status()
    
    # Show optimization trends
    print(f"\n📈 Optimization Trends:")
    trends = db.get_optimization_trend()
    for i, iteration in enumerate(trends["iterations"]):
        eco_cmd = trends["eco_commands"][i] or "baseline"
        violations = trends["total_violations"][i]
        power = trends["total_power"][i]
        print(f"  Iteration {iteration}: {eco_cmd:8s} - {violations:3d} violations, {power:.3e}W")
    
    # Show ECO history
    print(f"\n🔧 ECO Command History:")
    history = db.get_eco_history()
    for entry in history:
        print(f"  Iteration {entry['iteration']}: {entry['eco_option']} "
              f"(fixing: {'✅' if entry['has_fixing_report'] else '❌'}, "
              f"unfixable: {'✅' if entry['has_unfixable_reasons'] else '❌'})")
    
    print(f"\n✅ Demo completed! Database saved as: demo_eco_parsed.json")


def test_report_parsing():
    """Test report parsing functions directly"""
    print("🧪 Testing Report Parsing Functions")
    print("=" * 50)
    
    # Test QoR report parsing
    try:
        with open("reports/report_qor_0.txt", 'r') as f:
            qor_content = f.read()
        
        qor_data = ReportParser.parse_qor_report(qor_content)
        
        print("✅ QoR Report Parsed Successfully:")
        print(f"  Design: {qor_data['design_name']}")
        print(f"  Setup violations: {qor_data['setup_violating_paths']}")
        print(f"  Hold violations: {qor_data['hold_violating_paths']}")
        print(f"  Setup slack: {qor_data['setup_critical_path_slack']}")
        print(f"  Hold slack: {qor_data['hold_critical_path_slack']}")
        print(f"  Min cap violations: {qor_data['min_capacitance_count']}")
        print(f"  Max trans violations: {qor_data['max_transition_count']}")
        print(f"  Total cell area: {qor_data['total_cell_area']}")
        print(f"  Leaf cell count: {qor_data['leaf_cell_count']}")
        
    except Exception as e:
        print(f"❌ QoR parsing failed: {e}")
    
    print()
    
    # Test Power report parsing
    try:
        with open("reports/report_power_0.txt", 'r') as f:
            power_content = f.read()
        
        power_data = ReportParser.parse_power_report(power_content)
        
        print("✅ Power Report Parsed Successfully:")
        print(f"  Design: {power_data['design_name']}")
        print(f"  Total power: {power_data['total_power']:.3e}W")
        print(f"  Clock tree: {power_data['clock_tree_power']:.3e}W")
        print(f"  Register: {power_data['register_power']:.3e}W")
        print(f"  Combinational: {power_data['combinational_power']:.3e}W")
        print(f"  Net switching: {power_data['net_switching_power']:.3e}W")
        print(f"  Cell internal: {power_data['cell_internal_power']:.3e}W")
        print(f"  Leakage: {power_data['leakage_power']:.3e}W")
        print(f"  Power breakdown: {power_data['power_breakdown']}")
        
    except Exception as e:
        print(f"❌ Power parsing failed: {e}")
    
    print()
    
    # Test DRC log parsing
    try:
        with open("report_examples/fix_drc_0.txt", 'r') as f:
            drc_content = f.read()
        
        drc_data = ReportParser.parse_drc_log(drc_content)
        
        print("✅ DRC Log Parsed Successfully:")
        print(f"  Detected violations: {drc_data['detected_violations']}")
        print(f"  Remaining violations: {drc_data['remaining_violations']}")
        print(f"  Elapsed time: {drc_data['elapsed_time_seconds']} seconds")
        print(f"  Gate sizing commands: {drc_data['gate_sizing_commands']}")
        print(f"  Total commands: {drc_data['total_commands']}")
        print(f"  Area increased: {drc_data['area_increased']}")
        
    except Exception as e:
        print(f"❌ DRC log parsing failed: {e}")
    
    print()
    
    # Test DRC unfixing parsing
    try:
        with open("report_examples/unfix_drc_0.txt", 'r') as f:
            unfixing_content = f.read()
        
        unfixing_data = ReportParser.parse_drc_unfixing(unfixing_content)
        
        print("✅ DRC Unfixing Log Parsed Successfully:")
        print(f"  Total unfixable violations: {unfixing_data['total_unfixable_violations']}")
        print(f"  Reason definitions: {len(unfixing_data['reason_definitions'])}")
        
        for reason_code, data in unfixing_data['violations_by_reason'].items():
            if data['count'] > 0:
                definition = unfixing_data['reason_definitions'].get(reason_code, 'Unknown')
                print(f"  Reason {reason_code}: {data['count']} violations - {definition}")
        
    except Exception as e:
        print(f"❌ DRC unfixing parsing failed: {e}")
    
    print()
    
    # Test timing log parsing
    try:
        with open("report_examples/fix_timing_0.txt", 'r') as f:
            timing_content = f.read()
        
        timing_data = ReportParser.parse_timing_log(timing_content)
        
        print("✅ Timing Log Parsed Successfully:")
        print(f"  Elapsed time: {timing_data['elapsed_time_seconds']} seconds")
        print(f"  Gate sizing commands: {timing_data['gate_sizing_commands']}")
        print(f"  Total commands: {timing_data['total_commands']}")
        print(f"  Area increased: {timing_data['area_increased']}")
        print(f"  Endpoints found: {timing_data['total_violating_endpoints_found']}")
        print(f"  Endpoints fixed: {timing_data['total_violating_endpoints_fixed']}")
        print(f"  Endpoints remaining: {timing_data['total_violating_endpoints_remaining']}")
        print(f"  Fix percentage: {timing_data['percentage_violations_fixed']}%")
        
    except Exception as e:
        print(f"❌ Timing log parsing failed: {e}")
    
    print()
    
    # Test power log parsing
    try:
        with open("report_examples/fix_power_0.txt", 'r') as f:
            power_log_content = f.read()
        
        power_log_data = ReportParser.parse_power_log(power_log_content)
        
        print("✅ Power Log Parsed Successfully:")
        print(f"  Elapsed time: {power_log_data['elapsed_time_seconds']} seconds")
        print(f"  Gate sizing commands: {power_log_data['gate_sizing_commands']}")
        print(f"  Initial total area: {power_log_data['initial_total_area']}")
        print(f"  Final total area: {power_log_data['final_total_area']}")
        print(f"  Total area decreased: {power_log_data['total_area_decreased']}")
        print(f"  Percentage area decreased: {power_log_data['percentage_area_decreased']}%")
        print(f"  Datapath area decreased: {power_log_data['percentage_datapath_area_decreased']}%")
        print(f"  Initial setup endpoints: {power_log_data['setup_violating_endpoints_initial']}")
        print(f"  Final setup endpoints: {power_log_data['setup_violating_endpoints_final']}")
        
    except Exception as e:
        print(f"❌ Power log parsing failed: {e}")
    
    print()
    
    # Test timing unfixing parsing
    try:
        with open("report_examples/unfix_timing_0.txt", 'r') as f:
            timing_unfixing_content = f.read()
        
        timing_unfixing_data = ReportParser.parse_timing_unfixing(timing_unfixing_content)
        
        print("✅ Timing Unfixing Log Parsed Successfully:")
        print(f"  Reason definitions: {len(timing_unfixing_data['reason_definitions'])}")
        print(f"  Total timing paths: {timing_unfixing_data['summary']['total_paths']}")
        print(f"  Worst slack: {timing_unfixing_data['summary']['worst_slack']}")
        print(f"  Reason distribution: {timing_unfixing_data['summary']['unfix_reason_distribution']}")
        
        # Show top 3 most critical paths
        print("  Top 3 most critical paths:")
        for i, path in enumerate(timing_unfixing_data['paths_sorted_by_slack'][:3]):
            print(f"    {i+1}. Slack: {path['slack']}, Priority: {path['accumulated_priority']}, "
                  f"Reasons: {path['unfixable_reason_counts']}")
        
    except Exception as e:
        print(f"❌ Timing unfixing parsing failed: {e}")


if __name__ == "__main__":
    demo_database()
