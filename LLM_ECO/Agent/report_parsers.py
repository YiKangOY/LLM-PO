#!/usr/bin/env python3
"""
Report Parser Scripts for ECO Agent System
Provides specialized parsers for power and QoR reports to extract structured data
instead of relying on LLM parsing for basic data extraction.
"""

import re
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class PowerMetrics:
    """Structured power report metrics"""
    total_power: float = None
    internal_power: float = None
    switching_power: float = None
    leakage_power: float = None
    clock_tree_power: float = 0.0
    register_power: float = 0.0
    combinational_power: float = 0.0
    
    # Power breakdown percentages
    internal_power_pct: float = 0.0
    switching_power_pct: float = 0.0
    leakage_power_pct: float = 0.0
    
    # Power group details
    power_groups: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        if self.power_groups is None:
            self.power_groups = {}


@dataclass
class QoRMetrics:
    """Structured QoR report metrics"""
    # Setup timing metrics
    setup_critical_path_slack: float = None
    setup_total_negative_slack: float = None
    setup_violating_paths: int = None
    setup_levels_of_logic: int = None
    setup_critical_path_length: float = 0.0
    
    # Hold timing metrics
    hold_critical_path_slack: float = None
    hold_total_negative_slack: float = None
    hold_violating_paths: int = None
    hold_levels_of_logic: int = None
    hold_critical_path_length: float = 0.0
    
    # Area metrics
    total_cell_area: float = None
    net_interconnect_area: float = 0.0
    design_area: float = None
    
    # Cell and pin counts
    pin_count: int = 0
    leaf_cell_count: int = 0
    hierarchical_cell_count: int = 0
    
    # DRC violations
    min_capacitance_count: int = 0
    max_transition_count: int = 0
    min_capacitance_cost: float = 0.0
    max_transition_cost: float = 0.0
    total_drc_cost: float = 0.0
    
    # Design info
    design_name: str = ""
    report_date: str = ""
    version: str = ""


@dataclass
class ParsedPowerReport:
    """Complete parsed power report"""
    report_type: str = "power"
    timestamp: datetime = None
    raw_content: str = ""
    metrics: PowerMetrics = None
    parsing_successful: bool = False
    parsing_errors: List[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metrics is None:
            self.metrics = PowerMetrics()
        if self.parsing_errors is None:
            self.parsing_errors = []


@dataclass 
class ParsedQoRReport:
    """Complete parsed QoR report"""
    report_type: str = "qor"
    timestamp: datetime = None
    raw_content: str = ""
    metrics: QoRMetrics = None
    parsing_successful: bool = False
    parsing_errors: List[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metrics is None:
            self.metrics = QoRMetrics()
        if self.parsing_errors is None:
            self.parsing_errors = []


class PowerReportParser:
    """Parser for power reports from PT Shell"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse(self, report_content: str) -> ParsedPowerReport:
        """Parse power report content and extract structured metrics"""
        result = ParsedPowerReport(raw_content=report_content)
        
        try:
            # Extract basic header information
            self._parse_header(report_content, result)
            
            # Extract power group data
            self._parse_power_groups(report_content, result)
            
            # Extract total power summary
            self._parse_power_summary(report_content, result)
            
            result.parsing_successful = True
            
        except Exception as e:
            error_msg = f"Power report parsing failed: {str(e)}"
            result.parsing_errors.append(error_msg)
            self.logger.error(error_msg)
            result.parsing_successful = False
        
        return result
    
    def _parse_header(self, content: str, result: ParsedPowerReport):
        """Extract header information from power report"""
        # Extract design name
        design_match = re.search(r'Design\s*:\s*(\w+)', content)
        if design_match:
            result.metrics.design_name = design_match.group(1)
        
        # Extract report date  
        date_match = re.search(r'Date\s*:\s*(.+)', content)
        if date_match:
            result.metrics.report_date = date_match.group(1).strip()
    
    def _parse_power_groups(self, content: str, result: ParsedPowerReport):
        """Parse power group breakdown table"""
        # Find the power group table section - look for the table with specific pattern
        power_section = re.search(
            r'Power Group.*?-{70,}\n(.*?)(?=\n\s*Net Switching Power|\n\s*Total Power|\n\n|\Z)', 
            content, re.DOTALL
        )
        
        if not power_section:
            result.parsing_errors.append("Could not find power group table")
            return
        
        power_lines = power_section.group(1).strip().split('\n')
        
        for line in power_lines:
            line = line.strip()
            if not line or line.startswith('-'):
                continue
                
            # Parse each power group line
            # Format: group_name internal switching leakage total (pct%) attrs
            parts = line.split()
            if len(parts) >= 5:
                group_name = parts[0]
                try:
                    internal = float(parts[1]) if parts[1] != '0.0000' else 0.0
                    switching = float(parts[2]) if parts[2] != '0.0000' else 0.0  
                    leakage = float(parts[3]) if parts[3] != '0.0000' else 0.0
                    total = float(parts[4]) if parts[4] != '0.0000' else 0.0
                    
                    # Extract percentage if available
                    pct_match = re.search(r'\(\s*([0-9.]+)%\)', line)
                    percentage = float(pct_match.group(1)) if pct_match else 0.0
                    
                    result.metrics.power_groups[group_name] = {
                        'internal': internal,
                        'switching': switching, 
                        'leakage': leakage,
                        'total': total,
                        'percentage': percentage
                    }
                    
                    # Update specific metrics for key power groups
                    if group_name == 'clock_tree':
                        result.metrics.clock_tree_power = total
                    elif group_name == 'register':
                        result.metrics.register_power = total
                    elif group_name == 'combinational':
                        result.metrics.combinational_power = total
                        
                except ValueError as e:
                    result.parsing_errors.append(f"Error parsing power group {group_name}: {e}")
    
    def _parse_power_summary(self, content: str, result: ParsedPowerReport):
        """Parse power summary section"""
        # Extract Net Switching Power
        net_switching_match = re.search(r'Net Switching Power\s*=\s*([0-9.e-]+)\s*\(\s*([0-9.]+)%\)', content)
        if net_switching_match:
            result.metrics.switching_power = float(net_switching_match.group(1))
            result.metrics.switching_power_pct = float(net_switching_match.group(2))
        
        # Extract Cell Internal Power
        cell_internal_match = re.search(r'Cell Internal Power\s*=\s*([0-9.e-]+)\s*\(\s*([0-9.]+)%\)', content)
        if cell_internal_match:
            result.metrics.internal_power = float(cell_internal_match.group(1))
            result.metrics.internal_power_pct = float(cell_internal_match.group(2))
        
        # Extract Cell Leakage Power
        cell_leakage_match = re.search(r'Cell Leakage Power\s*=\s*([0-9.e-]+)\s*\(\s*([0-9.]+)%\)', content)
        if cell_leakage_match:
            result.metrics.leakage_power = float(cell_leakage_match.group(1))
            result.metrics.leakage_power_pct = float(cell_leakage_match.group(2))
        
        # Extract Total Power
        total_power_match = re.search(r'Total Power\s*=\s*([0-9.e-]+)\s*\(100\.00%\)', content)
        if total_power_match:
            result.metrics.total_power = float(total_power_match.group(1))


class QoRReportParser:
    """Parser for QoR reports from PT Shell"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse(self, report_content: str) -> ParsedQoRReport:
        """Parse QoR report content and extract structured metrics"""
        result = ParsedQoRReport(raw_content=report_content)
        
        try:
            # Extract basic header information
            self._parse_header(report_content, result)
            
            # Extract timing path groups
            self._parse_timing_groups(report_content, result)
            
            # Extract area information
            self._parse_area_info(report_content, result)
            
            # Extract cell and pin counts
            self._parse_cell_pin_counts(report_content, result)
            
            # Extract DRC violations
            self._parse_drc_violations(report_content, result)
            
            result.parsing_successful = True
            
        except Exception as e:
            error_msg = f"QoR report parsing failed: {str(e)}"
            result.parsing_errors.append(error_msg)
            self.logger.error(error_msg)
            result.parsing_successful = False
        
        return result
    
    def _parse_header(self, content: str, result: ParsedQoRReport):
        """Extract header information from QoR report"""
        # Extract design name
        design_match = re.search(r'Design\s*:\s*(\w+)', content)
        if design_match:
            result.metrics.design_name = design_match.group(1)
        
        # Extract version
        version_match = re.search(r'Version\s*:\s*(.+)', content)
        if version_match:
            result.metrics.version = version_match.group(1).strip()
        
        # Extract report date
        date_match = re.search(r'Date\s*:\s*(.+)', content)
        if date_match:
            result.metrics.report_date = date_match.group(1).strip()

    def _extract_timing_group_sections(self, content: str, analysis_type: str) -> List[Any]:
        """Extract timing path group sections for one analysis type."""
        pattern = (
            r"Timing Path Group\s+'([^']+)'\s+\(" +
            re.escape(analysis_type) +
            r"\)\s*-{40,}(.*?)-{40,}"
        )
        return re.findall(pattern, content, re.DOTALL)

    def _select_timing_group_data(self, timing_groups: List[Any]) -> str:
        """Select the first timing group that is not clock-gating."""
        for group_name, group_data in timing_groups:
            normalized_name = group_name.replace("*", "").strip().lower()
            if "clock_gating" in normalized_name:
                continue
            return group_data
        return timing_groups[0][1]

    def _parse_timing_groups(self, content: str, result: ParsedQoRReport):
        """Parse timing path group sections"""
        setup_groups = self._extract_timing_group_sections(content, "max_delay/setup")

        if setup_groups:
            setup_data = self._select_timing_group_data(setup_groups)
            
            # Extract setup metrics
            levels_match = re.search(r'Levels of Logic:\s*(\d+)', setup_data)
            if levels_match:
                result.metrics.setup_levels_of_logic = int(levels_match.group(1))
            
            path_length_match = re.search(r'Critical Path Length:\s*([0-9.-]+)', setup_data)
            if path_length_match:
                result.metrics.setup_critical_path_length = float(path_length_match.group(1))
                
            slack_match = re.search(r'Critical Path Slack:\s*([0-9.-]+)', setup_data)
            if slack_match:
                result.metrics.setup_critical_path_slack = float(slack_match.group(1))
                
            tns_match = re.search(r'Total Negative Slack:\s*([0-9.-]+)', setup_data)
            if tns_match:
                result.metrics.setup_total_negative_slack = float(tns_match.group(1))
                
            paths_match = re.search(r'No\. of Violating Paths:\s*(\d+)', setup_data)
            if paths_match:
                result.metrics.setup_violating_paths = int(paths_match.group(1))
        
        hold_groups = self._extract_timing_group_sections(content, "min_delay/hold")

        if hold_groups:
            hold_data = self._select_timing_group_data(hold_groups)
            
            # Extract hold metrics
            levels_match = re.search(r'Levels of Logic:\s*(\d+)', hold_data)
            if levels_match:
                result.metrics.hold_levels_of_logic = int(levels_match.group(1))
            
            path_length_match = re.search(r'Critical Path Length:\s*([0-9.-]+)', hold_data)
            if path_length_match:
                result.metrics.hold_critical_path_length = float(path_length_match.group(1))
                
            slack_match = re.search(r'Critical Path Slack:\s*([0-9.-]+)', hold_data)
            if slack_match:
                result.metrics.hold_critical_path_slack = float(slack_match.group(1))
                
            tns_match = re.search(r'Total Negative Slack:\s*([0-9.-]+)', hold_data)
            if tns_match:
                result.metrics.hold_total_negative_slack = float(tns_match.group(1))
                
            paths_match = re.search(r'No\. of Violating Paths:\s*(\d+)', hold_data)
            if paths_match:
                result.metrics.hold_violating_paths = int(paths_match.group(1))
    
    def _parse_area_info(self, content: str, result: ParsedQoRReport):
        """Parse area information section"""
        area_section = re.search(r'Area\s*-+\s*(.*?)\s*-+', content, re.DOTALL)
        
        if area_section:
            area_data = area_section.group(1)
            
            # Extract area metrics
            net_area_match = re.search(r'Net Interconnect area:\s*([0-9.]+)', area_data)
            if net_area_match:
                result.metrics.net_interconnect_area = float(net_area_match.group(1))
            
            cell_area_match = re.search(r'Total cell area:\s*([0-9.]+)', area_data)
            if cell_area_match:
                result.metrics.total_cell_area = float(cell_area_match.group(1))
            
            design_area_match = re.search(r'Design Area:\s*([0-9.]+)', area_data)
            if design_area_match:
                result.metrics.design_area = float(design_area_match.group(1))
    
    def _parse_cell_pin_counts(self, content: str, result: ParsedQoRReport):
        """Parse cell and pin count section"""
        count_section = re.search(r'Cell & Pin Count\s*-+\s*(.*?)\s*-+', content, re.DOTALL)
        
        if count_section:
            count_data = count_section.group(1)
            
            # Extract counts
            pin_count_match = re.search(r'Pin Count:\s*(\d+)', count_data)
            if pin_count_match:
                result.metrics.pin_count = int(pin_count_match.group(1))
            
            hier_cell_match = re.search(r'Hierarchical Cell Count:\s*(\d+)', count_data)
            if hier_cell_match:
                result.metrics.hierarchical_cell_count = int(hier_cell_match.group(1))
            
            leaf_cell_match = re.search(r'Leaf Cell Count:\s*(\d+)', count_data)
            if leaf_cell_match:
                result.metrics.leaf_cell_count = int(leaf_cell_match.group(1))
    
    def _parse_drc_violations(self, content: str, result: ParsedQoRReport):
        """Parse design rule violations section"""
        drc_section = re.search(r'Design Rule Violations\s*-+\s*(.*?)\s*-+', content, re.DOTALL)
        
        if drc_section:
            drc_data = drc_section.group(1)
            
            # Extract DRC metrics
            min_cap_count_match = re.search(r'min_capacitance Count:\s*(\d+)', drc_data)
            if min_cap_count_match:
                result.metrics.min_capacitance_count = int(min_cap_count_match.group(1))
            
            max_trans_count_match = re.search(r'max_transition Count:\s*(\d+)', drc_data)
            if max_trans_count_match:
                result.metrics.max_transition_count = int(max_trans_count_match.group(1))
            
            min_cap_cost_match = re.search(r'min_capacitance Cost:\s*([0-9.]+)', drc_data)
            if min_cap_cost_match:
                result.metrics.min_capacitance_cost = float(min_cap_cost_match.group(1))
            
            max_trans_cost_match = re.search(r'max_transition Cost:\s*([0-9.]+)', drc_data)
            if max_trans_cost_match:
                result.metrics.max_transition_cost = float(max_trans_cost_match.group(1))
            
            total_drc_cost_match = re.search(r'Total DRC Cost:\s*([0-9.]+)', drc_data)
            if total_drc_cost_match:
                result.metrics.total_drc_cost = float(total_drc_cost_match.group(1))


class ReportParserManager:
    """Manager class for handling multiple report types"""
    
    def __init__(self):
        self.power_parser = PowerReportParser()
        self.qor_parser = QoRReportParser()
        self.logger = logging.getLogger(__name__)
    
    def parse_power_report(self, report_content: str) -> ParsedPowerReport:
        """Parse a power report and return structured data"""
        return self.power_parser.parse(report_content)
    
    def parse_qor_report(self, report_content: str) -> ParsedQoRReport:
        """Parse a QoR report and return structured data"""
        return self.qor_parser.parse(report_content)
    
    def parse_report_file(self, file_path: str, report_type: str) -> Dict[str, Any]:
        """Parse a report file and return structured data as dictionary"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            if report_type.lower() == 'power':
                result = self.parse_power_report(content)
            elif report_type.lower() == 'qor':
                result = self.parse_qor_report(content)
            else:
                raise ValueError(f"Unsupported report type: {report_type}")
            
            # Convert to dictionary for easy JSON serialization
            return {
                'report_type': result.report_type,
                'timestamp': result.timestamp.isoformat(),
                'metrics': asdict(result.metrics),
                'parsing_successful': result.parsing_successful,
                'parsing_errors': result.parsing_errors,
                'raw_content_length': len(result.raw_content)
            }
            
        except Exception as e:
            error_msg = f"Failed to parse report file {file_path}: {str(e)}"
            self.logger.error(error_msg)
            return {
                'report_type': report_type,
                'timestamp': datetime.now().isoformat(),
                'metrics': {},
                'parsing_successful': False,
                'parsing_errors': [error_msg],
                'raw_content_length': 0
            }
    
    def parse_reports_batch(self, report_files: Dict[str, str]) -> Dict[str, Any]:
        """Parse multiple report files in batch
        
        Args:
            report_files: Dict mapping report_type to file_path
            
        Returns:
            Dict mapping report_type to parsed results
        """
        results = {}
        
        for report_type, file_path in report_files.items():
            try:
                results[report_type] = self.parse_report_file(file_path, report_type)
                self.logger.info(f"Successfully parsed {report_type} report: {file_path}")
            except Exception as e:
                error_msg = f"Failed to parse {report_type} report {file_path}: {str(e)}"
                self.logger.error(error_msg)
                results[report_type] = {
                    'parsing_successful': False,
                    'parsing_errors': [error_msg]
                }
        
        return results


# Utility functions for easy integration
def parse_power_report_file(file_path: str) -> Dict[str, Any]:
    """Convenience function to parse a single power report file"""
    manager = ReportParserManager()
    return manager.parse_report_file(file_path, 'power')


def parse_qor_report_file(file_path: str) -> Dict[str, Any]:
    """Convenience function to parse a single QoR report file"""
    manager = ReportParserManager()
    return manager.parse_report_file(file_path, 'qor')


def create_report_summary(parsed_reports: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary of key metrics from parsed reports"""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'reports_processed': len(parsed_reports),
        'parsing_success_rate': 0.0,
        'key_metrics': {}
    }
    
    successful_parses = sum(1 for r in parsed_reports.values() if r.get('parsing_successful', False))
    summary['parsing_success_rate'] = successful_parses / len(parsed_reports) if parsed_reports else 0.0
    
    # Extract key metrics for quick reference
    for report_type, report_data in parsed_reports.items():
        if not report_data.get('parsing_successful', False):
            continue
            
        metrics = report_data.get('metrics', {})
        
        if report_type == 'power':
            summary['key_metrics']['total_power'] = metrics.get('total_power', 0.0)
            summary['key_metrics']['leakage_power_pct'] = metrics.get('leakage_power_pct', 0.0)
        
        elif report_type == 'qor':
            summary['key_metrics']['setup_violating_paths'] = metrics.get('setup_violating_paths', 0)
            summary['key_metrics']['hold_violating_paths'] = metrics.get('hold_violating_paths', 0)
            summary['key_metrics']['total_cell_area'] = metrics.get('total_cell_area', 0.0)
            summary['key_metrics']['total_drc_cost'] = metrics.get('total_drc_cost', 0.0)
    
    return summary


def parse_elapsed_time(report_content: str) -> float:
    """
    Extract execution elapsed time from EDA tool report.

    The elapsed time appears in format: "Information: Elapsed time [               15 seconds ]"

    Args:
        report_content: Raw report content as string

    Returns:
        Elapsed time in seconds as float. Returns 0.0 if pattern not found.

    Raises:
        ValueError: If the pattern is found but time value cannot be parsed
    """
    if not report_content:
        return 0.0

    # Match pattern: "Elapsed time [" followed by whitespace, number, and "seconds ]"
    elapsed_match = re.search(r'Elapsed time\s*\[\s*(\d+)\s+seconds\s*\]', report_content)

    if not elapsed_match:
        return 0.0

    try:
        elapsed_time = float(elapsed_match.group(1))
        return elapsed_time
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Failed to parse elapsed time from report: {str(e)}")


def extract_json_from_response(response_text: str) -> str:
    """
    Extract JSON content from LLM response by removing irrelevant text before and after JSON.

    Args:
        response_text: Raw text response from LLM that contains JSON

    Returns:
        Clean JSON string ready for json.loads()

    Raises:
        ValueError: If no JSON braces are found
    """
    if not response_text:
        raise ValueError("Empty response text")

    # Find first { and last }
    first_brace = response_text.find('{')
    last_brace = response_text.rfind('}')

    if first_brace == -1 or last_brace == -1 or first_brace >= last_brace:
        raise ValueError("No valid JSON braces found in response")

    # Extract content between braces (inclusive)
    json_content = response_text[first_brace:last_brace + 1]

    return json_content


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) > 1:
        # Test with provided file
        file_path = sys.argv[1]
        report_type = sys.argv[2] if len(sys.argv) > 2 else 'qor'
        
        manager = ReportParserManager()
        result = manager.parse_report_file(file_path, report_type)
        
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python report_parsers.py <report_file> [power|qor]")
        print("Example: python report_parsers.py reports/report_qor_1.txt qor")
