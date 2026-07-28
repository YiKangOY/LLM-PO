#!/usr/bin/env python3
"""
Simple RL Command Executor
"""

import os
import subprocess
import sys
import re
import copy
import shutil
import time
import socket
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Ensure the sibling Agent package is importable when running from the RL folder.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)
_AGENT_ROOT = os.path.join(_ROOT, "Agent")
if _AGENT_ROOT not in sys.path:
    sys.path.append(_AGENT_ROOT)

from Agent.configs import BUFFER_LIST
from Agent.report_parsers import parse_elapsed_time
from rl_config import ENV_CONFIG, load_config

_RESERVED_PORTS = set()


@dataclass
class RunPaths:
    """
    Holds per-environment paths so parallel runs do not overwrite each other.
    """
    workspace: str  # Where pt_shell should be executed (design workspace)
    run_dir: str    # Root directory for this environment's artifacts
    env_id: int = 0
    reports_dir: Optional[str] = None
    logs_dir: Optional[str] = None
    scripts_dir: Optional[str] = None
    session_prefix: Optional[str] = None
    server_host: Optional[str] = None
    server_port: Optional[int] = None
    use_pt_server: bool = ENV_CONFIG.get("use_pt_server", True)

    def __post_init__(self):
        self.run_dir = os.path.abspath(self.run_dir)
        self.reports_dir = self.reports_dir or os.path.join(self.run_dir, "reports")
        self.logs_dir = self.logs_dir or os.path.join(self.run_dir, "logs")
        self.scripts_dir = self.scripts_dir or os.path.join(self.run_dir, "run_scripts")
        self.session_prefix = self.session_prefix or f"eco_session_env{self.env_id}"
        self.server_host = self.server_host or ENV_CONFIG.get("pt_server_host", "127.0.0.1")
        if self.server_port is None:
            base_port = ENV_CONFIG.get("pt_server_base_port", 9009)
            stride = ENV_CONFIG.get("pt_server_port_stride", 1)
            self.server_port = base_port + self.env_id * stride

    def prepare(self, skip_if_exists: bool = False):
        """Ensure directories exist and seed baseline artifacts when available."""
        run_dir_exists = os.path.exists(self.run_dir)
        for path in [self.run_dir, self.reports_dir, self.logs_dir, self.scripts_dir]:
            os.makedirs(path, exist_ok=True)
        if skip_if_exists and run_dir_exists:
            return
        self._copy_baseline_reports()
        self._copy_baseline_session()

    def session_file(self, iteration: int) -> str:
        """Return the session file path for a given iteration."""
        return os.path.join(self.run_dir, f"{self.session_prefix}_{iteration}")

    def ensure_server_port(self) -> int:
        """
        Try to reserve the pre-allocated port; fall back to a new one if needed.
        This should be called right before starting pt_shell.
        """
        if self.server_port is None:
            base_port = ENV_CONFIG.get("pt_server_base_port", 9009)
            stride = ENV_CONFIG.get("pt_server_port_stride", 1)
            self.server_port = base_port + self.env_id * stride

        if self._try_reserve_port(self.server_port):
            return self.server_port

        self.server_port = self._find_available_port(self.server_port + 1)
        return self.server_port

    def _find_available_port(self, preferred_port: int) -> int:
        """
        Choose an available port starting from the preferred one, incrementing
        until we can bind.
        """
        port = preferred_port
        while True:
            if self._try_reserve_port(port):
                return port
            port += 1

    def _try_reserve_port(self, port: int) -> bool:
        host = self.server_host or "127.0.0.1"
        if port in _RESERVED_PORTS:
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
            except OSError:
                return False
        _RESERVED_PORTS.add(port)
        return True

    def release_port(self):
        if self.server_port in _RESERVED_PORTS:
            _RESERVED_PORTS.remove(self.server_port)

    def _copy_baseline_reports(self):
        """Seed report_qor_0.txt and report_power_0.txt; fail if missing."""
        source_reports_dir = os.path.join(self.workspace, "reports")
        for report_name in ("report_qor_0.txt", "report_power_0.txt"):
            src = os.path.join(source_reports_dir, report_name)
            dst = os.path.join(self.reports_dir, report_name)
            if not os.path.exists(src):
                raise FileNotFoundError(
                    f"Baseline report not found: {src}"
                )
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)

    def _copy_baseline_session(self):
        """
        Seed the initial eco_session_0 if it exists.
        This function is synchronous and blocking.
        """
        src_session = os.path.join(self.workspace, "eco_session_0")
        dst_session = self.session_file(0)

        # 1. baseline 必须存在
        if not os.path.isdir(src_session):
            raise FileNotFoundError(
                f"Baseline session directory not found: {src_session}"
            )

        # 2. 目标目录已存在时，先清理（避免 copytree 报错）
        if os.path.exists(dst_session):
            shutil.rmtree(dst_session)

        # 3. 同步、阻塞复制
        shutil.copytree(src_session, dst_session)


def _formulate_tcl_command(
    tcl_command: str,
    iteration: int,
    run_paths: Optional[RunPaths] = None
) -> str:
    """
    Formulate TCL command for execution by processing VTH, area_cap, and other options.
    This mirrors the logic in eco_ppa_agent.py:_formulate_tcl_command

    Args:
        tcl_command: Raw TCL command from action decoder
        iteration: Current iteration number

    Returns:
        Formatted TCL command ready for execution
    """
    # Fill in the command and iteration variables
    # Use a different approach - replace the {} with command first, then handle ${} variables
    tcl_command = re.sub(r'\[(.*?)\]', lambda m: '{' + ' '.join(m.group(1).split(', ')) + '}', tcl_command)

    if not ENV_CONFIG:
        load_config()
    workspace_path = ENV_CONFIG["base_path"]
    reports_dir = run_paths.reports_dir if run_paths else os.path.join(workspace_path, "reports")

    if "timing" in tcl_command:
        tcl_command += " -verbose -unfixable_reasons_format text -unfixable_reasons_prefix {}/unfix_timing_{}".format(reports_dir, iteration)

    # process buffer lists
    if "buffer_insertion" in tcl_command:
        tcl_command += f" -buffer_list {BUFFER_LIST} "

    # add redirecting output
    if "opt_power" in tcl_command:
        tcl_command += " -verbose > {}/fix_power_{}.txt;".format(reports_dir, iteration)
    if "opt_area" in tcl_command:
        tcl_command.replace("opt_area", "opt_power")
        tcl_command += " -verbose > {}/fix_area_{}.txt;".format(reports_dir, iteration)
    elif "opt_timing" in tcl_command:
        tcl_command += " > {}/fix_timing_{}.txt;".format(reports_dir, iteration)

    vth_lib_mapping = {
        'RVT': 'tcbn28hpcplusbwp30p140tt0p9v85c',
        'LVT': 'tcbn28hpcplusbwp30p140lvttt0p9v85c',
        'HVT': 'tcbn28hpcplusbwp30p140hvttt0p9v85c'
    }
    all_vth_types = set(vth_lib_mapping.keys())

    # Extract VTH values from the string using regex (handle both -Vth {LVT RVT} and -Vth LVT RVT formats)
    vth_pattern_braces = r'-Vth\s+\{([^}]+)\}'
    vth_pattern_no_braces = r'-Vth\s+([A-Z]+(?:\s+[A-Z]+)*)'

    vth_match = re.search(vth_pattern_braces, tcl_command)
    if vth_match:
        # Handle -Vth {LVT RVT} format
        vth_values_str = vth_match.group(1)
        present_vth = set(vth_values_str.split())
        processed_string = re.sub(vth_pattern_braces, '', tcl_command).strip()
    else:
        # Try -Vth LVT RVT format (without braces)
        vth_match = re.search(vth_pattern_no_braces, tcl_command)
        if vth_match:
            vth_values_str = vth_match.group(1)
            present_vth = set(vth_values_str.split())
            processed_string = re.sub(vth_pattern_no_braces, '', tcl_command).strip()
        else:
            # If no -Vth found, assume all VTH types are missing
            present_vth = set()
            processed_string = tcl_command.strip()

    # Find missing VTH types
    missing_vth = all_vth_types - present_vth

    # Generate dont_use commands for missing VTH types using actual library names
    # If all VTHs are missing, don't set any dont_use commands (all VTHs are usable)
    dont_use_commands = []
    if missing_vth != all_vth_types:  # Only set dont_use if not all VTHs are missing
        for vth in sorted(missing_vth):  # Sort for consistent output
            lib_name = vth_lib_mapping[vth]
            dont_use_cmd = f"set_target_library_subset -top -dont_use [get_lib_cells {lib_name}/*];"
            dont_use_commands.append(dont_use_cmd)

    # Handle -area_cap option for timing commands
    area_cap_prefix = ""
    area_cap_suffix = ""

    if "opt_timing" in processed_string and "-area_cap" in processed_string:
        # Extract area limit value using regex
        area_cap_pattern = r'-area_cap\s+(\S+)'
        area_cap_match = re.search(area_cap_pattern, processed_string)
        if area_cap_match:
            area_cap_value = area_cap_match.group(1)
            area_cap_prefix = f"set_app_var eco_alternative_area_ratio_threshold {area_cap_value};\n"
            area_cap_suffix = "\nset_app_var eco_alternative_area_ratio_threshold 2;"
            # Remove the -area_cap option from the processed string
            processed_string = re.sub(area_cap_pattern, '', processed_string).strip()

    # Combine dont_use commands with the processed string
    if dont_use_commands:
        result = ' '.join(dont_use_commands) + ' ' + processed_string
    else:
        result = processed_string
    if 'opt_area' in tcl_command:
        result = result.replace('opt_area', 'opt_power')
    # Add area limit commands if needed
    result = area_cap_prefix + result + area_cap_suffix
    result += '\n remove_target_library_subset -top; \n'
    return result


def execute_tcl_command(
    tcl_command: str,
    iteration: int,
    command_type: str = "timing",
    run_paths: Optional[RunPaths] = None,
    pt_server: Optional[Any] = None,
    preformatted: bool = False,
) -> Dict[str, Any]:
    """
    Execute TCL command using pt_shell

    Args:
        tcl_command: TCL command to execute
        iteration: Current iteration
        command_type: "timing", "power", or "area"
        run_paths: Paths for the current env
        pt_server: Optional PTServerManager to reuse a persistent pt_shell
            instance. When provided and available, the command will be sent via
            RUN <script> instead of spawning pt_shell per call.
        preformatted: When True, skip the internal formatting step and execute
            the provided command as-is.

    Returns:
        Execution result dict
    """
    if not ENV_CONFIG:
        load_config()
    # Store original command
    ori_tcl_command = copy.deepcopy(tcl_command)

    # Formulate the command (process -area_cap, VTH, etc.)
    if preformatted:
        formatted_command = tcl_command
    else:
        formatted_command = _formulate_tcl_command(tcl_command, iteration, run_paths)

    workspace_path = run_paths.workspace if run_paths else ENV_CONFIG["base_path"]
    reports_dir = run_paths.reports_dir if run_paths else os.path.join(workspace_path, "reports")
    logs_dir = run_paths.logs_dir if run_paths else os.path.join(workspace_path, "logs")
    scripts_dir = run_paths.scripts_dir if run_paths else workspace_path
    session_prefix = run_paths.session_prefix if run_paths else "eco_session"

    # Ensure directories exist for this run
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)

    # Read template (server/client flow uses the lightweight run script format)
    template_path = os.path.join(os.path.dirname(__file__), 'example_env', 'Run_scripts_rl_env1_{iteration}.tcl')
    if not os.path.exists(template_path):
        fallback_path = os.path.join(os.path.dirname(__file__), '..', 'EDA_scripts', 'execute_command.tcl')
        if os.path.exists(fallback_path):
            template_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Template script not found: {template_path} (fallback missing: {fallback_path})"
            )
    with open(template_path, 'r') as f:
        template = f.read()

    # Fill template
    script_content = template.replace('{}', formatted_command)
    # Redirect baseline report outputs to the per-env reports dir without
    # rewriting any existing absolute paths inside the command itself.
    if run_paths:
        script_content = script_content.replace(
            'reports/report_qor_${i+1}.txt',
            os.path.join(reports_dir, 'report_qor_${i+1}.txt')
        )
        script_content = script_content.replace(
            'reports/report_power_${i+1}.txt',
            os.path.join(reports_dir, 'report_power_${i+1}.txt')
        )
    script_content = script_content.replace('${i}', str(iteration))
    script_content = script_content.replace('${i+1}', str(iteration + 1))

    script_filename = f'Run_scripts_rl_{iteration}.tcl'
    if run_paths:
        script_filename = f'Run_scripts_rl_env{run_paths.env_id}_{iteration}.tcl'
    script_path = os.path.join(scripts_dir, script_filename)

    with open(script_path, 'w') as f:
        f.write(script_content)

    prefer_server = (
        pt_server is not None
        and getattr(run_paths, "use_pt_server", ENV_CONFIG.get("use_pt_server", False))
    )
    server_response = None
    use_server = prefer_server and pt_server.ensure_running()
    if use_server:
        wall_start = time.perf_counter()
        success, server_response = pt_server.run_script(script_path)
        wall_duration = time.perf_counter() - wall_start
        if success:
            return {
                "success": True,
                "exit_code": 0,
                "execution_time": wall_duration,
                "log_path": None,
                "script_path": script_path,
                "raw_command": ori_tcl_command,
                "formatted_command": formatted_command,
                "reports_dir": reports_dir,
                "command_type": command_type,
                "server_response": server_response,
            }
        # Fall back to legacy per-command execution if server execution fails

    log_filename = f'pt_rl_{iteration}.log'
    if run_paths:
        log_filename = f'pt_rl_env{run_paths.env_id}_{iteration}.log'
    log_path = os.path.join(logs_dir, log_filename)

    # === 使用 pt_shell 自带的 log ===
    wall_start = time.perf_counter()
    proc = subprocess.run(
        [
            "pt_shell",
            "-file", script_path,
            "-output_log_file", log_path,
        ],
        cwd=workspace_path,
    )
    wall_duration = time.perf_counter() - wall_start

    exit_code = proc.returncode

    # Prefer measured wall time; fall back to parsed log time if available
    execution_time = wall_duration
    if os.path.exists(log_path):
        parsed_time = parse_elapsed_time(log_path)
        if parsed_time is not None:
            execution_time = parsed_time

    return {
        "success": exit_code == 0,
        "exit_code": exit_code,
        "execution_time": execution_time if execution_time is not None else 0.0,
        "log_path": log_path,
        "script_path": script_path,
        "raw_command": ori_tcl_command,
        "formatted_command": formatted_command,
        "reports_dir": reports_dir,
        "command_type": command_type,
        "server_response": server_response,
    }
