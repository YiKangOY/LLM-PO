#!/usr/bin/env python3
"""
PT shell command execution for Agent with server/client flow.
"""

import copy
import os
import re
import subprocess
import time
import sys

from datetime import datetime

AGENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Agent"))
if AGENT_ROOT not in sys.path:
    sys.path.append(AGENT_ROOT)

from config_noBoth import BUFFER_LIST, base_path
from report_parsers import parse_elapsed_time
from agent_run_paths_noBoth import AgentRunPaths
from agent_pt_server_noBoth import PTServerManager


def _formulate_tcl_command(tcl_command, iteration):
    """
    Formulate TCL command for execution.
    Mirrors eco_ppa_agent.py command formatting.
    """
    tcl_command = re.sub(r'\[(.*?)\]', lambda m: '{' + ' '.join(m.group(1).split(', ')) + '}', tcl_command)
    if "timing" in tcl_command:
        tcl_command += " -verbose -unfixable_reasons_format text -unfixable_reasons_prefix {}/reports/unfix_timing_{}".format(base_path, iteration)

    if "buffer_insertion" in tcl_command:
        tcl_command += f" -buffer_list {BUFFER_LIST} "
    if "opt_power" in tcl_command:
        tcl_command += " -verbose > {}/reports/fix_power_{}.txt;".format(base_path, iteration)
    if "opt_area" in tcl_command:
        tcl_command += " -verbose > {}/reports/fix_area_{}.txt;".format(base_path, iteration)
    elif "opt_timing" in tcl_command:
        tcl_command += " > {}/reports/fix_timing_{}.txt;".format(base_path, iteration)

    if "timing" not in tcl_command and "-area_cap" in tcl_command:
        area_cap_pattern = r'-area_cap\s+\S+'
        tcl_command = re.sub(area_cap_pattern, '', tcl_command).strip()

    vth_lib_mapping = {
        'RVT': 'tcbn28hpcplusbwp30p140tt0p9v85c',
        'LVT': 'tcbn28hpcplusbwp30p140lvttt0p9v85c',
        'HVT': 'tcbn28hpcplusbwp30p140hvttt0p9v85c'
    }
    all_vth_types = set(vth_lib_mapping.keys())

    vth_pattern_braces = r'-Vth\s+\{([^}]+)\}'
    vth_pattern_no_braces = r'-Vth\s+([A-Z]+(?:\s+[A-Z]+)*)'

    vth_match = re.search(vth_pattern_braces, tcl_command)
    if vth_match:
        vth_values_str = vth_match.group(1)
        present_vth = set(vth_values_str.split())
        processed_string = re.sub(vth_pattern_braces, '', tcl_command).strip()
    else:
        vth_match = re.search(vth_pattern_no_braces, tcl_command)
        if vth_match:
            vth_values_str = vth_match.group(1)
            present_vth = set(vth_values_str.split())
            processed_string = re.sub(vth_pattern_no_braces, '', tcl_command).strip()
        else:
            present_vth = set()
            processed_string = tcl_command.strip()

    missing_vth = all_vth_types - present_vth

    dont_use_commands = []
    if missing_vth != all_vth_types:
        for vth in sorted(missing_vth):
            lib_name = vth_lib_mapping[vth]
            dont_use_cmd = f"set_target_library_subset -top -dont_use [get_lib_cells {lib_name}/*];"
            dont_use_commands.append(dont_use_cmd)

    area_cap_prefix = ""
    area_cap_suffix = ""

    if "opt_timing" in processed_string and "-area_cap" in processed_string:
        area_cap_pattern = r'-area_cap\s+(\S+)'
        area_cap_match = re.search(area_cap_pattern, processed_string)
        if area_cap_match:
            area_cap_value = area_cap_match.group(1)
            area_cap_prefix = f"set_app_var eco_alternative_area_ratio_threshold {area_cap_value};\n"
            area_cap_suffix = "\nset_app_var eco_alternative_area_ratio_threshold 2;"
            processed_string = re.sub(area_cap_pattern, '', processed_string).strip()

    if dont_use_commands:
        result = ' '.join(dont_use_commands) + ' ' + processed_string
    else:
        result = processed_string
    if 'opt_area' in tcl_command:
        result = result.replace('opt_area', 'opt_power')
    result = area_cap_prefix + result + area_cap_suffix
    result += '\n remove_target_library_subset -top; \n'
    return result


def execute_tcl_command(tcl_command, command_type, iteration, round_index, run_paths=None, pt_server=None):
    """
    Execute TCL command via pt_shell server/client flow.
    """
    ori_tcl_command = copy.deepcopy(tcl_command)
    formatted_command = _formulate_tcl_command(tcl_command, iteration)

    if run_paths is None:
        run_paths = AgentRunPaths(round_index)
    run_paths.prepare()

    use_server = pt_server is not None and run_paths.use_pt_server
    server_ready = False
    if use_server:
        server_ready = pt_server.ensure_running()

    template_path = os.path.join(os.path.dirname(__file__), 'pt_server_run_script.tcl')

    with open(template_path, 'r') as f:
        template_content = f.read()

    filled_content = template_content.replace('{}', formatted_command)
    filled_content = filled_content.replace('${i}', str(iteration))
    filled_content = filled_content.replace('${i+1}', str(iteration + 1))

    script_path = os.path.join(run_paths.scripts_dir, f'Run_scripts_{iteration}.tcl')
    with open(script_path, 'w') as f:
        f.write(filled_content)

    server_response = None
    if server_ready:
        start_time = time.perf_counter()
        success, server_response = pt_server.run_script(script_path)
        end_time = time.perf_counter()
        if success:
            execution_time = _parse_execution_time(command_type, iteration)
            if execution_time == 0.0:
                execution_time = end_time - start_time
            return {
                'success': True,
                'message': f'Successfully executed {command_type} command',
                'tcl_command': ori_tcl_command,
                'execution_time': execution_time,
                'command_type': command_type,
                'output': server_response,
                'exit_code': 0,
                'script_path': script_path,
                'log_path': None,
                'timestamp': datetime.now().isoformat()
            }

    log_path = os.path.join(run_paths.logs_dir, f'pt_{iteration}.log')
    command = f'cd {base_path} && pt_shell -f {script_path} | tee {log_path}'
    exit_code = subprocess.call(["bash", "-lc", command])

    output = ""
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            output = f.read()

    execution_time = 0.0
    if exit_code == 0:
        execution_time = _parse_execution_time(command_type, iteration)

    success = exit_code == 0
    message = f'Successfully executed {command_type} command' if success else f'Command execution failed with exit code {exit_code}'

    return {
        'success': success,
        'message': message,
        'tcl_command': ori_tcl_command,
        'execution_time': execution_time,
        'command_type': command_type,
        'output': output[:2000] if len(output) > 2000 else output,
        'exit_code': exit_code,
        'script_path': script_path,
        'log_path': log_path,
        'timestamp': datetime.now().isoformat()
    }


def _parse_execution_time(command_type, iteration):
    report_file = None
    if 'timing' in command_type.lower():
        report_file = os.path.join(base_path, f'reports/fix_timing_{iteration}.txt')
    elif 'power' in command_type.lower():
        report_file = os.path.join(base_path, f'reports/fix_power_{iteration}.txt')
    elif 'area' in command_type.lower():
        report_file = os.path.join(base_path, f'reports/fix_area_{iteration}.txt')

    if report_file is None:
        return 0.0
    if not os.path.exists(report_file):
        return 0.0

    with open(report_file, 'r') as f:
        report_content = f.read()
    return parse_elapsed_time(report_content)


def build_pt_server(round_index, run_paths=None):
    if run_paths is None:
        run_paths = AgentRunPaths(round_index)
    run_paths.prepare()
    return PTServerManager(run_paths, enable=run_paths.use_pt_server)
