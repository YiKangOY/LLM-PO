#!/usr/bin/env python3
"""
ECO Agent System with pt_shell server/client execution (RAG enabled, no reflection agents).
"""

import argparse
import os
import pickle
import json
import shutil
from datetime import datetime
import sys
import time

AGENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Agent"))
if AGENT_ROOT not in sys.path:
    sys.path.append(AGENT_ROOT)

import config_noReflection as configs
# Ensure Agent modules importing "configs" resolve to the standalone config.
sys.modules["configs"] = configs
import rag_helpers_noReflection as rag_helpers_no_reflection
sys.modules["rag_helpers"] = rag_helpers_no_reflection

import agent_command_executor_noReflection as agent_command_executor
import agent_logs
import agent_run_paths_noReflection as agent_run_paths
import eco_ppa_agent
from eco_ppa_agent import ECOLangChainSystem
from eco_ppa_agent import load_reports_for_iteration
from eco_ppa_agent import TOOL_USING
from eco_ppa_agent import create_openai_llm
from eco_ppa_trace import TraceMemory
from agent_logs import LLMResponse
from utils import ECOType
from agent_command_executor_noReflection import execute_tcl_command, build_pt_server
from agent_run_paths_noReflection import AgentRunPaths

CONFIDENTIAL_UNFIXABLE = "removed due to confidential reasons."


class ECOLangChainSystemPTServerNoReflection(ECOLangChainSystem):
    """
    ECOLangChainSystem that executes commands via persistent pt_shell server.
    """

    def __init__(self, total_runtime_budget=3600, log_file="eco_agent_responses_langchain.json"):
        super().__init__(total_runtime_budget=total_runtime_budget, log_file=log_file)
        self.llm_summary = create_openai_llm("gpt-5-mini", configs.USE_OPENAI_REASONING)
        self.llm_area = create_openai_llm("gpt-5-mini", configs.USE_OPENAI_REASONING)
        self.llm_timing = create_openai_llm("gpt-5-mini", configs.USE_OPENAI_REASONING)
        self.llm_power = create_openai_llm("gpt-5-mini", configs.USE_OPENAI_REASONING)
        self._pt_run_paths = None
        self._pt_server = None
        self._pt_server_round = None
        self._last_llm_interaction_index = 0

    def _get_run_paths(self):
        if self._pt_run_paths is None or self._pt_run_paths.round_index != self.round_index:
            self._pt_run_paths = AgentRunPaths(self.round_index)
            self._pt_run_paths.prepare()
        return self._pt_run_paths

    def _update_unfixable_history(self, iteration, unfixable_reasons):
        super()._update_unfixable_history(iteration, CONFIDENTIAL_UNFIXABLE)

    def _extract_fix_unfix_reports(self, reports):
        fix_unfix_data = super()._extract_fix_unfix_reports(reports)
        for key in ("timing_unfix", "area_unfix", "power_unfix"):
            if key in fix_unfix_data:
                fix_unfix_data[key] = CONFIDENTIAL_UNFIXABLE
        return fix_unfix_data

    def _debug_print(self, message):
        print("[PTSERVER][round {}][iter {}] {}".format(
            self.round_index,
            self.iteration_count,
            message
        ), flush=True)

    def _get_pt_server(self, run_paths):
        if self._pt_server is None:
            self._debug_print("Building PT server manager")
            self._pt_server = build_pt_server(self.round_index, run_paths=run_paths)
        return self._pt_server

    def _execute_tcl_command(self, tcl_command, command_type, iteration):
        run_paths = self._get_run_paths()
        pt_server = self._get_pt_server(run_paths)
        self._debug_print("Executing TCL command for {}: {}".format(command_type, tcl_command))
        start_time = time.time()
        execution_result = execute_tcl_command(
            tcl_command,
            command_type,
            iteration,
            self.round_index,
            run_paths=run_paths,
            pt_server=pt_server
        )
        execution_result["wall_time"] = time.time() - start_time
        self._debug_print("TCL command finished with success={} exit_code={} wall_time={:.2f}s".format(
            execution_result["success"],
            execution_result["exit_code"],
            execution_result["wall_time"]
        ))
        return execution_result

    def start_round(self):
        if self._pt_server is None or self._pt_server_round != self.round_index:
            if self._pt_server is not None:
                self._debug_print("Shutting down stale PT server before starting new round")
                self._pt_server.shutdown()
            self._pt_run_paths = AgentRunPaths(self.round_index)
            self._pt_run_paths.prepare()
            self._debug_print("Preparing PT server assets under {}".format(self._pt_run_paths.run_dir))
            self._pt_server = build_pt_server(self.round_index, run_paths=self._pt_run_paths)
            self._debug_print("Ensuring PT server is running")
            self._pt_server.ensure_running()
            self._pt_server_round = self.round_index
            self._debug_print("PT server ready on {}:{}".format(
                self._pt_run_paths.server_host,
                self._pt_run_paths.server_port
            ))

    def _log_llm_call_runtimes(self):
        interactions = self.logger.llm_interactions
        new_interactions = interactions[self._last_llm_interaction_index:]
        for interaction in new_interactions:
            response = LLMResponse(
                agent_type="LLMCallRuntime",
                timestamp=interaction.timestamp,
                input_data={
                    "llm_agent_type": interaction.agent_type,
                    "model_name": interaction.model_name,
                    "round_index": interaction.round_index
                },
                output_data={"runtime_seconds": interaction.processing_time},
                processing_time=interaction.processing_time,
                iteration=interaction.iteration
            )
            self.logger.log_response(response)
        self._last_llm_interaction_index = len(interactions)

    def run_iteration(self, objectives, reports):
        self.iteration_count += 1
        iteration_start = time.time()
        self.current_iteration_llm_time = 0.0

        elapsed_runtime_start = time.time() - self.start_time
        remaining_budget_start = self.total_runtime_budget - elapsed_runtime_start

        self.logger.logger.info(f"[Round {self.round_index}] Starting ECO iteration {self.iteration_count}")
        self._debug_print("run_iteration entered")
        self._debug_print("Reports available: {}".format(sorted(reports.keys())))

        if self.persistent_state is None:
            self.logger.logger.info(f"[Round {self.round_index}] Creating initial ECO state for first iteration")
            self._debug_print("Creating initial persistent state")
            self.persistent_state = eco_ppa_agent.ECOState(
                iteration=self.iteration_count,
                reports=reports,
                unified_analysis={},
                command_proposals={},
                selected_command={},
                execution_result={},
                elapsed_runtime=elapsed_runtime_start,
                remaining_budget=remaining_budget_start,
                messages=[],
                system_status="running",
                selected_route="",
                rag_queries=[],
                rag_retrieved_content="",
                optimization_strategy="",
                objectives=objectives
            )
        else:
            self.logger.logger.info(f"[Round {self.round_index}] Updating persistent state for iteration {self.iteration_count}")
            self._debug_print("Updating persistent state")

            if 'iteration_history' not in self.persistent_state['unified_analysis']:
                self.persistent_state['unified_analysis']['iteration_history'] = {}

            prev_iteration = self.iteration_count - 1
            if prev_iteration >= 0:
                self._debug_print("Archiving iteration history for iteration {}".format(prev_iteration))
                self.persistent_state['unified_analysis']['iteration_history'][f'iteration_{prev_iteration}'] = {
                    'selected_command': self.persistent_state.get('selected_command', {}),
                    'execution_result': self.persistent_state.get('execution_result', {}),
                    'reports_summary': self._summarize_reports(self.persistent_state.get('reports', {}))
                }

            preserved_messages = self.persistent_state.get('messages', [])

            self.persistent_state.update({
                'iteration': self.iteration_count,
                'reports': reports,
                'elapsed_runtime': elapsed_runtime_start,
                'remaining_budget': remaining_budget_start,
                'system_status': "running",
                'messages': preserved_messages,
                'objectives': objectives
            })

        current_objectives = self.persistent_state['objectives']
        self._debug_print("Invoking LangGraph workflow")
        final_state = self.workflow.invoke(self.persistent_state)
        self._debug_print("LangGraph workflow returned")
        final_state['objectives'] = current_objectives

        self.persistent_state = final_state

        eda_execution_time = final_state["execution_result"].get('execution_time', 0.0)
        iteration_runtime = self.current_iteration_llm_time + eda_execution_time

        self.iteration_llm_times[self.iteration_count] = self.current_iteration_llm_time

        total_elapsed = sum(self.iteration_llm_times.values()) + sum(
            self.design_state_history[i].get('actual_execution_time', 0)
            for i in range(len(self.design_state_history) - 1)
        )
        total_elapsed += eda_execution_time

        remaining_budget_final = self.total_runtime_budget - total_elapsed

        self.persistent_state['elapsed_runtime'] = total_elapsed
        self.persistent_state['remaining_budget'] = remaining_budget_final

        result = {
            "iteration": self.iteration_count,
            "unified_analysis": final_state["unified_analysis"],
            "command_proposals": final_state["command_proposals"],
            "selected_command": final_state["selected_command"],
            "execution_result": final_state["execution_result"],
            "iteration_time": time.time() - iteration_start,
            "iteration_runtime": iteration_runtime,
            "llm_time": self.current_iteration_llm_time,
            "eda_time": eda_execution_time,
            "total_elapsed": total_elapsed,
            "remaining_budget": remaining_budget_final,
            "system_status": "healthy" if remaining_budget_final > 0 else "budget_exceeded"
        }

        self._log_llm_call_runtimes()
        self.logger.logger.info(
            f"[Round {self.round_index}] ECO iteration {self.iteration_count} completed: "
            f"LLM={self.current_iteration_llm_time:.1f}s, "
            f"EDA={eda_execution_time:.1f}s, "
            f"Total={iteration_runtime:.1f}s"
        )
        self._debug_print("run_iteration completed")
        return result

    def _analyze_reports_node(self, state):
        self._debug_print("Node analyze_reports start")
        result = super()._analyze_reports_node(state)
        self._debug_print("Node analyze_reports end route={} strategy={}".format(
            result["selected_route"],
            result["optimization_strategy"]
        ))
        return result

    def _generate_timing_command_node(self, state):
        self._debug_print("Node generate_timing_command start")
        result = super()._generate_timing_command_node(state)
        self._debug_print("Node generate_timing_command end command={}".format(
            result["selected_command"]["tcl_command"]
        ))
        return result

    def _generate_power_command_node(self, state):
        self._debug_print("Node generate_power_command start")
        result = super()._generate_power_command_node(state)
        self._debug_print("Node generate_power_command end command={}".format(
            result["selected_command"]["tcl_command"]
        ))
        return result

    def _generate_area_command_node(self, state):
        self._debug_print("Node generate_area_command start")
        result = super()._generate_area_command_node(state)
        self._debug_print("Node generate_area_command end command={}".format(
            result["selected_command"]["tcl_command"]
        ))
        return result

    def _execute_command_node(self, state):
        self._debug_print("Node execute_command start")
        result = super()._execute_command_node(state)
        execution_result = result["execution_result"]
        self._debug_print("Node execute_command end success={} exit_code={}".format(
            execution_result["success"],
            execution_result["exit_code"]
        ))
        return result


    def _build_timing_user_prompt(self, state):
        """Build user prompt with current design state, optimization history, and unfixable reasons (removed due to confidential reasons)"""
        user_input_parts = []
        objectives = state["objectives"]
        user_input_parts.append("**OBJECTIVES:**")
        user_input_parts.append(objectives)
        user_input_parts.append("")

        current_hist = self.design_state_history[-1] if self.design_state_history else None
        if current_hist:
            current_state_data = current_hist['design_state']
            remaining_iterations = self._calculate_remaining_iterations(current_hist['iteration'])
            user_input_parts.append("**CURRENT DESIGN STATE:**")
            user_input_parts.append(
                f"Iteration: {current_hist['iteration']}, Remaining iterations: {remaining_iterations}"
            )
            user_input_parts.append(f"  - Setup violations: {current_state_data['timing']['setup_violating_paths']} paths")
            user_input_parts.append(f"  - Hold violations: {current_state_data['timing']['hold_violating_paths']} paths")
            user_input_parts.append(f"  - Setup critical slack: {current_state_data['timing']['setup_critical_path_slack']}")
            user_input_parts.append(f"  - Hold critical slack: {current_state_data['timing']['hold_critical_path_slack']}")
            user_input_parts.append(f"  - Setup total negative slack: {current_state_data['timing']['setup_total_negative_slack']}")
            user_input_parts.append(f"  - Hold total negative slack: {current_state_data['timing']['hold_total_negative_slack']}")
            user_input_parts.append(f"  - Total power: {current_state_data['power']['total_power']:.3e}W")
            user_input_parts.append(f"  - Design area: {current_state_data['area']['design_area']} um^2")
            user_input_parts.append(f"  - Iteration budget: {self.iteration_budget}")
            user_input_parts.append(f"  - Remaining iterations: {remaining_iterations}")
            user_input_parts.append("")

        user_input_parts.append("**OPTIMIZATION HISTORY:**")
        if len(self.design_state_history) <= 1:
            user_input_parts.append("No previous optimization history available.")
        elif len(self.design_state_history) > 1:
            user_input_parts.append("Optimization history listed in chronological order.")

            hist_iterations = []
            hist_setup_violating_paths = []
            hist_hold_violating_paths = []
            hist_setup_critical_slacks = []
            hist_hold_critical_slacks = []
            hist_setup_total_negative_slacks = []
            hist_hold_total_negative_slacks = []
            hist_total_powers = []
            hist_design_areas = []
            hist_remaining_iterations = []
            hist_execution_times = []
            hist_executed_commands = []

            for hist_entry in self.design_state_history[:-1]:
                hist_iteration = hist_entry['iteration']
                hist_state = hist_entry['design_state']

                hist_iterations.append(str(hist_iteration))
                hist_setup_violating_paths.append(str(hist_state['timing']['setup_violating_paths']))
                hist_hold_violating_paths.append(str(hist_state['timing']['hold_violating_paths']))
                hist_setup_critical_slacks.append(str(hist_state['timing']['setup_critical_path_slack']))
                hist_hold_critical_slacks.append(str(hist_state['timing']['hold_critical_path_slack']))
                hist_setup_total_negative_slacks.append(str(hist_state['timing']['setup_total_negative_slack']))
                hist_hold_total_negative_slacks.append(str(hist_state['timing']['hold_total_negative_slack']))
                hist_total_powers.append(str(hist_state['power']['total_power']))
                hist_design_areas.append(str(hist_state['area']['design_area']))
                hist_remaining_iterations.append(str(self._calculate_remaining_iterations(hist_iteration)))
                hist_execution_times.append(f"{hist_entry['actual_execution_time']:.0f}s")
                hist_executed_commands.append(hist_entry['executed_command'].replace('{', '{{').replace('}', '}}'))

            hist_parts = [
                f"Iterations {hist_iterations}:",
                f"- Setup violations: {hist_setup_violating_paths} paths",
                f"- Hold violations: {hist_hold_violating_paths} paths",
                f"- Setup critical slack: {hist_setup_critical_slacks} ps",
                f"- Hold critical slack: {hist_hold_critical_slacks} ps",
                f"- Setup total negative slack: {hist_setup_total_negative_slacks} ps",
                f"- Hold total negative slack: {hist_hold_total_negative_slacks} ps",
                f"- Total power: {hist_total_powers}W",
                f"- Design area: {hist_design_areas} um^2",
                f"- Remaining iterations: {hist_remaining_iterations}",
                f"- Actual execution times: {hist_execution_times} s",
                f"- Executed commands: {hist_executed_commands}"
            ]
            user_input_parts.append("\t".join(hist_parts))

        if self.unfixable_analysis_history:
            user_input_parts.append("**UNFIXABLE REASON HISTORY:** removed due to confidential reasons.")
            user_input_parts.append("")
        return "\n".join(user_input_parts)

    def _build_power_user_prompt(self, state):
        """Build user prompt with current design state, optimization history, and unfixable reasons (removed due to confidential reasons)"""
        user_input_parts = []
        objectives = state["objectives"]
        user_input_parts.append("**OBJECTIVES:**")
        user_input_parts.append(objectives)
        user_input_parts.append("")

        current_hist = self.design_state_history[-1] if self.design_state_history else None
        if current_hist:
            current_state_data = current_hist['design_state']
            remaining_iterations = self._calculate_remaining_iterations(current_hist['iteration'])

            current_state_parts = [
                f"Iteration: {current_hist['iteration']}, Remaining iterations: {remaining_iterations}",
                f"- Total power: {current_state_data['power']['total_power']:.3e}W",
                f"- Clock network power: {current_state_data['power']['clock_tree_power']:.3e}W",
                f"- Register power: {current_state_data['power']['register_power']:.3e}W",
                f"- Combinational power: {current_state_data['power']['combinational_power']:.3e}W",
                f"- Leakage power: {current_state_data['power']['leakage_power']:.3e}W",
                f"- Dynamic power: {current_state_data['power']['dynamic_power']:.3e}W",
                f"- Design area: {current_state_data['area']['design_area']:.4f} um^2",
                f"- Iteration budget: {self.iteration_budget}",
                f"- Remaining iterations: {remaining_iterations}"
            ]

            user_input_parts.append("**CURRENT DESIGN STATE:**\t" + "\t".join(current_state_parts))
            user_input_parts.append("")

        user_input_parts.append("**OPTIMIZATION HISTORY:**")
        if len(self.design_state_history) <= 1:
            user_input_parts.append("No previous optimization history available.")
        elif len(self.design_state_history) > 1:
            user_input_parts.append("Optimization history listed in chronological order.")

            hist_iterations = []
            hist_total_powers = []
            hist_dynamic_powers = []
            hist_leakage_powers = []
            hist_clock_tree_powers = []
            hist_register_powers = []
            hist_combinational_powers = []
            hist_design_areas = []
            hist_remaining_iterations = []
            hist_execution_times = []
            hist_executed_commands = []

            for hist_entry in self.design_state_history[:-1]:
                hist_iteration = hist_entry['iteration']
                hist_state = hist_entry['design_state']
                hist_power = hist_state['power']

                hist_iterations.append(str(hist_iteration))
                hist_total_powers.append(f"{hist_power['total_power']:.4f}")
                hist_dynamic_powers.append(f"{hist_power['dynamic_power']:.4f}")
                hist_leakage_powers.append(f"{hist_power['leakage_power']:.4f}")
                hist_clock_tree_powers.append(f"{hist_power['clock_tree_power']:.4f}")
                hist_register_powers.append(f"{hist_power['register_power']:.4f}")
                hist_combinational_powers.append(f"{hist_power['combinational_power']:.4f}")
                hist_design_areas.append(f"{hist_state['area']['design_area']:.4f}")
                hist_remaining_iterations.append(str(self._calculate_remaining_iterations(hist_iteration)))
                hist_execution_times.append(f"{hist_entry['actual_execution_time']:.0f}s")
                hist_executed_commands.append(hist_entry['executed_command'].replace('{', '{{').replace('}', '}}'))

            hist_parts = [
                f"Iterations {hist_iterations}:",
                f"- Total power: {hist_total_powers}W",
                f"- Dynamic power: {hist_dynamic_powers}W",
                f"- Leakage power: {hist_leakage_powers}W",
                f"- Clock network power: {hist_clock_tree_powers}W",
                f"- Register power: {hist_register_powers}W",
                f"- Combinational power: {hist_combinational_powers}W",
                f"- Design area: {hist_design_areas} um^2",
                f"- Remaining iterations: {hist_remaining_iterations}",
                f"- Actual execution times: {hist_execution_times} s",
                f"- Executed commands: {hist_executed_commands}"
            ]
            user_input_parts.append("\t".join(hist_parts))

        if self.unfixable_analysis_history:
            user_input_parts.append("**UNFIXABLE REASON HISTORY:** removed due to confidential reasons.")
            user_input_parts.append("")
        return "\n".join(user_input_parts)

    def _build_area_user_prompt(self, state):
        """Build user prompt with current design state, optimization history, and unfixable reasons (removed due to confidential reasons)"""
        user_input_parts = []
        objectives = state["objectives"]
        user_input_parts.append("**OBJECTIVES:**")
        user_input_parts.append(objectives)
        user_input_parts.append("")

        current_hist = self.design_state_history[-1] if self.design_state_history else None
        if current_hist:
            current_state_data = current_hist['design_state']
            remaining_iterations = self._calculate_remaining_iterations(current_hist['iteration'])

            current_state_parts = [
                f"Iteration: {current_hist['iteration']}, Remaining iterations: {remaining_iterations}",
                f"- Design area: {current_state_data['area']['design_area']:.4f} um^2",
                f"- Total power: {current_state_data['power']['total_power']:.3e}W",
                f"- Iteration budget: {self.iteration_budget}",
                f"- Remaining iterations: {remaining_iterations}"
            ]
            user_input_parts.append("**CURRENT DESIGN STATE:**" + "".join(current_state_parts))
            user_input_parts.append("")

        user_input_parts.append("**OPTIMIZATION HISTORY:**")
        if len(self.design_state_history) <= 1:
            user_input_parts.append("No previous optimization history available.")
        elif len(self.design_state_history) > 1:
            user_input_parts.append("Optimization history listed in chronological order.")

            hist_iterations = []
            hist_design_areas = []
            hist_total_powers = []
            hist_remaining_iterations = []
            hist_execution_times = []
            hist_executed_commands = []

            for hist_entry in self.design_state_history[:-1]:
                hist_iteration = hist_entry['iteration']
                hist_state = hist_entry['design_state']

                hist_iterations.append(str(hist_iteration))
                hist_design_areas.append(f"{hist_state['area']['design_area']:.4f}")
                hist_total_powers.append(f"{hist_state['power']['total_power']:.4f}")
                hist_remaining_iterations.append(str(self._calculate_remaining_iterations(hist_iteration)))
                hist_execution_times.append(f"{hist_entry['actual_execution_time']:.0f}s")
                hist_executed_commands.append(hist_entry['executed_command'].replace('{', '{{').replace('}', '}}'))

            hist_parts = [
                f"Iterations {hist_iterations}:",
                f"- Design area: {hist_design_areas} um^2",
                f"- Total power: {hist_total_powers}W",
                f"- Remaining iterations: {hist_remaining_iterations}",
                f"- Actual execution times: {hist_execution_times} s",
                f"- Executed commands: {hist_executed_commands}"
            ]
            user_input_parts.append("\t".join(hist_parts))

        if self.unfixable_analysis_history:
            user_input_parts.append("**UNFIXABLE REASON HISTORY:** removed due to confidential reasons.")
            user_input_parts.append("")
        return "\n".join(user_input_parts)
def create_eco_system_ptserver(runtime_budget=3600, log_file="eco_agent_responses_langchain.json"):
    return ECOLangChainSystemPTServerNoReflection(runtime_budget, log_file)


def _command_type_from_string(command_type_str):
    if command_type_str.lower() == "area":
        return ECOType.AREA
    elif command_type_str.lower() == "timing":
        return ECOType.TIMING
    elif command_type_str.lower() == "power":
        return ECOType.POWER
    else:
        return None


def _load_resume_checkpoint(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


def _copy_existing_file(src_path, dst_dir, copied_files):
    if os.path.exists(src_path):
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        copied_files.append(dst_path)


def _snapshot_round_reports(round_index, results):
    round_paths = AgentRunPaths(round_index)
    round_paths.prepare()
    reports_src_dir = os.path.join(configs.base_path, "reports")
    reports_dst_dir = os.path.join(round_paths.run_dir, "reports_backup")
    copied_files = []

    for result in results:
        iteration = result["iteration"]
        _copy_existing_file(
            os.path.join(reports_src_dir, f"report_qor_{iteration}.txt"),
            reports_dst_dir,
            copied_files
        )
        _copy_existing_file(
            os.path.join(reports_src_dir, f"report_power_{iteration}.txt"),
            reports_dst_dir,
            copied_files
        )
        _copy_existing_file(
            os.path.join(reports_src_dir, f"report_qor_{iteration + 1}.txt"),
            reports_dst_dir,
            copied_files
        )
        _copy_existing_file(
            os.path.join(reports_src_dir, f"report_power_{iteration + 1}.txt"),
            reports_dst_dir,
            copied_files
        )

        command = result["execution_result"]["tcl_command"]
        if "opt_area" in command:
            _copy_existing_file(
                os.path.join(reports_src_dir, f"fix_area_{iteration}.txt"),
                reports_dst_dir,
                copied_files
            )
        elif "opt_timing" in command:
            _copy_existing_file(
                os.path.join(reports_src_dir, f"fix_timing_{iteration}.txt"),
                reports_dst_dir,
                copied_files
            )
            _copy_existing_file(
                os.path.join(reports_src_dir, f"unfix_timing_{iteration}_eco_tim.txt"),
                reports_dst_dir,
                copied_files
            )
        elif "opt_power" in command:
            _copy_existing_file(
                os.path.join(reports_src_dir, f"fix_power_{iteration}.txt"),
                reports_dst_dir,
                copied_files
            )

    manifest_path = os.path.join(round_paths.run_dir, "reports_backup_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "round_index": round_index,
            "source_reports_dir": reports_src_dir,
            "reports_backup_dir": reports_dst_dir,
            "copied_files": copied_files,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    return copied_files


def _save_round_resume_state(agent_dir, design, round_index, num_iterations, trace_memory, results):
    trace_memory_path = os.path.join(agent_dir, "trace_memory.pkl")
    round_trace_memory_path = os.path.join(agent_dir, f"trace_memory_after_round_{round_index}.pkl")
    checkpoint_path = os.path.join(agent_dir, "resume_checkpoint.pkl")
    round_checkpoint_path = os.path.join(agent_dir, f"resume_checkpoint_after_round_{round_index}.pkl")
    round_paths = AgentRunPaths(round_index)
    round_paths.prepare()

    with open(trace_memory_path, "wb") as f:
        pickle.dump(trace_memory.traces, f)
    with open(round_trace_memory_path, "wb") as f:
        pickle.dump(trace_memory.traces, f)

    round_results_path = os.path.join(round_paths.run_dir, "round_results.pkl")
    with open(round_results_path, "wb") as f:
        pickle.dump(results, f)

    copied_reports = _snapshot_round_reports(round_index, results)
    checkpoint = {
        "design": design,
        "next_round_index": round_index + 1,
        "last_completed_round": round_index,
        "num_iterations": num_iterations,
        "trace_memory_traces": trace_memory.traces,
        "trace_memory_path": trace_memory_path,
        "round_trace_memory_path": round_trace_memory_path,
        "last_round_results_path": round_results_path,
        "last_round_reports_backup_count": len(copied_reports),
        "last_round_reports_backup_dir": os.path.join(round_paths.run_dir, "reports_backup"),
        "timestamp": datetime.now().isoformat()
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)
    with open(round_checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)
    return checkpoint_path, trace_memory_path


def _apply_design_config(design):
    design_config = configs.DESIGN_CONFIG_OVERRIDES[design]
    configs.base_path = design_config["base_path"]
    eco_ppa_agent.base_path = configs.base_path
    agent_command_executor.base_path = configs.base_path
    agent_run_paths.base_path = configs.base_path
    agent_logs.base_path = configs.base_path
    return design_config


def simulate_eco_iterations_ptserver(design, num_iterations=None, num_rounds=1, resume=False, checkpoint_path=None):
    design_config = _apply_design_config(design)
    runtime_budget = configs.design_runtime_budget[design]
    system = create_eco_system_ptserver(int(runtime_budget))
    trace_memory = TraceMemory()
    agent_dir = configs.get_agent_dir()
    os.makedirs(agent_dir, exist_ok=True)
    trace_memory_path = os.path.join(agent_dir, "trace_memory.pkl")
    if checkpoint_path is None:
        checkpoint_path = os.path.join(agent_dir, "resume_checkpoint.pkl")

    start_round_index = 0
    if resume:
        checkpoint = _load_resume_checkpoint(checkpoint_path)
        if checkpoint["design"] != design:
            raise ValueError(f"Checkpoint design {checkpoint['design']} does not match requested design {design}")
        trace_memory.traces = checkpoint["trace_memory_traces"]
        start_round_index = checkpoint["next_round_index"]
        print(f"Resuming from checkpoint {checkpoint_path}")
        print(f"Loaded {len(trace_memory.traces)} previous traces; next round is {start_round_index}")

    print("ECO LangChain System Test Started")
    if num_iterations is None:
        num_iterations = design_config["max_iterations_per_trace"]
    print(f"Running {num_iterations} iterations")
    print("=" * 70)

    objectives = design_config["objectives"]
    end_round_index = start_round_index + num_rounds

    for round_index in range(start_round_index, end_round_index):
        if system._pt_server is not None:
            system._pt_server.shutdown()
        system.round_index = round_index
        system.persistent_state = None
        system.iteration_count = -1
        system.current_iteration_llm_time = 0.0
        system.iteration_llm_times = {}
        system.design_state_history = []
        system.unfixable_analysis_history = []
        system.short_term_reflection = ""
        system.long_term_reflection = ""
        system._pt_run_paths = None
        system._pt_server = None
        system._pt_server_round = None
        system.start_round()
        system.start_time = time.time()
        results = []
        last_command_type = None
        last_executed_command = ""
        reports = {}
        for i in range(num_iterations):
            print(f"\\nIteration {i}")
            print("-" * 50)
            reports = load_reports_for_iteration(
                i,
                last_command_type=last_command_type,
                last_reports=reports,
                last_executed_command=last_executed_command,
                tool_using=TOOL_USING
            )

            if not reports:
                print(f"No reports found for iteration {i}, stopping")
                break

            result = system.run_iteration(objectives, reports)
            results.append(result)

            selected_command = result["selected_command"]
            execution_result = result["execution_result"]

            print(f"Iteration {i} Results:")
            print(f"  - Selected Command: {selected_command['command_type']}")
            print(f"  - Execution Success: {execution_result['success']}")
            print(f"  - Iteration Time: {result['iteration_time']:.1f}s")
            print(f"  - System Status: {result['system_status']}")

            command_type_str = selected_command["command_type"]
            last_executed_command = execution_result["tcl_command"]
            if command_type_str:
                last_command_type = _command_type_from_string(command_type_str)
                print(f"  - Next iteration will load reports for: {last_command_type.value if last_command_type else 'None'}")
            else:
                last_command_type = None

        print(f"\nTest completed - {len(results)} iterations")
        if results:
            trace_memory.add_trace(results)
            checkpoint_path, trace_memory_path = _save_round_resume_state(
                agent_dir,
                design,
                round_index,
                num_iterations,
                trace_memory,
                results
            )
            print(f"Saved trace memory backup after round {round_index} to {trace_memory_path}")
            print(f"Saved resume checkpoint after round {round_index} to {checkpoint_path}")

    if system._pt_server is not None:
        system._pt_server.shutdown()

    with open(trace_memory_path, "wb") as f:
        pickle.dump(trace_memory.traces, f)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run ECO PT server agent without reflections.")
    parser.add_argument("--design", default="NV_NVDLA_partition_m_test")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-path")
    args = parser.parse_args()
    simulate_eco_iterations_ptserver(
        args.design,
        args.iterations,
        args.rounds,
        resume=args.resume,
        checkpoint_path=args.checkpoint_path
    )


if __name__ == "__main__":
    main()
