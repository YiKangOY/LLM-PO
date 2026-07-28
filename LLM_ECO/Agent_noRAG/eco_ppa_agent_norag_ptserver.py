#!/usr/bin/env python3
"""
ECO Agent System - No-RAG LangChain/LangGraph Implementation with pt_shell server.
Merged structure from eco_ppa_agent.py and eco_ppa_agent_ptserver.py.
"""

import argparse
import os
import pickle
import time
import json
import sys

from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

AGENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Agent"))
if AGENT_ROOT not in sys.path:
    sys.path.append(AGENT_ROOT)

import agent_command_executor_noRAG as agent_command_executor
import agent_logs
import agent_run_paths_noRAG as agent_run_paths
import config_noRAG as configs
import configs as agent_configs
import eco_ppa_agent
from eco_ppa_agent import ECOLangChainSystem
from eco_ppa_agent import ECOState
from eco_ppa_agent import TOOL_USING
from eco_ppa_agent import ShortTermReflectionAgent
from eco_ppa_agent import LongTermReflectionAgent
from eco_ppa_agent import load_reports_for_iteration
from eco_ppa_agent import create_openai_llm
from eco_ppa_trace import TraceMemory
from agent_logs import ECOLogger, LLMResponse, LLMInteraction
from report_parsers import ReportParserManager
from utils import ECOType, extract_content_from_llm_response, extract_json_from_thinking_response
from agent_command_executor_noRAG import execute_tcl_command, build_pt_server
from agent_run_paths_noRAG import AgentRunPaths

CONFIDENTIAL_UNFIXABLE = "removed due to confidential reasons."


class ECOLangChainSystemNoRAG(ECOLangChainSystem):
    """
    ECO System implemented with LangChain/LangGraph without RAG.
    """

    def __init__(self, total_runtime_budget=3600, log_file="eco_agent_responses_langchain.json"):
        self.logger = ECOLogger(log_file)
        self.total_runtime_budget = total_runtime_budget
        self.design_name = os.path.basename(configs.base_path)
        self.iteration_budget = configs.design_max_iterations_per_trace[self.design_name]
        self.start_time = time.time()
        self.iteration_count = -1
        self.round_index = 0

        self.persistent_state = None

        self.current_iteration_llm_time = 0.0
        self.iteration_llm_times = {}

        self.llm_summary = create_openai_llm("gpt-5-mini", configs.USE_OPENAI_REASONING)
        self.llm_area = create_openai_llm("gpt-5-mini", configs.USE_OPENAI_REASONING)
        self.llm_timing = create_openai_llm("gpt-5-mini", configs.USE_OPENAI_REASONING)
        self.llm_power = create_openai_llm("gpt-5-mini", configs.USE_OPENAI_REASONING)

        self.design_state_history = []
        self.unfixable_analysis_history = []
        self.long_term_reflection = ""
        self.short_term_reflection = ""

        self.parser_manager = ReportParserManager()

        self.workflow = self._create_workflow()

        self.logger.logger.info(f"ECO LangChain System (No-RAG) initialized with {total_runtime_budget}s budget")

    def _analyze_reports_node(self, state: ECOState) -> ECOState:
        """Summary agent without RAG; direct target/option decision in one LLM call."""
        start_time = time.time()
        reports = state["reports"]
        iteration = state["iteration"]

        from rag_helpers import extract_current_state, extract_fixing_results

        current_state = extract_current_state(reports)
        fixing_results, unfixable_reasons = extract_fixing_results(reports, state, iteration)
        unfixable_reasons = CONFIDENTIAL_UNFIXABLE

        self._update_design_state_history(state, current_state, fixing_results)
        self._update_unfixable_history(iteration, unfixable_reasons)

        system_prompt_parts = [
            """You are an expert IC design ECO (Engineering Change Order) optimization engineer responsible for optimization scheduling.

**TASK**: You will work as a scheduler to select the optimization target and optimization option strategy for the current ECO iteration based on (with Priority):
1. The current design state, optimization history, and objectives. Unfixable reasons: removed due to confidential reasons.
2. The reflection over strategy.
You need to schedule in following way:
1. Select the optimization target for current iteration (timing/power/area) and choose the optimization selection strategy (Exploration/Exploitation).
2. Summarize your analysis process for final decision making.

**ECO Background**: ECO is an incremental design optimization process that iteratively improves the design by fixing violations and optimizing metrics such as timing, area, and power. Each iteration involves analyzing the current design state, selecting an optimization target, generating and executing ECO commands, and evaluating the results. The goal is to achieve a balanced optimization across timing, area, and power while adhering to an iteration budget.
As the proceeding of ECO iterations, the design gets optimized and the optimization benefit tend to decrease. The success of ECO lies in: 1. Extensively explore optimization target (timing, area, power) and optimization options for each target; 2. Exploit the optimization history and reports to select targeted optimization options. 3. Always keep the iteration budget and remaining iterations in mind.""",
            "",
            """**Report Content**:
- CURRENT DESIGN STATE: The current design state includes timing, power, and area metrics after the most recent optimization iteration.
- OPTIMIZATION HISTORY: The optimization history includes the design states and already performed optimization commands from all previous iterations.
- UNFIXABLE ISSUES HISTORY: removed due to confidential reasons.
- OBJECTIVES: The objectives describe optimization priorities for this run.
            """,
            "",
            "**GUIDELINES**:",
            "1. Please combine the report details and reflection for final decision.",
            "2. For detailed optimization trends that are not clear, you can refer to the detailed attached report.",
            "3. Do not mention any detailed optimization actions like gate_sizing.",
            "4. Do not mention any detailed values of metrics. Unfixable reasons are removed due to confidential reasons.",
            "**RESPONSE FORMAT**: Always respond in valid JSON format."
        ]

        objectives = state["objectives"]
        user_prompt_parts = []
        user_prompt_parts.append("**OBJECTIVES:**")
        user_prompt_parts.append(objectives)
        user_prompt_parts.append("")

        current_hist = self.design_state_history[-1] if self.design_state_history else None
        if current_hist:
            current_state_data = current_hist['design_state']
            user_prompt_parts.append("**CURRENT DESIGN STATE:**")
            user_prompt_parts.append(f"Iteration: {current_hist['iteration']}")
            user_prompt_parts.append(f"  - Setup violations: {current_state_data['timing']['setup_violating_paths']} paths")
            user_prompt_parts.append(f"  - Hold violations: {current_state_data['timing']['hold_violating_paths']} paths")
            user_prompt_parts.append(f"  - Setup critical slack: {current_state_data['timing']['setup_critical_path_slack']}")
            user_prompt_parts.append(f"  - Hold critical slack: {current_state_data['timing']['hold_critical_path_slack']}")
            user_prompt_parts.append(f"  - Setup total negative slack: {current_state_data['timing']['setup_total_negative_slack']}")
            user_prompt_parts.append(f"  - Hold total negative slack: {current_state_data['timing']['hold_total_negative_slack']}")
            user_prompt_parts.append(f"  - Total power: {current_state_data['power']['total_power']:.3e}W")
            user_prompt_parts.append(f"  - Dynamic power: {current_state_data['power']['dynamic_power']:.3e}W")
            user_prompt_parts.append(f"  - Leakage power: {current_state_data['power']['leakage_power']:.3e}W")
            user_prompt_parts.append(f"  - Design area: {current_state_data['area']['design_area']} um^2")
            user_prompt_parts.append("")

        if len(self.design_state_history) > 1:
            user_prompt_parts.append("**OPTIMIZATION HISTORY:**")
            user_prompt_parts.append("Optimization history listed in chronological order.")

            hist_iterations = []
            hist_setup_violating_paths = []
            hist_hold_violating_paths = []
            hist_setup_critical_slacks = []
            hist_hold_critical_slacks = []
            hist_setup_total_negative_slacks = []
            hist_hold_total_negative_slacks = []
            hist_total_powers = []
            hist_dynamic_powers = []
            hist_leakage_powers = []
            hist_design_areas = []
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
                hist_total_powers.append(f"{hist_state['power']['total_power']:.3e}")
                hist_dynamic_powers.append(f"{hist_state['power']['dynamic_power']:.3e}")
                hist_leakage_powers.append(f"{hist_state['power']['leakage_power']:.3e}")
                hist_design_areas.append(str(hist_state['area']['design_area']))
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
                f"- Dynamic power: {hist_dynamic_powers}W",
                f"- Leakage power: {hist_leakage_powers}W",
                f"- Design area: {hist_design_areas} um^2",
                f"- Executed commands: {hist_executed_commands}"
            ]
            user_prompt_parts.append("\t".join(hist_parts))

        if self.unfixable_analysis_history:
            user_prompt_parts.append("**UNFIXABLE REASON HISTORY:** removed due to confidential reasons.")
            user_prompt_parts.append("")
        user_prompt_parts.extend([
            "Reflection on Strategy:",
            self.long_term_reflection,
            f"**CURRENT ITERATION:** {iteration}",
            "**RESPONSE FORMAT**: Always respond in valid JSON format:",
            "{{",
            '  "target": "timing|power|area",',
            '  "option": "Exploration|Exploitation",',
            '  "reasoning": "A short summary of the reason you choose this target and option",',
            "}}"
        ])

        messages = [
            ("system", "\n".join(system_prompt_parts)),
            ("human", "\n".join(user_prompt_parts))
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        formatted_messages = prompt.format_messages()
        messages_sent = [{"role": msg.type, "content": msg.content} for msg in formatted_messages]

        llm_start_time = time.time()
        response = self.llm_summary.invoke(formatted_messages)
        processing_time = time.time() - llm_start_time

        content_text = extract_content_from_llm_response(response.content)
        json_content = extract_json_from_thinking_response(content_text)
        parser = JsonOutputParser()
        llm_analysis = parser.parse(json_content)

        token_usage = {
            "input_tokens": response.usage_metadata["input_tokens"],
            "output_tokens": response.usage_metadata["output_tokens"]
        }

        target = llm_analysis['target']
        option = llm_analysis['option']
        reasoning = llm_analysis['reasoning']

        interaction = LLMInteraction(
            agent_type="SummaryAgent_FinalAnalysis",
            timestamp=datetime.now(),
            iteration=iteration,
            round_index=self.round_index,
            messages_sent=messages_sent,
            response_received={
                "content": content_text,
                "type": "json",
                "parsed_decision": {
                    "target": target,
                    "option": option,
                    "reasoning": reasoning
                }
            },
            processing_time=processing_time,
            model_name=self.llm_summary.model_name,
            token_usage=token_usage
        )
        self.logger.log_llm_interaction(interaction)

        self.current_iteration_llm_time += processing_time

        unified_analysis = state["unified_analysis"]
        unified_analysis['iteration'] = iteration
        unified_analysis['current_state'] = current_state
        unified_analysis['fixing_results'] = fixing_results
        unified_analysis['unfixable_reasons'] = unfixable_reasons
        unified_analysis['llm_summary'] = llm_analysis
        unified_analysis['rag_queries'] = []
        unified_analysis['optimization_strategy'] = option
        unified_analysis['processing_time'] = time.time() - start_time

        unified_analysis.update(self._extract_fix_unfix_reports(reports))
        for key in ("timing_unfix", "area_unfix", "power_unfix"):
            if key in unified_analysis:
                unified_analysis[key] = CONFIDENTIAL_UNFIXABLE

        state["unified_analysis"] = unified_analysis
        state["rag_queries"] = []
        state["rag_retrieved_content"] = ""
        state["selected_route"] = target
        state["optimization_strategy"] = option

        response = LLMResponse(
            agent_type="SummaryAgent",
            timestamp=datetime.now(),
            input_data={
                'reports_processed': list(reports.keys()),
                'iteration': iteration,
                'has_dialogue_history': len(state['messages']) > 0,
                'selected_route': target,
                'optimization_strategy': option,
                'decision_reasoning': reasoning
            },
            output_data=unified_analysis,
            processing_time=time.time() - start_time,
            iteration=iteration
        )
        self.logger.log_response(response)

        return state

    def _generate_timing_command_node(self, state: ECOState) -> ECOState:
        """Generate timing command without RAG in a single LLM call."""
        start_time = time.time()

        unified_analysis = state["unified_analysis"]
        current_state = unified_analysis['current_state']
        llm_summary = unified_analysis['llm_summary']
        strategy = llm_summary['option']
        objectives = state["objectives"]

        setup_violations = current_state['timing']['setup_violating_paths']
        hold_violations = current_state['timing']['hold_violating_paths']

        command_system_prompt = f"""You are an expert in IC design timing fixing responsible for fixing timing violations.
Your task is to analyze the current timing violations and generate an effective command based on: 1. optimization strategies and objectives 2. The current design state and optimization history. 3. The previous unfixable analysis removed due to confidential reasons.
Then generate an optimal opt_timing command.

**COMMAND FORMAT:**
Command format: opt_timing -violation_type [setup|hold] -cell_class [cell_class] -actions [method] -site_mode [mode] (-area_cap [x]) (-slack_above [slack_limit1]) (-slack_below [slack_limit2])
Available violation type: setup or hold, you can only choose one of them.
Available actions for setup: gate_sizing, gate_sizing_side_load, buffer_insertion, you can select one method each time. The most common choice is gate_sizing, then buffer_insertion, then gate_sizing_side_load.
Available actions for hold: gate_sizing, buffer_insertion, you can select one method each time. The most common choice is gate_sizing, then buffer_insertion.
Available cell types: combinational, sequential, clock_tree, you can only choose one of them. If you do not choose combinational, then you can only use gate_sizing method. The most common cell type is combinational, then sequential, use clock_tree only if needed.
Available physical modes: open_slot, occupied_slot. The common selection is open_slot, you can explore occupied_slot if you find the timing optimization result is not promising.
Optional -area_cap: Specify area increment limit for gate_sizing, only use it when gate_sizing option selected, default value is 2, available values are [4|8|10|12|16|20]. In exploration stage, use the default value; if needed, increase these values.
Optional -slack_above slack_limit1: Default is 0. The slack_limit1 can be set to a positive value, then the command will try to improve the timing of paths with slacks less than the slack_limit1 (aka more positive slacks). This may increase area and power but can further improve timing. Only use it in exploitation stage when you see WNS can not be optimized futher but there is still TNS violation. Also do not use it when the remaining iterations are few to ensure there is time to make power/area recovery. Give pure number, no units.
Optional: -slack_below slack_limit2: Default is 0. The slack_limit2 can be set to a negative value, the command will NOT try to improve the timing of paths with slacks larger than the slack_limit2 (aka more negative slacks). This may limit the timing optimization but lead to reduced power and area incresing. Do not use it when you need substantial timing optimization. Give pure number, no units.

**UNFIXABLE REASONS:** removed due to confidential reasons.

**RESPONSE FORMAT**: Always respond in valid JSON format.
"""

        command_user_input = f"Strategy: {strategy}\nObjectives: {objectives}\n\n"
        command_user_input += self._build_timing_user_prompt(state)
        command_user_input += "\n**Now You Can generate the response with JSON resonse:**\n"
        command_user_input += "{{\n"
        command_user_input += '  "command": "The generated command"\n'
        command_user_input += "}}"

        command_prompt = ChatPromptTemplate.from_messages([
            ("system", command_system_prompt),
            ("human", command_user_input)
        ])

        command_messages = command_prompt.format_messages()
        command_messages_sent = [{"role": msg.type, "content": msg.content} for msg in command_messages]

        llm_start_time = time.time()
        command_response = self.llm_timing.invoke(command_messages)
        command_time = time.time() - llm_start_time
        command_token_usage = {
            "input_tokens": command_response.usage_metadata["input_tokens"],
            "output_tokens": command_response.usage_metadata["output_tokens"]
        }

        command_content_text = extract_content_from_llm_response(command_response.content)
        command_json = extract_json_from_thinking_response(command_content_text)
        parser = JsonOutputParser()
        command_result_data = parser.parse(command_json)

        self.current_iteration_llm_time += command_time

        command = command_result_data['command']

        interaction = LLMInteraction(
            agent_type="TimingCommandGenerator_Command",
            timestamp=datetime.now(),
            iteration=state["iteration"],
            round_index=self.round_index,
            messages_sent=command_messages_sent,
            response_received={"content": command_content_text, "type": "json"},
            processing_time=command_time,
            model_name=self.llm_timing.model_name,
            token_usage=command_token_usage
        )
        self.logger.log_llm_interaction(interaction)

        command_result = {
            'command_type': ECOType.TIMING.value,
            'tcl_command': command
        }

        response = LLMResponse(
            agent_type="TimingCommandGenerator",
            timestamp=datetime.now(),
            input_data={
                'setup_violations': setup_violations,
                'hold_violations': hold_violations,
                'strategy': strategy,
                'evaluation': "N/A"
            },
            output_data=command_result,
            processing_time=time.time() - start_time,
            iteration=state["iteration"]
        )
        self.logger.log_response(response)

        return {"selected_command": command_result, "command_proposals": {"timing": command_result}}

    def _generate_power_command_node(self, state: ECOState) -> ECOState:
        """Generate power command without RAG in a single LLM call."""
        start_time = time.time()

        unified_analysis = state["unified_analysis"]
        current_state = unified_analysis['current_state']
        llm_summary = unified_analysis['llm_summary']
        strategy = llm_summary['option']

        power_metrics = current_state['power']
        total_power = power_metrics['total_power']
        objectives = state["objectives"]

        command_system_prompt = f"""You are an expert in IC design power optimization responsible for generating optimal opt_power commands.
Your task is to analyze the current power consumption and generate an effective command based on: 1. optimization strategies and objectives 2. The current design state and optimization history. 3. The previous unfixable analysis removed due to confidential reasons.
Then generate an optimal opt_power command.
Note: The reduction of power and reduction of area are usually accompanied with each other.

**UNFIXABLE REASONS:** removed due to confidential reasons.
T - Sizing cell might degrade timing -> Try other options
W - Sizing cell might degrade DRC -> Try other options
X - Cell is unusable for ECO -> Try other options
Z - Cell is sized -> Try other options

**COMMAND FORMAT:**
Command format: opt_power -actions [method] -cell_class [cell_class] -power_scope [mode] (-setup_guard [setup_guard])
Available actions: gate_sizing, buffer_removal
Available power_scope: total | dynamic | leakage. Total power optimization is more suitable at earlier stage or exploration to directly optimize power. dynamic and leakage can work on later exploitation stage when you find corresponding power component is not optimized much or  re-rised after timing optimization.
Available cell types: [combinational, sequential] you can choose any one from them. You can also choose this to target power optimization on combinational or sequential cells. Usually, at earlier stage, can start with combinational.
Note: buffer_removal cannot be used with other options like -cell_class.
Optional: -setup_guard setup_guard: Specify the setup timing margin to ensure that the power optimization does not degrade setup beyond this margin. Default is 0. Set it as a negative value to allow setup degradation when you optimize power with high efforts. Give pure number, no units.

**RESPONSE FORMAT**: Always respond in valid JSON format.
"""

        command_user_input = f"Strategy: {strategy}\nObjectives: {objectives}\n\n"
        command_user_input += self._build_power_user_prompt(state)
        command_user_input += "\n**Now You Can generate the response with JSON resonse:**\n"
        command_user_input += "{{\n"
        command_user_input += '  "command": "The generated command"\n'
        command_user_input += "}}"

        command_prompt = ChatPromptTemplate.from_messages([
            ("system", command_system_prompt),
            ("human", command_user_input)
        ])

        command_messages = command_prompt.format_messages()
        command_messages_sent = [{"role": msg.type, "content": msg.content} for msg in command_messages]

        llm_start_time = time.time()
        command_response = self.llm_power.invoke(command_messages)
        command_time = time.time() - llm_start_time
        command_token_usage = {
            "input_tokens": command_response.usage_metadata["input_tokens"],
            "output_tokens": command_response.usage_metadata["output_tokens"]
        }

        command_content_text = extract_content_from_llm_response(command_response.content)
        command_json = extract_json_from_thinking_response(command_content_text)
        parser = JsonOutputParser()
        command_result_data = parser.parse(command_json)

        self.current_iteration_llm_time += command_time

        command = command_result_data['command']

        interaction = LLMInteraction(
            agent_type="PowerCommandGenerator_Command",
            timestamp=datetime.now(),
            iteration=state["iteration"],
            round_index=self.round_index,
            messages_sent=command_messages_sent,
            response_received={"content": command_content_text, "type": "json"},
            processing_time=command_time,
            model_name=self.llm_power.model_name,
            token_usage=command_token_usage
        )
        self.logger.log_llm_interaction(interaction)

        command_result = {
            'command_type': ECOType.POWER.value,
            'tcl_command': command
        }

        response = LLMResponse(
            agent_type="PowerCommandGenerator",
            timestamp=datetime.now(),
            input_data={
                'total_power': total_power,
                'strategy': strategy,
                'evaluation': "N/A"
            },
            output_data=command_result,
            processing_time=time.time() - start_time,
            iteration=state["iteration"]
        )
        self.logger.log_response(response)

        return {"selected_command": command_result, "command_proposals": {"power": command_result}}

    def _generate_area_command_node(self, state: ECOState) -> ECOState:
        """Generate area command without RAG in a single LLM call."""
        start_time = time.time()

        unified_analysis = state["unified_analysis"]
        current_state = unified_analysis['current_state']
        llm_summary = unified_analysis['llm_summary']
        strategy = llm_summary['option']

        area_state = current_state['area']
        objectives = state["objectives"]

        command_system_prompt = f"""You are an expert in IC design area optimization responsible for generating optimal opt_area commands.
Your task is to analyze the current area consumption and generate an effective command based on: 1. optimization strategies and objectives 2. The current design state and optimization history. 3. The previous unfixable analysis removed due to confidential reasons.
Then generate an optimal opt_area command.
Note: The reduction of power and area are usually accompanied with each other.

**UNFIXABLE REASONS:** removed due to confidential reasons.
T - Sizing cell might degrade timing -> Try other options
W - Sizing cell might degrade DRC -> Try other options
X - Cell is unusable for ECO -> Try other options
Z - Cell is sized -> Try other options

**COMMAND FORMAT:**
Command format: opt_area -actions [method] -cell_class [cell_class] (-setup_guard [setup_guard])
Available actions: gate_sizing, buffer_removal
Available cell types: [combinational, sequential, clock_tree] you can choose any one from them. You can also choose this to target area optimization on combinational or sequential cells. Usually, at earlier stage, can start with combinational.
Note: buffer_removal cannot be used with any other options, including -cell_class.
Optional: -setup_guard setup_guard: Specify the setup timing margin to ensure that the area optimization does not degrade setup beyond this margin. Default is 0. Set it as a negative value to allow setup degradation when you optimize area with high efforts. Give pure number, no units.

**RESPONSE FORMAT**: Always respond in valid JSON format.
"""

        command_user_input = f"Strategy: {strategy}\nObjectives: {objectives}\n\n"
        command_user_input += self._build_area_user_prompt(state)
        command_user_input += "\n**Now You Can generate the response with JSON resonse:**\n"
        command_user_input += "{{\n"
        command_user_input += '  "command": "The generated command"\n'
        command_user_input += "}}"

        command_prompt = ChatPromptTemplate.from_messages([
            ("system", command_system_prompt),
            ("human", command_user_input)
        ])

        command_messages = command_prompt.format_messages()
        command_messages_sent = [{"role": msg.type, "content": msg.content} for msg in command_messages]

        llm_start_time = time.time()
        command_response = self.llm_area.invoke(command_messages)
        command_time = time.time() - llm_start_time
        command_token_usage = {
            "input_tokens": command_response.usage_metadata["input_tokens"],
            "output_tokens": command_response.usage_metadata["output_tokens"]
        }

        command_content_text = extract_content_from_llm_response(command_response.content)
        command_json = extract_json_from_thinking_response(command_content_text)
        parser = JsonOutputParser()
        command_result_data = parser.parse(command_json)

        self.current_iteration_llm_time += command_time

        command = command_result_data['command']

        interaction = LLMInteraction(
            agent_type="AreaCommandGenerator_Command",
            timestamp=datetime.now(),
            iteration=state["iteration"],
            round_index=self.round_index,
            messages_sent=command_messages_sent,
            response_received={"content": command_content_text, "type": "json"},
            processing_time=command_time,
            model_name=self.llm_area.model_name,
            token_usage=command_token_usage
        )
        self.logger.log_llm_interaction(interaction)

        command_result = {
            'command_type': ECOType.AREA.value,
            'tcl_command': command
        }

        response = LLMResponse(
            agent_type="AreaCommandGenerator",
            timestamp=datetime.now(),
            input_data={
                'design_area': area_state['design_area'],
                'strategy': strategy,
                'evaluation': "N/A"
            },
            output_data=command_result,
            processing_time=time.time() - start_time,
            iteration=state["iteration"]
        )
        self.logger.log_response(response)

        return {"selected_command": command_result, "command_proposals": {"area": command_result}}


class ECOLangChainSystemNoRAGPTServer(ECOLangChainSystemNoRAG):
    """
    ECOLangChainSystemNoRAG that executes commands via persistent pt_shell server.
    """

    def __init__(self, total_runtime_budget=3600, log_file="eco_agent_responses_langchain.json"):
        super().__init__(total_runtime_budget=total_runtime_budget, log_file=log_file)
        self._pt_run_paths = None
        self._pt_server = None
        self._pt_server_round = None
        self._last_llm_interaction_index = 0

    def _get_run_paths(self):
        if self._pt_run_paths is None or self._pt_run_paths.round_index != self.round_index:
            self._pt_run_paths = AgentRunPaths(self.round_index)
            self._pt_run_paths.prepare()
        return self._pt_run_paths

    def _get_pt_server(self, run_paths):
        if self._pt_server is None:
            self._pt_server = build_pt_server(self.round_index, run_paths=run_paths)
        return self._pt_server

    def _execute_tcl_command(self, tcl_command, command_type, iteration):
        run_paths = self._get_run_paths()
        pt_server = self._get_pt_server(run_paths)
        print(f"Executing TCL command (iteration {iteration}): {tcl_command}")
        start_time = time.time()
        execution_result = execute_tcl_command(
            tcl_command,
            command_type,
            iteration,
            self.round_index,
            run_paths=run_paths,
            pt_server=pt_server
        )
        execution_result['wall_time'] = time.time() - start_time
        return execution_result

    def start_round(self):
        if self._pt_server is None or self._pt_server_round != self.round_index:
            if self._pt_server is not None:
                self._pt_server.shutdown()
            self._pt_run_paths = AgentRunPaths(self.round_index)
            self._pt_run_paths.prepare()
            self._pt_server = build_pt_server(self.round_index, run_paths=self._pt_run_paths)
            self._pt_server.ensure_running()
            self._pt_server_round = self.round_index

    def _log_llm_call_runtimes(self):
        interactions = self.logger.llm_interactions
        new_interactions = interactions[self._last_llm_interaction_index:]
        for interaction in new_interactions:
            response = LLMResponse(
                agent_type="LLMCallRuntime",
                timestamp=interaction.timestamp,
                input_data={
                    'llm_agent_type': interaction.agent_type,
                    'model_name': interaction.model_name,
                    'round_index': interaction.round_index
                },
                output_data={'runtime_seconds': interaction.processing_time},
                processing_time=interaction.processing_time,
                iteration=interaction.iteration
            )
            self.logger.log_response(response)
        self._last_llm_interaction_index = len(interactions)

    def run_iteration(self, objectives, reports):
        result = super().run_iteration(objectives, reports)
        self._log_llm_call_runtimes()
        return result


def create_eco_system_no_rag(runtime_budget=3600, log_file="eco_agent_responses_langchain.json"):
    return ECOLangChainSystemNoRAG(runtime_budget, log_file)


def create_eco_system_no_rag_ptserver(runtime_budget=3600, log_file="eco_agent_responses_langchain.json"):
    return ECOLangChainSystemNoRAGPTServer(runtime_budget, log_file)


def _apply_design_config(design):
    design_config = configs.DESIGN_CONFIG_OVERRIDES[design]
    configs.base_path = design_config["base_path"]
    agent_configs.base_path = configs.base_path
    agent_configs.AGENT_DIR_NAME = configs.AGENT_DIR_NAME
    eco_ppa_agent.base_path = configs.base_path
    agent_command_executor.base_path = configs.base_path
    agent_run_paths.base_path = configs.base_path
    agent_logs.base_path = configs.base_path
    return design_config


def simulate_eco_iterations_no_rag_ptserver(design, num_iterations=None, num_rounds=1, reflection_model_name="gpt-4o"):
    design_config = _apply_design_config(design)
    runtime_budget = configs.design_runtime_budget[design]
    system = create_eco_system_no_rag_ptserver(int(runtime_budget))
    system._pt_run_paths = design_config['base_path']
    trace_memory = TraceMemory()
    reflection_agent = ShortTermReflectionAgent(
        system.logger,
        model_name=reflection_model_name,
        use_openai_reasoning=configs.USE_OPENAI_REASONING
    )
    long_term_reflection_agent = LongTermReflectionAgent(
        system.logger,
        model_name=reflection_model_name,
        use_openai_reasoning=configs.USE_OPENAI_REASONING
    )

    print("ECO LangChain System (No-RAG) Test Started")
    if num_iterations is None:
        num_iterations = design_config["max_iterations_per_trace"]
    print(f"Running {num_iterations} iterations")
    print("=" * 70)

    objectives = design_config["objectives"]

    for round_index in range(num_rounds):
        if system._pt_server is not None:
            system._pt_server.shutdown()
        system.round_index = round_index
        system.persistent_state = None
        system.iteration_count = -1
        system.current_iteration_llm_time = 0.0
        system.iteration_llm_times = {}
        system.design_state_history = []
        system.unfixable_analysis_history = []
        system._pt_run_paths = None
        system._pt_server = None
        system._pt_server_round = None
        system._pt_run_paths = design_config['base_path']
        system.start_round()
        if round_index == 0:
            system.long_term_reflection = ""
            system.short_term_reflection = ""
        system.start_time = time.time()
        results = []
        last_command_type = None
        last_executed_command = ""
        reports = {}
        for i in range(num_iterations):
            print(f"\nIteration {i}")
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

            selected_command = result['selected_command']
            execution_result = result['execution_result']

            print(f"Iteration {i} Results:")
            print(f"  - Selected Command: {selected_command['command_type']}")
            print(f"  - Execution Success: {execution_result['success']}")
            print(f"  - Iteration Time: {result['iteration_time']:.1f}s")
            print(f"  - System Status: {result['system_status']}")

            command_type_str = selected_command['command_type']
            last_executed_command = execution_result['tcl_command']
            if command_type_str:
                if command_type_str.lower() == 'area':
                    last_command_type = ECOType.AREA
                elif command_type_str.lower() == 'timing':
                    last_command_type = ECOType.TIMING
                elif command_type_str.lower() == 'power':
                    last_command_type = ECOType.POWER
                else:
                    last_command_type = None
                print(f"  - Next iteration will load reports for: {last_command_type.value if last_command_type else 'None'}")
            else:
                last_command_type = None

        print(f"\nTest completed - {len(results)} iterations")
        if results:
            trace_memory.add_trace(results)
            pareto_traces = trace_memory.extract_pareto_traces()
            print(f"Pareto-optimal traces: {len(pareto_traces)}")
            reflection = reflection_agent.generate_short_term_reflection(
                objectives,
                pareto_traces,
                len(results) - 1,
                system.round_index
            )
            print("\nShort-term reflection:")
            print(reflection)
            system.short_term_reflection = reflection
            long_term_reflection = long_term_reflection_agent.generate_long_term_reflection(
                objectives,
                pareto_traces,
                len(results) - 1,
                system.round_index
            )
            print("\nLong-term reflection:")
            print(long_term_reflection)
            system.long_term_reflection = long_term_reflection

        if system._pt_server is not None:
            system._pt_server.shutdown()

    agent_dir = configs.get_agent_dir()
    os.makedirs(agent_dir, exist_ok=True)
    trace_memory_path = os.path.join(agent_dir, "trace_memory.pkl")
    with open(trace_memory_path, "wb") as f:
        pickle.dump(trace_memory.traces, f)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run ECO PT server agent (No-RAG).")
    parser.add_argument("--design", default="hidden1")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--reflection-model", default="gpt-5-mini")
    args = parser.parse_args()
    simulate_eco_iterations_no_rag_ptserver(
        args.design,
        args.iterations,
        args.rounds,
        args.reflection_model
    )


if __name__ == "__main__":
    main()
