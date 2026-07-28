#!/usr/bin/env python3
"""
ECO Agent System - LangChain/LangGraph Implementation
Multi-agent system for Engineering Change Order optimization
Reimplemented using LangChain and LangGraph for simplicity
"""

import argparse
import copy
import re
import sys
import time
import json
import os
import pickle
import shutil

from datetime import datetime
from typing import Dict, List, Any, TypedDict, Annotated

from dataclasses import dataclass, asdict

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from langgraph.graph import StateGraph, END

# Import existing report parsers
import agent_run_paths
import configs
from report_parsers import ReportParserManager, PowerMetrics, QoRMetrics, extract_json_from_response
from eco_database import ReportParser
from eco_ppa_trace import TraceMemory
from configs import BUFFER_LIST, base_path, design_runtime_budget, design_max_iterations_per_trace, USE_OPENAI_REASONING
from configs import get_agent_dir
from agent_run_paths import AgentRunPaths

# Import RAG utilities
from rag_utils import create_eco_rag_system

# Import logs
from agent_logs import ECOLogger, LLMResponse, LLMInteraction
# Import utils
from utils import ECOType, extract_content_from_llm_response, extract_json_from_thinking_response, escape_unquoted_quotes_in_json
# Configuration

OPENAI_REASONING_CONFIG = {
    "effort": "low",
    "summary": "auto"
}

def create_openai_llm(model_name, use_openai_reasoning):
    llm_kwargs = {"model": model_name}
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if api_key:
        llm_kwargs["api_key"] = api_key
    if base_url:
        llm_kwargs["base_url"] = base_url
    if use_openai_reasoning:
        llm_kwargs["reasoning"] = OPENAI_REASONING_CONFIG
    return ChatOpenAI(**llm_kwargs)

# Tool usage configuration
TOOL_USING = True  # Enable slack distribution in timing unfixing reports

# Pydantic models for structured outputW
class CommandOutput(BaseModel):
    """Structured output for ECO commands"""
    tcl_command: str = Field(description="Complete TCL command to execute")
    reasoning: str = Field(description="Reasoning for this command")


# State definition for LangGraph
def combine_command_proposals(existing: Dict[str, Dict[str, Any]], new: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Combine command proposals from multiple nodes running in parallel"""
    if existing is None:
        existing = {}
    result = existing.copy()
    result.update(new)
    return result

class ECOState(TypedDict):
    """State definition for the ECO workflow"""
    iteration: int
    reports: Dict[str, str]
    unified_analysis: Dict[str, Any]
    command_proposals: Annotated[Dict[str, Dict[str, Any]], combine_command_proposals]
    selected_command: Dict[str, Any]
    execution_result: Dict[str, Any]
    elapsed_runtime: float
    remaining_budget: float
    messages: List
    system_status: str
    selected_route: str  # New field to store which node to route to
    rag_queries: List[str]  # Generated queries for knowledge retrieval
    rag_retrieved_content: str  # Formatted retrieved knowledge content
    optimization_strategy: str  # Exploration/Exploitation optimization strategy
    objectives: str  # Design objectives for the ECO run

# chat history
from langchain_core.chat_history import InMemoryChatMessageHistory

chats_by_session_id = {}


def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    chat_history = chats_by_session_id.get(session_id)
    if chat_history is None:
        chat_history = InMemoryChatMessageHistory()
        chats_by_session_id[session_id] = chat_history
    return chat_history

class ECOLangChainSystem:
    """
    ECO System implemented with LangChain and LangGraph
    """
    
    def __init__(self, total_runtime_budget: float = 3600, log_file: str = "eco_agent_responses_langchain.json"):
        self.logger = ECOLogger(log_file)
        self.total_runtime_budget = total_runtime_budget
        self.design_name = os.path.basename(base_path)
        self.iteration_budget = design_max_iterations_per_trace[self.design_name]
        self.start_time = time.time()
        self.iteration_count = -1
        self.round_index = 0

        # Initialize persistent state for accumulating context across iterations
        self.persistent_state: ECOState = None

        # Track LLM interaction time per iteration
        self.current_iteration_llm_time = 0.0
        self.iteration_llm_times = {}  # {iteration: total_llm_time}
        
        # Initialize separate LLM instances for each node
        self.llm_summary = create_openai_llm("gpt-4o-mini", USE_OPENAI_REASONING)
        
        self.llm_area = create_openai_llm("gpt-4o-mini", USE_OPENAI_REASONING)
        
        self.llm_timing = create_openai_llm("gpt-4o-mini", USE_OPENAI_REASONING)
        
        self.llm_power = create_openai_llm("gpt-4o-mini", USE_OPENAI_REASONING)
        
        

        # Initialize optimization history lists for summary agent
        self.design_state_history = []
        self.unfixable_analysis_history = []
        self.long_term_reflection = ""
        self.short_term_reflection = ""
        
        # Initialize report parser
        self.parser_manager = ReportParserManager()
        
        # Initialize RAG systems for knowledge retrieval
        self.rag_system_summary = create_eco_rag_system(source_file='summarized_opt_strategies.txt')
        self.rag_system_timing = create_eco_rag_system(source_file='command_timing.txt')
        self.rag_system_area = create_eco_rag_system(source_file='command_area.txt')
        self.rag_system_power = create_eco_rag_system(source_file='command_power.txt')
        
        # Create the LangGraph workflow
        self.workflow = self._create_workflow()
        
        self.logger.logger.info(f"ECO LangChain System initialized with {total_runtime_budget}s budget")

    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(ECOState)
        
        # Add nodes
        workflow.add_node("analyze_reports", self._analyze_reports_node)
        
        workflow.add_node("generate_area_command", self._generate_area_command_node)
        workflow.add_node("generate_timing_command", self._generate_timing_command_node)
        workflow.add_node("generate_power_command", self._generate_power_command_node)
        workflow.add_node("execute_command", self._execute_command_node)
        
        # Define the flow with conditional routing
        workflow.set_entry_point("analyze_reports")
        
        # Add conditional edges based on analyze_reports decision
        workflow.add_conditional_edges(
            "analyze_reports",
            self._route_to_command_generator,
            {
                "area": "generate_area_command",
                "timing": "generate_timing_command", 
                "power": "generate_power_command"
            }
        )
        
        # Direct routing from command generators to execution
        workflow.add_edge("generate_area_command", "execute_command")
        workflow.add_edge("generate_timing_command", "execute_command")
        workflow.add_edge("generate_power_command", "execute_command")
        
        workflow.add_edge("execute_command", END)
        
        # Compile workflow
        return workflow.compile()

    def _calculate_remaining_iterations(self, iteration_index: int) -> int:
        remaining_iterations = self.iteration_budget - (iteration_index + 1)
        if remaining_iterations < 0:
            return 0
        return remaining_iterations
    
    def _update_design_state_history(self, state: ECOState, current_state: Dict, fixing_results: Dict) -> None:
        """Update design state history with current iteration data and previous iteration's execution results"""
        iteration = state["iteration"]

        # Update previous iteration's executed command (iteration N-1 when processing iteration N)
        if iteration > 0 and len(self.design_state_history) > 0:
            prev_iteration_idx = iteration - 1
            if prev_iteration_idx < len(self.design_state_history):
                # Validate execution_result exists and has required fields
                if 'execution_result' not in state:
                    raise KeyError(f"execution_result not found in state for iteration {iteration}")

                execution_result = state['execution_result']
                if 'tcl_command' not in execution_result:
                    raise KeyError(f"tcl_command not found in execution_result for iteration {iteration}")
                if 'execution_time' not in execution_result:
                    raise KeyError(f"execution_time not found in execution_result for iteration {iteration}")

                # Validate selected_command exists
                if 'selected_command' not in state:
                    raise KeyError(f"selected_command not found in state for iteration {iteration}")

                executed_command = execution_result['tcl_command']
                execution_time = execution_result['execution_time']

                self.design_state_history[prev_iteration_idx]['executed_command'] = executed_command
                self.design_state_history[prev_iteration_idx]['actual_execution_time'] = execution_time

        # Append current iteration data (command will be filled in next iteration)
        self.design_state_history.append({
            'iteration': iteration,
            'design_state': current_state,
            'fixing_results': fixing_results,
            'remaining_budget': state['remaining_budget'],
            'remaining_iterations': self._calculate_remaining_iterations(iteration),
            'actual_execution_time': 0,  # Will be updated in next iteration
            'executed_command': ''  # Will be updated in next iteration
        })

    def _update_unfixable_history(self, iteration: int, unfixable_reasons: Dict) -> None:
        """Update unfixable analysis history if there are unfixable issues"""
        if unfixable_reasons:
            self.unfixable_analysis_history.append({
                'iteration': iteration,
                'unfixable_reasons': unfixable_reasons
            })

    def _extract_fix_unfix_reports(self, reports: Dict) -> Dict:
        """Extract fix/unfix data from reports, excluding raw content and detailed paths"""
        fix_unfix_data = {}

        if 'timing_fix' in reports:
            fix_unfix_data['timing_fix'] = {
                k: v for k, v in reports['timing_fix'].items()
                if k not in ['raw_content', 'timing_paths', 'paths_sorted_by_slack']
            }
            fix_unfix_data['timing_unfix'] = {
                k: v for k, v in reports['timing_unfix'].items()
                if k not in ['raw_content', 'timing_paths', 'paths_sorted_by_slack']
            }

        if 'area_fix' in reports:
            fix_unfix_data['area_fix'] = {k: v for k, v in reports['area_fix'].items() if k != 'raw_content'}
            fix_unfix_data['area_unfix'] = {k: v for k, v in reports['area_unfix'].items() if k != 'raw_content'}

        if 'power_fix' in reports:
            fix_unfix_data['power_fix'] = {k: v for k, v in reports['power_fix'].items() if k != 'raw_content'}
            fix_unfix_data['power_unfix'] = {k: v for k, v in reports['power_unfix'].items() if k != 'raw_content'}

        return fix_unfix_data

    def _analyze_reports_node(self, state: ECOState) -> ECOState:
        """RAG-Enhanced Summary agent with two-phase RAG process:
        Phase 1: Generate combined target/option selection evaluation and query → RAG retrieval
        Phase 2: Make final decision using combined RAG results
        """
        start_time = time.time()
        reports = state["reports"]
        iteration = state["iteration"]

        # Import RAG helper functions
        from rag_helpers import (
            extract_current_state,
            extract_fixing_results,
            retrieve_knowledge,
            generate_selection_queries,
            generate_final_decision
        )

        # ===== STEP 1: Extract and organize data =====
        current_state = extract_current_state(reports)
        fixing_results, unfixable_reasons = extract_fixing_results(reports, state, iteration)

        # Update historical tracking
        self._update_design_state_history(state, current_state, fixing_results)
        self._update_unfixable_history(iteration, unfixable_reasons)

        # ===== STEP 2: Phase 1 - Combined Selection (Evaluation → Query → RAG) =====
        rag_messages = []  # Conversation context for query generation

        selection_query, evaluation_on_selection, selection_query_time = generate_selection_queries(
            TOOL_USING, current_state, fixing_results, unfixable_reasons, iteration, self.round_index, state,
            self.llm_summary, self.design_state_history, self.unfixable_analysis_history,
            rag_messages, self.logger
        )
        self.current_iteration_llm_time += selection_query_time

        rag_retrieved_content = retrieve_knowledge(selection_query, self.rag_system_summary, k=2)

        # ===== STEP 3: Phase 2 - Final Decision with combined RAG results =====
        llm_analysis, target, option, reasoning, decision_time = generate_final_decision(
            TOOL_USING, rag_retrieved_content, evaluation_on_selection, self.long_term_reflection,
            iteration, self.round_index, state, self.llm_summary,
            self.design_state_history, self.unfixable_analysis_history, rag_messages, self.logger
        )
        self.current_iteration_llm_time += decision_time

        # ===== STEP 4: Prepare unified analysis output =====
        unified_analysis = {
            'iteration': iteration,
            'current_state': current_state,
            'fixing_results': fixing_results,
            'unfixable_reasons': unfixable_reasons,
            'llm_summary': llm_analysis,
            'rag_queries': {'target': selection_query, 'option': selection_query},
            'optimization_strategy': option,
            'processing_time': time.time() - start_time
        }

        # Add fix/unfix report data
        unified_analysis.update(self._extract_fix_unfix_reports(reports))

        # Preserve iteration history from previous state
        if 'iteration_history' in state.get('unified_analysis', {}):
            unified_analysis['iteration_history'] = state['unified_analysis']['iteration_history']

        # ===== STEP 5: Update state =====
        state["unified_analysis"] = unified_analysis
        state["rag_queries"] = selection_query  # Last query for backward compatibility
        state["rag_retrieved_content_1"] = rag_retrieved_content
        state["rag_retrieved_content_2"] = rag_retrieved_content
        state["selected_route"] = target
        state["optimization_strategy"] = option

        # ===== STEP 6: Log final response =====
        response = LLMResponse(
            agent_type="SummaryAgent",
            timestamp=datetime.now(),
            input_data={
                'reports_processed': list(reports.keys()),
                'iteration': iteration,
                'has_dialogue_history': len(state['messages']) > 0,
                'selected_route': target,
                'optimization_strategy': option,
                'evaluation_on_target': evaluation_on_selection
            },
            output_data=unified_analysis,
            processing_time=time.time() - start_time,
            iteration=iteration
        )
        self.logger.log_response(response)

        return state
    
    def _route_to_command_generator(self, state: ECOState) -> str:
        """Route to the appropriate command generator based on analyze_reports decision"""
        return state["selected_route"]
    
    def _summarize_reports(self, reports: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of reports for archival in iteration history"""
        summary = {}
        
        qor = reports['timing']
        summary['timing'] = {
            'setup_violations': qor['setup_violating_paths'],
            'hold_violations': qor['hold_violating_paths'],
            'setup_slack': qor['setup_critical_path_slack'],
            'hold_slack': qor['hold_critical_path_slack'],
        }
        
        area = reports['area']
        summary['area'] = {
            'design_area': area['design_area'],
        }

        power = reports['power']
        summary['power'] = {
            'total_power': power['total_power'],
            'dynamic_power': power['total_power'] - power['leakage_power'],
            'leakage_power': power['leakage_power']
        }

        if 'area_fix' in reports:
            eco_data = reports['area_fix']
            eco_unfix_data = reports['area_unfix']
            summary['last_fix'] = {
                'type': 'area',
                'total_area_decreased': eco_data['total_area_decreased'],
                'percentage_area_decreased': eco_data['percentage_area_decreased'],
                'elapsed_time': eco_data['elapsed_time_seconds']
            }
            summary['last_unfix'] = eco_unfix_data
        elif 'timing_fix' in reports:
            eco_data = reports['timing_fix']
            eco_unfix_data = reports['timing_unfix']
            summary['last_fix'] = {
                'type': 'timing',
                'endpoints_found': eco_data['total_violating_endpoints_found'],
                'endpoints_fixed': eco_data['total_violating_endpoints_fixed'],
                'fix_percentage': eco_data['percentage_violations_fixed']
            }
            summary['last_unfix'] = {"fix_type": "timing", "violations_summary": eco_unfix_data['summary']}
        elif 'power_fix' in reports:
            eco_data = reports['power_fix']
            eco_unfix_data = reports['power_unfix']
            summary['last_fix'] = {
                'type': 'power',
                'total_power_decreased': eco_data['total_power_decreased'],
                'total_area_decreased': eco_data['total_area_decreased'],
                'elapsed_time': eco_data['elapsed_time_seconds']
            }
            summary['last_unfix'] = eco_unfix_data
        
        return summary
    
    def _build_timing_user_prompt(self, state: ECOState) -> str:
        """Build user prompt with current design state, optimization history, and unfixable reasons"""
        user_input_parts = []
        objectives = state["objectives"]
        user_input_parts.append("**OBJECTIVES:**")
        user_input_parts.append(objectives)
        user_input_parts.append("")

        # Add current design state
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

        # Add optimization history (only previous iterations, not current)
        user_input_parts.append("**OPTIMIZATION HISTORY:**")
        if len(self.design_state_history) <= 1:
            user_input_parts.append("No previous optimization history available.")
        elif len(self.design_state_history) > 1:
            user_input_parts.append("Optimization history listed in chronological order.")

            # Collect all historical data into lists for compact table-like display
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

            for hist_entry in self.design_state_history[:-1]:  # Exclude current iteration
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

        # Add unfixable history
        if self.unfixable_analysis_history:
            user_input_parts.append("**UNFIXABLE Reasons HISTORY:**")
            if TOOL_USING:
                user_input_parts.append("Explanation for reason_distribution: The unfixable reasons are weighted by the negative slack they causes. Explanation for slack_distribution: The number of paths in each negative slack range.\n")
            for unfixable_entry in self.unfixable_analysis_history:
                unfixable_iteration = unfixable_entry['iteration']
                unfixable_data = unfixable_entry['unfixable_reasons']
                unfixable_str = json.dumps(unfixable_data).replace('{', '{{').replace('}', '}}')
                user_input_parts.append(f"Iteration {unfixable_iteration}: {unfixable_str}")
            user_input_parts.append("")

        user_input_parts.append("**REFLECTION on command usage:**")
        user_input_parts.append(self.short_term_reflection)
        user_input_parts.append("")

        return "\n".join(user_input_parts)

    def _generate_timing_command_node(self, state: ECOState) -> ECOState:
        """Generate timing command using LangChain with two-step RAG approach:
        1. Generate evaluation of timing trends
        2. Retrieve relevant knowledge and generate final command
        """
        start_time = time.time()

        # Extract data from state
        unified_analysis = state["unified_analysis"]
        current_state = unified_analysis['current_state']
        llm_summary = unified_analysis['llm_summary']
        reasoning = llm_summary['reasoning']
        strategy = llm_summary['option']  # Changed from 'option_strategy' to 'option'
        objectives = state["objectives"]


        setup_violations = current_state['timing']['setup_violating_paths']
        hold_violations = current_state['timing']['hold_violating_paths']

        # ===== STEP 1: Generate evaluation of timing trends =====
        # System prompt for evaluation generation
        evaluation_system_prompt = """
**Background**: You are an expert in multi-iteration Engineering Change Order (ECO) for IC design with limited iteration budget. You will be given optimization trends, including current design state, optimization history, and unfixable issues. You will be also given the optimization strategy and objectives. You will generate evaluation on current timing optimization trends and unfixable reasons to help making informed decisions.

**Report Content**:
- CURRENT DESIGN STATE: The current design state includes timing, power, and area metrics after the most recent optimization iteration.
- OPTIMIZATION HISTORY: The optimization history includes the design states and already performed optimization commands from all previous iterations.
- UNFIXABLE ISSUES HISTORY: The unfixable issues history includes the design states and reasons for unfixable violations from all previous iterations.
- OBJECTIVES: The design objectives for the ECO run, including timing, power, and area goals.

**Strategy**:
- Strategy: Exploration or Exploitation.
    -- Exploration means explore different optimization command options and combinations to cover more possibilities. It can also reveal unfixable reasons for future iterations.
    -- Exploitation means understand the optimization history and unfixable reasons to target the potential fixing options. Also avoid repeating the previous optimization attempts that has no effect if metrics did not change due to the tradeoff between timing and power. If tradeoff happened, you can still refer to the command that cause the trade off and select target.


**Task**: Based on your analysis, generate Your evaluation on the optimization trend of timing, power, area from previous iterations. (Do not repeat detailed values from reports in the evaluation):
**Evaluation Format**: Given [your evaluation in plain text], what is the timing fixing guidelines and opt_timing command options?

**Evaluation Content** (You do not need to cover all of them. You can only cover the ones you found in reports):
1. Evaluate the trend of timing. Use the best achieved value (the default is iteration 0) as the optimization baseline.
2. Evaluate the current existing unfixable reasons.
3. Evaluate the effectiveness of previous optimization histories based on the timing's change and the unfixable reasons.
4. Evaluate the trade-off between setup and hold violations.
5. Evaluate whether previous explorations have revealed new unfixable reasons.
**Evaluation Guidelines**:
1. Have seperate evaluation for setup and hold violations. For unfixable reasons, you can know whether they belong to setup or hold from the command executed at that iteration.
2. When evaluating trends, mind the trade-off between timing, power, and area across optimization iterations.
3. Do not mention specific numerical values from reports. Do note mention detailed optimization actions like gate_sizing.

**COMMAND FORMAT:**
Command format: opt_timing -violation_type [setup|hold] -cell_class [cell_class] -actions [method] -site_mode [mode] (-area_cap [x]) (-slack_above [slack_limit]) (-slack_below [slack_limit])
Available type: setup or hold, you can only choose one of them.
Available actions for setup: gate_sizing, gate_sizing_side_load, buffer_insertion, you can select one method each time.
Available actions for hold: gate_sizing, buffer_insertion, you can select one method each time.
Available cell types: combinational, sequential, clock_tree, you can only choose one of them.
Available physical modes: open_slot, occupied_slot, you can only choose one of them.
Optional -area_cap.
Optional -slack_above slack_limit1.
Optional: -slack_below slack_limit2.

**DEFINITIONS OF UNFIX REASONS and potential solutions:** You only need to focus on timing related reasons.
A - There are available library cells outside area limit -> Increase the area limit
B - Delay improvement is too small to fix the violation -> Try other option combinations
C - The violation is in clock tree -> Target clock_tree cell type
D - Cell or net is located in high density area -> Unsolveable
I - Buffer insertion with given library cells cannot fix the violation -> Consider cell sizing
L - Available physical area limits the use of one or more library cells -> Unsolvable, try other option combinations
O - No open free site is available -> Try occupied_slot mode
R - No locations are available in parasitics or location transformation failed -> Unsolveable
S - Cell sizing with alternative library cells cannot fix the violation -> Try other options instead of sizing
T - Timing margin is too tight to fix the violation -> Try other option option combinations
W - Fixing the violation might degrade DRC violations -> add -ignore_drc to ignore DRC violations in the command

**RESPONSE FORMAT**: Always respond in plain text.
"""
        # Build user input with strategy, reasoning, and historical data
        evaluation_user_input = f"Strategy: {strategy}\nObjectives: {objectives}\n\n"
        evaluation_user_input += self._build_timing_user_prompt(state)
        evaluation_user_input += "\n**Now You Can generate the evaluation in plain text, no more than 100 words.:**\n"


        # First LLM call: Generate evaluation
        prompt = ChatPromptTemplate.from_messages([
            ("system", evaluation_system_prompt),
            ("human", evaluation_user_input)
        ])

        evaluation_messages = prompt.format_messages()
        evaluation_messages_sent = [{"role": msg.type, "content": msg.content} for msg in evaluation_messages]

        llm_start_time = time.time()
        evaluation_response = self.llm_timing.invoke(evaluation_messages)
        evaluation_time = time.time() - llm_start_time
        evaluation_token_usage = {
            "input_tokens": evaluation_response.usage_metadata["input_tokens"],
            "output_tokens": evaluation_response.usage_metadata["output_tokens"]
        }

        # # Extract and parse evaluation
        # evaluation_content_text = extract_content_from_llm_response(evaluation_response.content)
        # evaluation_json = extract_json_from_thinking_response(evaluation_content_text)
        # parser = JsonOutputParser()
        # evaluation_result = parser.parse(evaluation_json)
        # evaluation = evaluation_result['evaluation']
        
        evaluation = extract_content_from_llm_response(evaluation_response.content)
        # Log first LLM interaction
        interaction = LLMInteraction(
            agent_type="TimingCommandGenerator_Evaluation",
            timestamp=datetime.now(),
            iteration=state["iteration"],
            round_index=self.round_index,
            messages_sent=evaluation_messages_sent,
            response_received={"content": evaluation, "type": "json"},
            processing_time=evaluation_time,
            model_name=self.llm_timing.model_name,
            token_usage=evaluation_token_usage
        )
        self.logger.log_llm_interaction(interaction)

        # Accumulate LLM time
        self.current_iteration_llm_time += evaluation_time

        # ===== STEP 2: RAG retrieval =====
        from rag_helpers import retrieve_knowledge
        rag_retrieved_content = retrieve_knowledge(["Timing Fixing Evaluations: "+ evaluation], self.rag_system_timing)
        # ===== STEP 3: Generate final command with RAG knowledge =====
        # System prompt for command generation with RAG knowledge
        command_system_prompt = f"""You are an expert in IC design timing fixing responsible for fixing timing violations.
Your task is to analyze the current timing violations and generate an effective command based on: 1. optimization strategies and objectives 2. The current design state and optimization history. 3. The previous unfixable analysis (if any).
Then generate an optimal opt_timing command.

**COMMAND FORMAT:**
Command format: opt_timing -violation_type [setup|hold] -cell_class [cell_class] -actions [method] -site_mode [mode] (-area_cap [x]) (-slack_above [slack_limit1]) (-slack_below [slack_limit2])
Available type: setup or hold, you can only choose one of them.
Available actions for setup: gate_sizing, gate_sizing_side_load, buffer_insertion, you can select one method each time.
Available actions for hold: gate_sizing, buffer_insertion, you can select one method each time.
Available cell types: combinational, sequential, clock_tree, you can only choose one of them.
Available physical modes: open_slot, occupied_slot, you can only choose one of them.
Optional -area_cap.
Optional -slack_above slack_limit1.
Optional: -slack_below slack_limit2.

**DEFINITIONS OF UNFIX REASONS and potential solutions:**
A - There are available library cells outside area limit -> Increase the area limit
B - Delay improvement is too small to fix the violation -> Try other option combinations
C - The violation is in clock tree -> Target clock_tree cell type
D - Cell or net is located in high density area -> Unsolveable
I - Buffer insertion with given library cells cannot fix the violation -> Consider cell sizing
L - Available physical area limits the use of one or more library cells -> Unsolvable, try other option combinations
O - No open free site is available -> Try occupied_slot mode
R - No locations are available in parasitics or location transformation failed -> Unsolveable
S - Cell sizing with alternative library cells cannot fix the violation -> Try other options instead of sizing
T - Timing margin is too tight to fix the violation -> Try other option option combinations
W - Fixing the violation might degrade DRC violations -> add -ignore_drc to ignore DRC violations in the command

**RESPONSE FORMAT**: Always respond in valid JSON format.
"""

        # Build user input for command generation (reuse helper function)
        command_user_input = f"Strategy: {strategy}\nObjectives: {objectives}\n\n"
        command_user_input += self._build_timing_user_prompt(state)
        command_user_input += "\n**Guidelines:**\n"
        command_user_input += f"{rag_retrieved_content}\n"
        command_user_input += "\n**Now You Can generate the response with JSON resonse:**\n"
        command_user_input += "{{\n"
        command_user_input += '  "command": "The generated command"\n'
        command_user_input += "}}"

        # Second LLM call: Generate command
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

        # Extract and parse command
        command_content_text = extract_content_from_llm_response(command_response.content)
        command_json = extract_json_from_thinking_response(command_content_text)
        command_json = escape_unquoted_quotes_in_json(command_json)
        parser = JsonOutputParser()
        command_result_data = parser.parse(command_json)

        # Accumulate LLM time
        self.current_iteration_llm_time += command_time

        # Extract command
        command = command_result_data['command']

        # Log second LLM interaction
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

        # ===== STEP 4: Prepare final result =====
        command_result = {
            'command_type': ECOType.TIMING.value,
            'tcl_command': command
        }

        # Log final response
        response = LLMResponse(
            agent_type="TimingCommandGenerator",
            timestamp=datetime.now(),
            input_data={
                'setup_violations': setup_violations,
                'hold_violations': hold_violations,
                'strategy': strategy,
                'evaluation': evaluation
            },
            output_data=command_result,
            processing_time=time.time() - start_time,
            iteration=state["iteration"]
        )
        self.logger.log_response(response)

        # Store command result directly as selected command
        return {"selected_command": command_result, "command_proposals": {"timing": command_result}}
    
    def _build_power_user_prompt(self, state: ECOState) -> str:
        """Build user prompt with current design state, optimization history, and unfixable reasons"""
        user_input_parts = []
        objectives = state["objectives"]
        user_input_parts.append("**OBJECTIVES:**")
        user_input_parts.append(objectives)
        user_input_parts.append("")

        # Add current design state
        current_hist = self.design_state_history[-1] if self.design_state_history else None
        if current_hist:
            current_state_data = current_hist['design_state']
            remaining_iterations = self._calculate_remaining_iterations(current_hist['iteration'])

            current_state_parts = [
                f"Iteration: {current_hist['iteration']}, Remaining iterations: {remaining_iterations}",
                f"- Total power: {current_state_data['power']['total_power']:.3e}W",
                f"- Clock tree power: {current_state_data['power']['clock_tree_power']:.3e}W",
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

        # Add optimization history (only previous iterations, not current)
        user_input_parts.append("**OPTIMIZATION HISTORY:**")
        if len(self.design_state_history) <= 1:
            user_input_parts.append("No previous optimization history available.")
        elif len(self.design_state_history) > 1:
            user_input_parts.append("Optimization history listed in chronological order.")

            # Collect all historical data into lists for compact table-like display
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

            for hist_entry in self.design_state_history[:-1]:  # Exclude current iteration
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
                f"- Clock tree power: {hist_clock_tree_powers}W",
                f"- Register power: {hist_register_powers}W",
                f"- Combinational power: {hist_combinational_powers}W",
                f"- Design area: {hist_design_areas} um^2",
                f"- Remaining iterations: {hist_remaining_iterations}",
                f"- Actual execution times: {hist_execution_times} s",
                f"- Executed commands: {hist_executed_commands}"
            ]
            user_input_parts.append("\t".join(hist_parts))

        # Add unfixable history
        if self.unfixable_analysis_history:
            user_input_parts.append("**UNFIXABLE ISSUES HISTORY:**")
            if TOOL_USING:
                user_input_parts.append("\tExplanation for reason_distribution: The unfixable reasons are weighted by the negative slack they causes. Explanation for slack_distribution: The number of paths in each negative slack range.\n")
            for unfixable_entry in self.unfixable_analysis_history:
                unfixable_iteration = unfixable_entry['iteration']
                unfixable_data = unfixable_entry['unfixable_reasons']
                unfixable_str = json.dumps(unfixable_data).replace('{', '{{').replace('}', '}}')
                user_input_parts.append(f"Iteration {unfixable_iteration}: {unfixable_str}")
            user_input_parts.append("")

        user_input_parts.append("**REFLECTION on command usage:**")
        user_input_parts.append(self.short_term_reflection)
        user_input_parts.append("")

        return "\n".join(user_input_parts)

    def _generate_power_command_node(self, state: ECOState) -> ECOState:
        """Generate power command using LangChain with two-step RAG approach:
        1. Generate evaluation of power optimization trends
        2. Retrieve relevant knowledge and generate final command
        """
        start_time = time.time()

        # Extract data from state
        unified_analysis = state["unified_analysis"]
        current_state = unified_analysis['current_state']
        llm_summary = unified_analysis['llm_summary']
        reasoning = llm_summary['reasoning']
        strategy = llm_summary['option']  # Changed from 'option_strategy' to 'option'

        power_metrics = current_state['power']
        total_power = power_metrics['total_power']
        objectives = state["objectives"]
        # ===== STEP 1: Generate evaluation of power optimization trends =====
        # System prompt for evaluation generation
        evaluation_system_prompt = """
**Background**: You are an expert in multi-iteration Engineering Change Order (ECO) for IC design with limited iteration budget. You will be given optimization trends, including current design state, optimization history, and unfixable issues. You will be also given the optimization strategy and objectives. You will generate evaluation on current power optimization trends and unfixable reasons to help making informed decisions.

**Report Content**:
- CURRENT DESIGN STATE: The current design state includes timing, power, and area metrics after the most recent optimization iteration.
- OPTIMIZATION HISTORY: The optimization history includes the design states and already performed optimization commands from all previous iterations.
- UNFIXABLE ISSUES HISTORY: The unfixable issues history includes the design states and reasons for unfixable violations from all previous iterations.
- OBJECTIVES: The design objectives for the ECO run, including timing, power, and area goals.

**Strategy**:
- Strategy: Exploration or Exploitation.
    -- Exploration means explore different optimization command options and combinations to cover more possibilities. It can also reveal unfixable reasons for future iterations.
    -- Exploitation means understand the optimization history and unfixable reasons to target the potential fixing options. Also avoid repeating the previous optimization attempts that has no effect if metrics did not change due to the tradeoff between timing and power. If tradeoff happened, you can still refer to the command that cause the trade off and select target.

**Task**: Based on your analysis, generate Your evaluation on the optimization trend of power from previous iterations. (Do not repeat detailed values from reports in the evaluation):
**Evaluation Format**: Given [your evaluation in plain text], what is the power fixing guidelines?

**Evaluation Content** (You do not need to cover all of them. You can only cover the ones you found in reports):
1. Evaluate the trend of power (total, dynamic, leakage). Use the best achieved value (the default is iteration 0) as the optimization baseline.
2. Evaluate the current existing unfixable reasons.
3. Evaluate the effectiveness of previous optimization histories based on the power's change and the unfixable reasons.
4. Evaluate the trade-off between power and timing/area.
5. Evaluate whether previous explorations have revealed new unfixable reasons.
**Evaluation Guidelines**:
1. When evaluating trends, mind the trade-off between timing, power, and area across optimization iterations
2. Do not mention specific numerical values from reports. Do note mention detailed command options like gate_sizing.

**COMMAND FORMAT:**
Command format: opt_power -actions [method] -cell_class [cell_class] -power_scope [mode] (-setup_guard [setup_guard])
Available actions: gate_sizing, buffer_removal
Available power_scope: total | dynamic | leakage. Choose one of them.
Available cell types: [combinational, sequential] choose any one from them.
Note: buffer_removal cannot be used with other options like -cell_class.
Optional: -setup_guard setup_guard.

**DEFINITIONS OF UNFIX REASONS and potential solutions:**
B - Benefit from sizing cell is too small -> Try other options
L - Physical constraints restrict sizing -> Try other options
S - Cell has no alternate library cell with better power -> Try other options
T - Sizing cell might degrade timing -> Try other options
W - Sizing cell might degrade DRC -> Try other options
X - Cell is unusable for ECO -> Try other options
Z - Cell is sized -> Try other options

**RESPONSE FORMAT**: Always respond in plain text.
"""
        # Build user input with strategy, reasoning, and historical data
        evaluation_user_input = f"Strategy: {strategy}\n"
        evaluation_user_input += self._build_power_user_prompt(state)
        evaluation_user_input += "\n**Now You Can generate the evaluation in plain text:**\n"


        # First LLM call: Generate evaluation
        prompt = ChatPromptTemplate.from_messages([
            ("system", evaluation_system_prompt),
            ("human", evaluation_user_input)
        ])

        evaluation_messages = prompt.format_messages()
        evaluation_messages_sent = [{"role": msg.type, "content": msg.content} for msg in evaluation_messages]

        llm_start_time = time.time()
        evaluation_response = self.llm_power.invoke(evaluation_messages)
        evaluation_time = time.time() - llm_start_time
        evaluation_token_usage = {
            "input_tokens": evaluation_response.usage_metadata["input_tokens"],
            "output_tokens": evaluation_response.usage_metadata["output_tokens"]
        }

        # # Extract and parse evaluation
        # evaluation_content_text = extract_content_from_llm_response(evaluation_response.content)
        # evaluation_json = extract_json_from_thinking_response(evaluation_content_text)
        # parser = JsonOutputParser()
        # evaluation_result = parser.parse(evaluation_json)
        # evaluation = evaluation_result['evaluation']

        evaluation = extract_content_from_llm_response(evaluation_response.content)
        # Log first LLM interaction
        interaction = LLMInteraction(
            agent_type="PowerCommandGenerator_Evaluation",
            timestamp=datetime.now(),
            iteration=state["iteration"],
            round_index=self.round_index,
            messages_sent=evaluation_messages_sent,
            response_received={"content": evaluation, "type": "json"},
            processing_time=evaluation_time,
            model_name=self.llm_power.model_name,
            token_usage=evaluation_token_usage
        )
        self.logger.log_llm_interaction(interaction)

        # Accumulate LLM time
        self.current_iteration_llm_time += evaluation_time

        # ===== STEP 2: RAG retrieval =====
        from rag_helpers import retrieve_knowledge
        rag_retrieved_content = retrieve_knowledge(["Power Fixing Evaluations: "+ evaluation], self.rag_system_power)

        # ===== STEP 3: Generate final command with RAG knowledge =====
        # System prompt for command generation with RAG knowledge
        command_system_prompt = f"""You are an expert in IC design power optimization responsible for generating optimal opt_power commands.
Your task is to analyze the current power consumption and generate an effective command based on: 1. optimization strategies and objectives 2. The current design state and optimization history. 3. The previous unfixable analysis (if any).
Then generate an optimal opt_power command.
Note: The reduction of power and reduction of area are usually accompanied with each other.

**DEFINITIONS OF UNFIX REASONS and potential solutions:**
B - Benefit from sizing cell is too small -> Try other options
L - Physical constraints restrict sizing -> Try other options
S - Cell has no alternate library cell with better power -> Try other options
T - Sizing cell might degrade timing -> Try other options
W - Sizing cell might degrade DRC -> Try other options
X - Cell is unusable for ECO -> Try other options
Z - Cell is sized -> Try other options

**COMMAND FORMAT:**
Command format: opt_power -actions [method] -cell_class [cell_class] -power_scope [mode] (-setup_guard [setup_guard])
Available actions: gate_sizing, buffer_removal
Available power_scope: total | dynamic | leakage. Choose one of them.
Available cell types: [combinational, sequential] choose any one from them.
Note: buffer_removal cannot be used with other options like -cell_class.
Optional: -setup_guard setup_guard.

**RESPONSE FORMAT**: Always respond in valid JSON format.
"""

        # Build user input for command generation (reuse helper function)
        command_user_input = f"Strategy: {strategy}\nObjectives: {objectives}\n\n"
        command_user_input += self._build_power_user_prompt(state)
        command_user_input += "\n**Guidelines:**\n"
        command_user_input += f"{rag_retrieved_content}\n"
        command_user_input += "\n**Now You Can generate the response with JSON resonse:**\n"
        command_user_input += "{{\n"
        command_user_input += '  "command": "The generated command"\n'
        command_user_input += "}}"

        # Second LLM call: Generate command
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

        # Extract and parse command
        command_content_text = extract_content_from_llm_response(command_response.content)
        command_json = extract_json_from_thinking_response(command_content_text)
        command_json = escape_unquoted_quotes_in_json(command_json)
        parser = JsonOutputParser()
        command_result_data = parser.parse(command_json)

        # Accumulate LLM time
        self.current_iteration_llm_time += command_time

        # Extract command
        command = command_result_data['command']

        # Log second LLM interaction
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

        # ===== STEP 4: Prepare final result =====
        command_result = {
            'command_type': ECOType.POWER.value,
            'tcl_command': command
        }

        # Log final response
        response = LLMResponse(
            agent_type="PowerCommandGenerator",
            timestamp=datetime.now(),
            input_data={
                'total_power': total_power,
                'strategy': strategy,
                'evaluation': evaluation
            },
            output_data=command_result,
            processing_time=time.time() - start_time,
            iteration=state["iteration"]
        )
        self.logger.log_response(response)

        # Store command result directly as selected command
        return {"selected_command": command_result, "command_proposals": {"power": command_result}}
    def _build_area_user_prompt(self, state: ECOState) -> str:
        """Build user prompt with current design state, optimization history, and unfixable reasons"""
        user_input_parts = []
        objectives = state["objectives"]
        user_input_parts.append("**OBJECTIVES:**")
        user_input_parts.append(objectives)
        user_input_parts.append("")

        # Add current design state
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

        # Add optimization history (only previous iterations, not current)
        user_input_parts.append("**OPTIMIZATION HISTORY:**")
        if len(self.design_state_history) <= 1:
            user_input_parts.append("No previous optimization history available.")
        elif len(self.design_state_history) > 1:
            user_input_parts.append("Optimization history listed in chronological order.")

            # Collect all historical data into lists for compact table-like display
            hist_iterations = []
            hist_design_areas = []
            hist_total_powers = []
            hist_remaining_iterations = []
            hist_execution_times = []
            hist_executed_commands = []

            for hist_entry in self.design_state_history[:-1]:  # Exclude current iteration
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

        # Add unfixable history
        if self.unfixable_analysis_history:
            user_input_parts.append("**UNFIXABLE REASON HISTORY:**")
            if TOOL_USING:
                user_input_parts.append("Explanation for reason_distribution: The unfixable reasons are weighted by the negative slack they causes. Explanation for slack_distribution: The number of paths in each negative slack range.\n")
            for unfixable_entry in self.unfixable_analysis_history:
                unfixable_iteration = unfixable_entry['iteration']
                unfixable_data = unfixable_entry['unfixable_reasons']
                unfixable_str = json.dumps(unfixable_data).replace('{', '{{').replace('}', '}}')
                user_input_parts.append(f"Iteration {unfixable_iteration}: {unfixable_str}")
            user_input_parts.append("")

        user_input_parts.append("**REFLECTION on command usage:**")
        user_input_parts.append(self.short_term_reflection)
        user_input_parts.append("")

        return "\n".join(user_input_parts)

    def _generate_area_command_node(self, state: ECOState) -> ECOState:
        """Generate area command using LangChain with two-step RAG approach:
        1. Generate evaluation of area optimization trends
        2. Retrieve relevant knowledge and generate final command
        """
        start_time = time.time()

        # Extract data from state
        unified_analysis = state["unified_analysis"]
        current_state = unified_analysis['current_state']
        llm_summary = unified_analysis['llm_summary']
        reasoning = llm_summary['reasoning']
        strategy = llm_summary['option']  # Changed from 'option_strategy' to 'option'

        area_state = current_state['area']
        objectives = state["objectives"]
        # ===== STEP 1: Generate evaluation of area optimization trends =====
        # System prompt for evaluation generation
        evaluation_system_prompt = """
**Background**: You are an expert in multi-iteration Engineering Change Order (ECO) for IC design with limited iteration budget. You will be given optimization trends, including current design state, optimization history, and unfixable issues. You will be also given the optimization strategy and reasons from an ECO scheduling expert. You will generate evaluation on current area optimization trends and unfixable reasons to help making informed decisions.

**Report Content**:
- CURRENT DESIGN STATE: The current design state includes timing, power, and area metrics after the most recent optimization iteration.
- OPTIMIZATION HISTORY: The optimization history includes the design states and already performed optimization commands from all previous iterations.
- UNFIXABLE ISSUES HISTORY: The unfixable issues history includes the design states and reasons for unfixable violations from all previous iterations.
- OBJECTIVES: The design objectives for the ECO run, including timing, power, and area goals.

**Strategy**:
- Strategy: Exploration or Exploitation.
    -- Exploration means explore different optimization command options and combinations to cover more possibilities. It can also reveal unfixable reasons for future iterations.
    -- Exploitation means understand the optimization history and unfixable reasons to target the potential fixing options. Also avoid repeating the previous optimization attempts that has no effect if metrics did not change due to the tradeoff between timing and power. If tradeoff happened, you can still refer to the command that cause the trade off and select target.


**Task**: Based on your analysis, generate Your evaluation on the optimization trend of area from previous iterations. (Do not repeat detailed values from reports in the evaluation):
**Evaluation Format**: Given [your evaluation in plain text], what is the area fixing guidelines?

**Evaluation Content**:
1. Evaluate the trend of area. Use the best achieved value (the default is iteration 0) as the optimization baseline.
2. Evaluate the current existing unfixable reasons.
3. Evaluate the effectiveness of previous optimization histories based on the area's change and the unfixable reasons.
4. Evaluate the trade-off between area and timing/power.
5. Evaluate whether previous explorations have revealed new unfixable reasons.
**Evaluation Guidelines**:
1. When evaluating trends, mind the trade-off between timing, power, and area across optimization iterations.
2. Do not mention specific numerical values from reports. Do note mention detailed optimization actions like gate_sizing.

**COMMAND FORMAT:**
Command format: opt_area -actions [method] -cell_class [cell_class] (-setup_guard [setup_guard])
Available actions: gate_sizing, buffer_removal
Available cell types: [combinational, sequential, clock_tree] choose one from them
Note: buffer_removal cannot be used with any other options, including -cell_class.
Optional: -setup_guard setup_guard

**DEFINITIONS OF UNFIX REASONS and potential solutions:**
B - Benefit from sizing cell is too small -> Try other options
L - Physical constraints restrict sizing -> Try other options
S - Cell has no alternate library cell with better power -> Try other options
T - Sizing cell might degrade timing -> Try other options
W - Sizing cell might degrade DRC -> Try other options
X - Cell is unusable for ECO -> Try other options
Z - Cell is sized -> Try other options

**RESPONSE FORMAT**: Always respond in plain text.
"""
        # Build user input with strategy, reasoning, and historical data
        evaluation_user_input = f"Strategy: {strategy}\nObjectives:{objectives}\n"
        evaluation_user_input += self._build_area_user_prompt(state)
        evaluation_user_input += "\n**Now You Can generate the evaluation in plain text:**\n"


        # First LLM call: Generate evaluation
        prompt = ChatPromptTemplate.from_messages([
            ("system", evaluation_system_prompt),
            ("human", evaluation_user_input)
        ])

        evaluation_messages = prompt.format_messages()
        evaluation_messages_sent = [{"role": msg.type, "content": msg.content} for msg in evaluation_messages]

        llm_start_time = time.time()
        evaluation_response = self.llm_area.invoke(evaluation_messages)
        evaluation_time = time.time() - llm_start_time
        evaluation_token_usage = {
            "input_tokens": evaluation_response.usage_metadata["input_tokens"],
            "output_tokens": evaluation_response.usage_metadata["output_tokens"]
        }

        # # Extract and parse evaluation
        # evaluation_content_text = extract_content_from_llm_response(evaluation_response.content)
        # evaluation_json = extract_json_from_thinking_response(evaluation_content_text)
        # parser = JsonOutputParser()
        # evaluation_result = parser.parse(evaluation_json)
        # evaluation = evaluation_result['evaluation']

        evaluation = extract_content_from_llm_response(evaluation_response.content)
        # Log first LLM interaction
        interaction = LLMInteraction(
            agent_type="AreaCommandGenerator_Evaluation",
            timestamp=datetime.now(),
            iteration=state["iteration"],
            round_index=self.round_index,
            messages_sent=evaluation_messages_sent,
            response_received={"content": evaluation, "type": "json"},
            processing_time=evaluation_time,
            model_name=self.llm_area.model_name,
            token_usage=evaluation_token_usage
        )
        self.logger.log_llm_interaction(interaction)

        # Accumulate LLM time
        self.current_iteration_llm_time += evaluation_time

        # ===== STEP 2: RAG retrieval =====
        from rag_helpers import retrieve_knowledge
        rag_retrieved_content = retrieve_knowledge(["Area Fixing Evaluations: "+ evaluation], self.rag_system_area)

        # ===== STEP 3: Generate final command with RAG knowledge =====
        # System prompt for command generation with RAG knowledge
        command_system_prompt = f"""You are an expert in IC design area optimization responsible for generating optimal opt_area commands.
Your task is to analyze the current area consumption and generate an effective command based on: 1. optimization strategies and objectives 2. The current design state and optimization history. 3. The previous unfixable analysis (if any).
Then generate an optimal opt_area command.
Note: The reduction of power and area are usually accompanied with each other.

**DEFINITIONS OF UNFIX REASONS and potential solutions:**
B - Benefit from sizing cell is too small -> Try other options
L - Physical constraints restrict sizing -> Try other options
S - Cell has no alternate library cell with smaller area -> Try other options
T - Sizing cell might degrade timing -> Try other options
W - Sizing cell might degrade DRC -> Try other options
X - Cell is unusable for ECO -> Try other options
Z - Cell is sized -> Try other options

**COMMAND FORMAT:**
Command format: opt_area -actions [method] -cell_class [cell_class] (-setup_guard [setup_guard])
Available actions: gate_sizing, buffer_removal
Available cell types: [combinational, sequential, clock_tree] choose one from them
Note: buffer_removal cannot be used with any other options, including -cell_class.
Optional: -setup_guard setup_guard

**RESPONSE FORMAT**: Always respond in valid JSON format.
"""

        # Build user input for command generation (reuse helper function)
        command_user_input = f"Strategy: {strategy}\nObjectives: {objectives}\n\n"
        command_user_input += self._build_area_user_prompt(state)
        command_user_input += "\n**Guidelines:**\n"
        command_user_input += f"{rag_retrieved_content}\n"
        command_user_input += "\n**Now You Can generate the response with JSON resonse:**\n"
        command_user_input += "{{\n"
        command_user_input += '  "command": "The generated command"\n'
        command_user_input += "}}"

        # Second LLM call: Generate command
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

        # Extract and parse command
        command_content_text = extract_content_from_llm_response(command_response.content)
        command_json = extract_json_from_thinking_response(command_content_text)
        command_json = escape_unquoted_quotes_in_json(command_json)
        parser = JsonOutputParser()
        command_result_data = parser.parse(command_json)

        # Accumulate LLM time
        self.current_iteration_llm_time += command_time

        # Extract command
        command = command_result_data['command']

        # Log second LLM interaction
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

        # ===== STEP 4: Prepare final result =====
        command_result = {
            'command_type': ECOType.AREA.value,
            'tcl_command': command
        }

        # Log final response
        response = LLMResponse(
            agent_type="AreaCommandGenerator",
            timestamp=datetime.now(),
            input_data={
                'design_area': area_state['design_area'],
                'strategy': strategy,
                'evaluation': evaluation
            },
            output_data=command_result,
            processing_time=time.time() - start_time,
            iteration=state["iteration"]
        )
        self.logger.log_response(response)

        # Store command result directly as selected command
        return {"selected_command": command_result, "command_proposals": {"area": command_result}}

    def _execute_command_node(self, state: ECOState) -> ECOState:
        """Execute the selected command"""
        start_time = time.time()
        
        selected_command = state["selected_command"]
        tcl_command = selected_command['tcl_command']
        command_type = selected_command['command_type']
        
        if not tcl_command:
            execution_result = {
                'success': False,
                'message': 'No command to execute',
                'execution_time': 0,
                'command_type': command_type,
                'output': 'No TCL command provided'
            }
        else:
            execution_result = self._execute_tcl_command(tcl_command, command_type, state["iteration"])
        
        state["execution_result"] = execution_result
        
        # Log response
        response = LLMResponse(
            agent_type="CommandExecutor",
            timestamp=datetime.now(),
            input_data={'command_type': command_type, 'tcl_command': tcl_command},
            output_data=execution_result,
            processing_time=time.time() - start_time,
            iteration=state["iteration"]
        )
        self.logger.log_response(response)
        
        return state
    
    def _formulate_tcl_command(self, tcl_command: str, iteration: int) -> str:
        # Fill in the command and iteration variables
        # Use a different approach - replace the {} with command first, then handle ${} variables
        tcl_command = re.sub(r'\[(.*?)\]', lambda m: '{' + ' '.join(m.group(1).split(', ')) + '}', tcl_command)
        if "timing" in tcl_command:
            tcl_command+= " -verbose -unfixable_reasons_format text -unfixable_reasons_prefix {}/reports/unfix_timing_{}".format(base_path, iteration)

        # process buffer lists 
        if "buffer_insertion" in tcl_command:
            tcl_command += f" -buffer_list {BUFFER_LIST} "
        # add redirecting output
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
                area_cap_prefix = f"set_app_var area_cap {area_cap_value};\n"
                area_cap_suffix = "\nset_app_var area_cap 2;"
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
    def _execute_tcl_command(self, tcl_command: str, command_type: str, iteration: int) -> Dict[str, Any]:
        """Execute TCL command using pt_shell"""
        from report_parsers import parse_elapsed_time

        os.makedirs(os.path.join(base_path, 'logs'), exist_ok=True)

        # Read the execute_command.tcl template
        template_path = os.path.join("LLM_ECO", "EDA_scripts", "execute_command.tcl")
        try:
            with open(template_path, 'r') as f:
                template_content = f.read()
        except FileNotFoundError: 
            return {
                'success': False,
                'message': 'Template file not found',
                'execution_time': 0.0,
                'command_type': command_type,
                'output': f'Could not find template: {template_path}'
            }
        ori_tcl_command = copy.deepcopy(tcl_command)
        tcl_command = self._formulate_tcl_command(tcl_command, iteration)
        filled_content = template_content.replace('{}', tcl_command)
        filled_content = filled_content.replace('${i}', str(iteration))
        filled_content = filled_content.replace('${i+1}', str(iteration + 1))

        # Write the filled script
        round_dir = os.path.join(get_agent_dir(), f"round_{self.round_index}")
        os.makedirs(round_dir, exist_ok=True)
        script_path = os.path.join(round_dir, f'Run_scripts_{iteration}.tcl')
        with open(script_path, 'w') as f:
            f.write(filled_content)

        # Execute pt_shell
        log_path = os.path.join(base_path, f'logs/pt_{iteration}.log')
        command = f'cd {base_path} && pt_shell -f {script_path} | tee {log_path}'

        self.logger.logger.info(f"[Round {self.round_index}] Executing pt_shell command: {command}")
        exit_code = os.system(command)
        exit_code = 0 # for debugging

        # Read the log output
        try:
            with open(log_path, 'r') as f:
                output = f.read()
        except FileNotFoundError:
            output = "Log file not found"

        # Parse execution time from EDA tool report instead of wall-clock time
        execution_time = 0.0
        if exit_code == 0:
            # Determine report file path based on command type
            if 'timing' in command_type.lower():
                report_file = os.path.join(base_path, f'reports/fix_timing_{iteration}.txt')
            elif 'power' in command_type.lower():
                report_file = os.path.join(base_path, f'reports/fix_power_{iteration}.txt')
            elif 'area' in command_type.lower():
                report_file = os.path.join(base_path, f'reports/fix_area_{iteration}.txt')
            else:
                report_file = None

            # Parse execution time from report
            if report_file and os.path.exists(report_file):
                try:
                    with open(report_file, 'r') as f:
                        report_content = f.read()
                    execution_time = parse_elapsed_time(report_content)
                    self.logger.logger.info(f"[Round {self.round_index}] Parsed execution time from {report_file}: {execution_time}s")
                except Exception as e:
                    self.logger.logger.warning(f"Failed to parse execution time from {report_file}: {e}")
                    execution_time = 0.0

        # Determine success
        success = (exit_code == 0)
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

    def run_iteration(self, objectives, reports: Dict[str, str]) -> Dict[str, Any]:
        """Run one complete ECO iteration using persistent state"""
        self.iteration_count += 1
        iteration_start = time.time()

        # Reset LLM time accumulator for this iteration
        self.current_iteration_llm_time = 0.0

        # Calculate budget at iteration start for initial state
        elapsed_runtime_start = time.time() - self.start_time
        remaining_budget_start = self.total_runtime_budget - elapsed_runtime_start

        self.logger.logger.info(f"[Round {self.round_index}] Starting ECO iteration {self.iteration_count}")

        # First iteration: Create initial persistent state
        if self.persistent_state is None:
            self.logger.logger.info(f"[Round {self.round_index}] Creating initial ECO state for first iteration")
            self.persistent_state = ECOState(
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
            # Subsequent iterations: Update persistent state while preserving accumulated context
            self.logger.logger.info(f"[Round {self.round_index}] Updating persistent state for iteration {self.iteration_count}")

            # Preserve accumulated analysis and add new iteration key
            if 'iteration_history' not in self.persistent_state['unified_analysis']:
                self.persistent_state['unified_analysis']['iteration_history'] = {}

            # Archive previous iteration's data for context
            prev_iteration = self.iteration_count - 1
            if prev_iteration >= 0:
                self.persistent_state['unified_analysis']['iteration_history'][f'iteration_{prev_iteration}'] = {
                    'selected_command': self.persistent_state.get('selected_command', {}),
                    'execution_result': self.persistent_state.get('execution_result', {}),
                    'reports_summary': self._summarize_reports(self.persistent_state.get('reports', {}))
                }

            # Update state with new iteration data while preserving accumulated context
            # Preserve the messages from previous iteration
            preserved_messages = self.persistent_state.get('messages', [])

            self.persistent_state.update({
                'iteration': self.iteration_count,
                'reports': reports,
                'elapsed_runtime': elapsed_runtime_start,
                'remaining_budget': remaining_budget_start,
                'system_status': "running",
                'messages': preserved_messages,  # Explicitly preserve messages
                'objectives': objectives
                # NOTE: unified_analysis, command_proposals are preserved!
                # selected_command and execution_result will be updated by workflow
            })

        # Run the workflow with persistent state
        # config = {"configurable": {"thread_id": f"eco_optimization_persistent"}}  # Use same thread for continuity

        current_objectives = self.persistent_state['objectives']
        final_state = self.workflow.invoke(self.persistent_state)
        final_state['objectives'] = current_objectives

        # Update persistent state with final results
        self.persistent_state = final_state

        # Calculate iteration runtime based on: LLM time + EDA execution time
        eda_execution_time = final_state["execution_result"].get('execution_time', 0.0)
        iteration_runtime = self.current_iteration_llm_time + eda_execution_time

        # Store LLM time for this iteration
        self.iteration_llm_times[self.iteration_count] = self.current_iteration_llm_time

        # Calculate cumulative runtime (sum of all iteration runtimes, not wall-clock time)
        total_elapsed = sum(self.iteration_llm_times.values()) + sum(
            self.design_state_history[i].get('actual_execution_time', 0)
            for i in range(len(self.design_state_history) - 1)  # Exclude current iteration
        )
        total_elapsed += eda_execution_time  # Add current iteration's EDA time

        remaining_budget_final = self.total_runtime_budget - total_elapsed

        # Update state with final accurate budget values
        self.persistent_state['elapsed_runtime'] = total_elapsed
        self.persistent_state['remaining_budget'] = remaining_budget_final

        result = {
            "iteration": self.iteration_count,
            "unified_analysis": final_state["unified_analysis"],
            "command_proposals": final_state["command_proposals"],
            "selected_command": final_state["selected_command"],
            "execution_result": final_state["execution_result"],
            "iteration_time": time.time() - iteration_start,  # Wall-clock time for logging
            "iteration_runtime": iteration_runtime,  # LLM + EDA time
            "llm_time": self.current_iteration_llm_time,
            "eda_time": eda_execution_time,
            "total_elapsed": total_elapsed,
            "remaining_budget": remaining_budget_final,
            "system_status": "healthy" if remaining_budget_final > 0 else "budget_exceeded"
        }

        self.logger.logger.info(
            f"[Round {self.round_index}] ECO iteration {self.iteration_count} completed: "
            f"LLM={self.current_iteration_llm_time:.1f}s, "
            f"EDA={eda_execution_time:.1f}s, "
            f"Total={iteration_runtime:.1f}s"
        )
        return result
    
    def should_continue_optimization(self) -> bool:
        """Determine if optimization should continue"""
        elapsed_time = time.time() - self.start_time
        remaining_budget = self.total_runtime_budget - elapsed_time
        
        if remaining_budget <= 0:
            self.logger.logger.info(f"[Round {self.round_index}] Optimization stopped: Runtime budget exhausted")
            return False
            
        if self.iteration_count >= 20:  # Arbitrary limit
            self.logger.logger.info(f"[Round {self.round_index}] Optimization stopped: Maximum iterations reached")
            return False
            
        return True

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        elapsed_time = time.time() - self.start_time
        remaining_budget = self.total_runtime_budget - elapsed_time
        
        return {
            "iteration_count": self.iteration_count,
            "elapsed_time": elapsed_time,
            "remaining_budget": remaining_budget,
            "total_runtime_budget": self.total_runtime_budget,
            "system_health": "healthy" if remaining_budget > 0 else "budget_exhausted",
            "eco_flow_phase": "ready" if self.iteration_count == -1 else "active"
        }
def load_reports_for_iteration(iteration: int, last_command_type: ECOType = None, last_reports: dict = {}, last_executed_command: str = "", tool_using: bool = False, reports_base_path: str = None) -> Dict[str, Any]:
    """Load and parse reports for a given iteration using eco_database ReportParser"""
    path_base = reports_base_path or base_path
    if iteration > 0:
        last_power_data = copy.deepcopy(last_reports["power"])
    parsed_reports = {}
    
    if iteration == 0:
        # First iteration: baseline reports only
        report_files = {
            "qor": os.path.join(path_base, "reports/report_qor_0.txt"),
            "power": os.path.join(path_base, "reports/report_power_0.txt")
        }
        print(f"Loading baseline reports for iteration {iteration}...")
    else:
        # Subsequent iterations: baseline + ECO fixing reports based on last command
        report_files = {
            "qor": os.path.join(path_base, f"reports/report_qor_{iteration}.txt"),
            "power": os.path.join(path_base, f"reports/report_power_{iteration}.txt")
        }
        
        # Add ECO fixing reports based on last command type
        # Use previous iteration index for ECO fix reports
        prev_iteration = iteration - 1
        if 'opt_area' in last_executed_command:
            report_files["area_fix"] = os.path.join(path_base, f"reports/fix_area_{prev_iteration}.txt")
            # Check for unfixing report
            unfix_file = os.path.join(path_base, f"reports/fix_area_{prev_iteration}.txt")
        elif 'opt_timing' in last_executed_command:
            report_files["timing_fix"] = os.path.join(path_base, f"reports/fix_timing_{prev_iteration}.txt")
            # Check for unfixing report
            unfix_file = os.path.join(path_base, f"reports/unfix_timing_{prev_iteration}_eco_tim.txt")
            report_files["timing_unfix"] = unfix_file
        elif 'opt_power' in last_executed_command:
            report_files["power_fix"] = os.path.join(path_base, f"reports/fix_power_{prev_iteration}.txt")
            unfix_file = os.path.join(path_base, f"reports/fix_power_{prev_iteration}.txt")

        last_command_label = last_command_type.value if last_command_type else "None"
        print(f"Loading reports for iteration {iteration} (last command: {last_command_label})")
        print(f"Report files: {list(report_files.keys())}")
    
    
    # Load and parse each report
    for report_type, file_path in report_files.items():
        
        
        # Parse reports using eco_database ReportParser
        if report_type == "qor":
            with open(file_path, 'r') as f:
                content = f.read()
            parsed_qor = ReportParser.parse_qor_report(content)
            parsed_reports["timing"] = {"setup_violating_paths": parsed_qor["setup_violating_paths"], "hold_violating_paths": parsed_qor["hold_violating_paths"], "setup_critical_path_slack": parsed_qor["setup_critical_path_slack"], "hold_critical_path_slack": parsed_qor["hold_critical_path_slack"], "setup_total_negative_slack": parsed_qor["setup_total_negative_slack"], "hold_total_negative_slack": parsed_qor["hold_total_negative_slack"], "clock_period": parsed_qor["clock_period"]}
            parsed_reports['area'] = {"design_area": parsed_qor["design_area"]}
            
        elif report_type == "power":
            with open(file_path, 'r') as f:
                content = f.read()
            parsed_reports["power"] = ReportParser.parse_power_report(content)
    if iteration > 0:
        if 'opt_timing' in last_executed_command:
            content = open(report_files['timing_fix'], 'r').read()
            parsed_reports["timing_fix"] = ReportParser.parse_timing_log(content)

            # Extract timing type from the last executed command
            type_match = re.search(r'-violation_type\s+(setup|hold)', last_executed_command, re.IGNORECASE)
            if not type_match:
                violation_type = 'undefined'
            else:
                violation_type = type_match.group(1).lower()

            # Check if timing unfix report exists
            if os.path.exists(report_files['timing_unfix']):
                unfix_content = open(report_files['timing_unfix'], 'r').read()
                parsed_reports["timing_unfix"] = ReportParser.parse_timing_unfixing(unfix_content, tool_using)
                # Add timing type to summary
                parsed_reports["timing_unfix"]["summary"]["type"] = violation_type
            else:
                # No unfixing report means all violations were fixed (0 violations)
                parsed_reports["timing_unfix"] = {
                    "fix_type": "timing",
                    "reason_definitions": {},
                    "timing_paths": [],
                    "paths_sorted_by_slack": [],
                    "summary": {
                        "type": violation_type,
                        "total_paths": 0,
                        "worst_slack": 0.0,
                        "reason_distribution": {}
                    }
                }
        elif 'opt_area' in last_executed_command:
            content = open(report_files['area_fix'], 'r').read()
            area_fix = ReportParser.parse_area_log(content)
            parsed_reports["area_fix"] = area_fix
            area_unfix = ReportParser.parse_area_power_unfixing(content)
            if area_unfix:
                area_unfix["fix_type"] = "area"
                parsed_reports["area_unfix"] = area_unfix
            else:
                parsed_reports["area_unfix"] = {"fix_type": "area", "violations_summary": "Removing buffers cannot give unfixable reasons."}
        elif 'opt_power' in last_executed_command:
            content = open(report_files['power_fix'], 'r').read()
            power_fix = ReportParser.parse_power_log(content)
            if not power_fix.get('report_format'):
                raise ValueError(f"Failed to determine report_format from power fix report. Report may not contain 'Final ECO Summary' or 'Fixing Summary' sections.")
            buffer_removal_used = "buffer_removal" in last_executed_command.lower()
            parsed_reports['power_fix'] = {
                'fix_type': 'power',
                'report_format': power_fix['report_format'],
                'elapsed_time_seconds': power_fix['elapsed_time_seconds'],
                'total_power_decreased': round(last_power_data['total_power'] - parsed_reports['power']['total_power'], 4),
                'leakage_power_decreased': round(last_power_data['leakage_power'] - parsed_reports['power']['leakage_power'], 4),
                'dynamic_power_decreased': round((last_power_data['total_power'] - last_power_data['leakage_power']) - (parsed_reports['power']['total_power'] - parsed_reports['power']['leakage_power']), 4),
                'total_area_decreased': power_fix['total_area_decreased'],
                'percentage_area_decreased': power_fix['percentage_area_decreased']
            }
            if buffer_removal_used and power_fix['report_format'] == 'fixing_summary':
                parsed_reports['power_fix']['buffers_removed'] = power_fix.get('buffers_removed', 0)
                parsed_reports['power_fix']['percentage_buffers_removed'] = power_fix.get('percentage_buffers_removed', 0.0)
                parsed_reports['power_fix']['percentage_cells_removed'] = power_fix.get('percentage_cells_removed', 0.0)
            power_unfix = ReportParser.parse_area_power_unfixing(content)
            if power_unfix:
                power_unfix["fix_type"] = "power"
                parsed_reports["power_unfix"] = power_unfix
            else:
                parsed_reports["power_unfix"] = {"fix_type": "power", "violations_summary": "Removing buffers cannot give unfixable reasons."}  

    
    return parsed_reports


def create_eco_system(runtime_budget: float = 3600, log_file: str = "eco_agent_responses_langchain.json") -> ECOLangChainSystem:
    """Factory function to create a configured ECO system"""
    return ECOLangChainSystem(runtime_budget, log_file)

def clean_run_dir(base_path):
    os.system(f"rm -rf {base_path}/reports/*")
    os.system(f"rm -rf {base_path}/logs/*")
    os.system(f"rm -rf {base_path}/Run_scripts_*")
    os.system(f'cd {base_path} && pt_shell -x "restore_session eco_session_0; report_qor > reports/report_qor_0.txt; report_power -nosplit > reports/report_power_0.txt; exit;"')


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


def _save_round_resume_state(
        agent_dir,
        design,
        round_index,
        num_iterations,
        trace_memory,
        results,
        short_term_reflection,
        long_term_reflection):
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
        "short_term_reflection": short_term_reflection,
        "long_term_reflection": long_term_reflection,
        "timestamp": datetime.now().isoformat()
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)
    with open(round_checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)
    return checkpoint_path, trace_memory_path


def _apply_design_config(design):
    global base_path

    design_config = configs.DESIGN_CONFIG_OVERRIDES[design]
    configs.base_path = design_config["base_path"]
    base_path = configs.base_path
    agent_run_paths.base_path = configs.base_path
    return design_config


class ShortTermReflectionAgent:
    """LLM agent for short-term reflection on Pareto-optimal traces."""

    def __init__(self, logger, model_name="gpt-4o-mini", use_openai_reasoning=USE_OPENAI_REASONING):
        self.logger = logger
        self.reflection_llm = create_openai_llm(model_name, use_openai_reasoning)

    def generate_short_term_reflection(self, objectives, pareto_traces, iteration, round_index):
        """Generate short-term reflection based on Pareto-optimal traces."""
        if pareto_traces is None:
            raise ValueError("pareto_traces is required for short-term reflection")

        lines = []
        for idx, trace in enumerate(pareto_traces, start=1):
            command_history, design_trace = trace
            last_idx = len(design_trace["iterations"]) - 1
            setup_tns = design_trace["setup_TNS_trace"][last_idx]
            setup_wns = design_trace["setup_WNS_trace"][last_idx]
            hold_tns = design_trace["hold_TNS_trace"][last_idx]
            hold_wns = design_trace["hold_WNS_trace"][last_idx]
            power = design_trace["total_power_trace"][last_idx]
            dynamic_power = design_trace["dynamic_power_trace"][last_idx]
            leakage_power = design_trace["leakage_power_trace"][last_idx]
            area = design_trace["area_trace"][last_idx]
            commands_text = "; ".join([str(cmd) for cmd in command_history])
            design_trace_parts = [
                f"iterations={design_trace['iterations']}",
                f"setup_wns={design_trace['setup_WNS_trace']}",
                f"hold_wns={design_trace['hold_WNS_trace']}",
                f"setup_tns={design_trace['setup_TNS_trace']}",
                f"hold_tns={design_trace['hold_TNS_trace']}",
                f"total_power={design_trace['total_power_trace']}",
                f"dynamic_power={design_trace['dynamic_power_trace']}",
                f"leakage_power={design_trace['leakage_power_trace']}",
                f"design_area={design_trace['area_trace']}",
                f"executed_commands={design_trace['command_trace']}",
            ]
            design_trace_text = "; ".join(design_trace_parts)
            lines.append(
                f"Trace {idx}: design_state [{design_trace_text}]"
            )
        pareto_summary = "\n".join(lines) if lines else "No pareto traces available."

        command_guidelines = {
            "timing": """
Command format: opt_timing -violation_type [setup|hold] -cell_class [cell_class] -actions [method] -site_mode [mode] (-area_cap [area_cap]) (-slack_above [slack_limit]) (-slack_below [slack_limit])
Available type: setup or hold, you can only choose one of them.
Available actions for setup: gate_sizing, gate_sizing_side_load, buffer_insertion, you can select one method each time. The most common choice is gate_sizing, then buffer_insertion, then gate_sizing_side_load.
Available actions for hold: gate_sizing, buffer_insertion, you can select one method each time. The most common choice is gate_sizing, then buffer_insertion.
Available cell types: combinational, sequential, clock_tree, you can only choose one of them. If you do not choose combinational, then you can only use gate_sizing method. The most common cell type is combinational, then sequential, use clock_tree only if you are explicitly asked.
Available physical modes: open_slot, occupied_slot. The common selection is open_slot, you can explore occupied_slot if you find the timing optimization result is not promising.
Optional: -area_cap area_cap: Specify area increment limit for gate_sizing, only use it when gate_sizing option selected, default value is 2, available values are [4|8|10|12|16|20].
Optional: -slack_above slack_limit: By setting the slack limit to a positive value, the command will try to improve the timing of paths with slacks less than the slack_limit (more positive). This may increase area and power but can further improve timing. Use it when using very high effort to optimize timing. Do not use it at the end of optimization iterations when you do not have budget to do power recovery.
Optional: -slack_below slack_limit: By setting the slack limit to a negative value, the command will NOT try to improve the timing of paths with slacks larger than the slack_limit (more negative). This may limit the timing optimization but lead to reduced power and area incresing. Do not use it when you need high-effort timing optimization.
""",
            "power": """
Command format: opt_power -actions [method] -cell_class [cell_class] -power_scope [mode] (-setup_guard [setup_guard])
Available actions: gate_sizing, buffer_removal. Note: buffer_removal cannot be used with other options like -cell_class.
Available power_scope: total | dynamic | leakage. Total power optimization is more suitable at earlier stage or exploration to directly optimize power. dynamic and leakage can work on later exploitation stage when you find corresponding power component is not optimized much or  re-rised after timing optimization.
Available cell types: [combinational, sequential] you can choose any one from them. You can also choose this to target timing optimization on combinational or sequential cells. Usually, at earlier stage, can start with combinational.
Optional: -setup_guard setup_guard: Specify the setup timing margin to ensure that the timing optimization does not degrade timing beyond this margin. Default is 0. Set it as a negative value to allow timing degradation when you optimize power with high efforts.
""",
            "area": """
Command format: opt_area -actions [method] -cell_class [cell_class] (-setup_guard [setup_guard])
Available actions: gate_sizing, buffer_removal. Note: buffer_removal cannot be used with any other options, including -cell_class.
Available cell types: [combinational, sequential, clock_tree] you can choose any one from them. You can also choose this to target timing optimization on combinational or sequential cells. Usually, at earlier stage, can start with combinational.
Optional: -setup_guard setup_guard: Specify the setup timing margin to ensure that the timing optimization does not degrade timing beyond this margin. Default is 0. Set it as a negative value to allow timing degradation when you optimize area with high efforts.
"""
        }
        unfix_definitions = {
            "timing": """Remove for confidential reasons""",
            "area": """Remove for confidential reasons"""
        }

        system_prompt = (
            """You are an expert IC design ECO (Engineering Change Order) optimization engineer responsible for comprehensive analysis.
You will work as an reviewer engineer to evaluate the optimization history regarding timing, power, and area w.r.t the Objectives. The response should be less than 100 words.

**ECO Background**: ECO is an incremental design optimization process that iteratively improves the design by fixing violations and optimizing metrics such as timing, area, and power. Each iteration involves analyzing the current design state and optimization histories and the optimization of one target metric with specific optimization options. The success of ECO lies in: 1. Extensively explore optimization targets (timing, area, power); 2. Keep track of the tradeoff among 3 targets to ensure balanced optimization and do recovery of a metrics if it is deteriorated due to optimizing other metrics; 3. Do not waste time on unpromissing targets.
The timing slacks are the more positive the better, while area and power are the more lower the better. The optimization benefit tend to decrease as the proceeding of ECO iterations. The goal is to achieve the optimization objective.

**Response Guidelines**:\n
"""
    "You should analyze the trend of timing, power, area change in iterations related to objectives and the trend of unfixable reasons' change. \n"
    "You should comment on the effectiveness of command that optimize the same objective with different options.\n"
    "You should comment on command options relate to unfixable reasons and whether those options helped.\n"
        )
        user_prompt = (
            "Objectives:\n"
            f"{objectives}\n\n"
            "Pareto optimzal traces:\n"
            f"{pareto_summary}\n\n"
            "Timing command format:\n"
            f"{command_guidelines['timing']}\n"
            "Power command format:\n"
            f"{command_guidelines['power']}\n"
            "Area command format:\n"
            f"{command_guidelines['area']}\n"
            "Unfixable reason hints:\n"
            f"Timing: {unfix_definitions['timing']}\n"
            f"Power: {unfix_definitions['power']}\n"
            "Area: The same as power.\n"
            "Now give your comment."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        messages_sent = [
            {"role": "system", "content": system_prompt},
            {"role": "human", "content": user_prompt}
        ]

        llm_start = time.time()
        response = self.reflection_llm.invoke(messages)
        elapsed = time.time() - llm_start
        content = extract_content_from_llm_response(response.content).strip()
        token_usage = {
            "input_tokens": response.usage_metadata["input_tokens"],
            "output_tokens": response.usage_metadata["output_tokens"]
        }

        if self.logger is not None:
            interaction = LLMInteraction(
                agent_type="ShortTermReflectionAgent",
                timestamp=datetime.now(),
                iteration=iteration,
                round_index=round_index,
                messages_sent=messages_sent,
                response_received={"content": content, "type": "text"},
                processing_time=elapsed,
                model_name=self.reflection_llm.model_name,
                token_usage=token_usage
            )
            self.logger.log_llm_interaction(interaction)

        return content

class LongTermReflectionAgent:
    """LLM agent for long-term reflection on Pareto-optimal traces."""

    def __init__(self, logger, model_name="gpt-4o-mini", use_openai_reasoning=USE_OPENAI_REASONING):
        self.logger = logger
        self.reflection_llm = create_openai_llm(model_name, use_openai_reasoning)

    def generate_long_term_reflection(self, objectives, pareto_traces, iteration, round_index):
        """Generate long-term reflection based on Pareto-optimal traces."""
        if pareto_traces is None:
            raise ValueError("pareto_traces is required for long-term reflection")

        lines = []
        for idx, trace in enumerate(pareto_traces, start=1):
            command_history, design_trace = trace
            last_idx = len(design_trace["iterations"]) - 1
            setup_tns = design_trace["setup_TNS_trace"][last_idx]
            setup_wns = design_trace["setup_WNS_trace"][last_idx]
            hold_tns = design_trace["hold_TNS_trace"][last_idx]
            hold_wns = design_trace["hold_WNS_trace"][last_idx]
            power = design_trace["total_power_trace"][last_idx]
            dynamic_power = design_trace["dynamic_power_trace"][last_idx]
            leakage_power = design_trace["leakage_power_trace"][last_idx]
            area = design_trace["area_trace"][last_idx]
            commands_text = "; ".join([str(cmd) for cmd in command_history])
            design_trace_parts = [
                f"iterations={design_trace['iterations']}",
                f"setup_wns={design_trace['setup_WNS_trace']}",
                f"hold_wns={design_trace['hold_WNS_trace']}",
                f"setup_tns={design_trace['setup_TNS_trace']}",
                f"hold_tns={design_trace['hold_TNS_trace']}",
                f"total_power={design_trace['total_power_trace']}",
                f"dynamic_power={design_trace['dynamic_power_trace']}",
                f"leakage_power={design_trace['leakage_power_trace']}",
                f"design_area={design_trace['area_trace']}",
                f"executed_commands={design_trace['command_trace']}"
            ]
            design_trace_text = "; ".join(design_trace_parts)
            lines.append(
                f"Trace {idx}: commands [{commands_text}]; metrics [setup_tns={setup_tns}, setup_wns={setup_wns}, "
                f"hold_tns={hold_tns}, hold_wns={hold_wns}, total_power={power}, dynamic_power={dynamic_power}, "
                f"leakage_power={leakage_power}, area={area}];"
            )
        pareto_summary = "\n".join(lines) if lines else "No pareto traces available."

        system_prompt = (
"""You are an expert IC design ECO (Engineering Change Order) optimization engineer responsible for optimization history analysis.
You will work as an reviewer engineer to evaluate the optimization history regarding timing, power, and area w.r.t the Objectives. The response should be less than 100 words.

**ECO Background**: ECO is an incremental design optimization process that iteratively improves the design by fixing violations and optimizing metrics such as timing, area, and power. Each iteration involves analyzing the current design state and optimization histories and the optimization of one target metric with specific optimization options. The success of ECO lies in: 1. Extensively explore optimization targets (timing, area, power); 2. Keep track of the tradeoff among 3 targets to ensure balanced optimization and do recovery of a metrics if it is deteriorated due to optimizing other metrics; 3. Do not waste time on unpromissing targets.
The timing slacks are the more positive the better, while area and power are the more lower the better. The optimization benefit tend to decrease as the proceeding of ECO iterations. The goal is to achieve the optimization objective.

**Response Guidelines**:\n
"""
            "Produce a comment on optimization target ordering and effort selection Pareto frontier observations. Provide guidance only; no tool commands or lists.\n"
            "Each trace consists of a sequence of commands with opt_[timing/power/area] and metrics change.\n"
            "Focus on how objectives were selected and how many iterations were spent on each target.\n"
            "Comment on whether some targets need more or fewer iterations for optimization.\n"
            "Highlight whether any target needs recovery after being sacrificed when optimizing another target.\n"
        )
        user_prompt = (
            "Objectives:\n"
            f"{objectives}\n\n"
            "Pareto optimal traces:\n"
            f"{pareto_summary}\n\n"
            "In your comment, do not mention any tool commands or specific actions like gate_sizing. Focus on high-level strategy insights.\n"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        messages_sent = [
            {"role": "system", "content": system_prompt},
            {"role": "human", "content": user_prompt}
        ]

        llm_start = time.time()
        response = self.reflection_llm.invoke(messages)
        elapsed = time.time() - llm_start
        content = extract_content_from_llm_response(response.content).strip()
        token_usage = {
            "input_tokens": response.usage_metadata["input_tokens"],
            "output_tokens": response.usage_metadata["output_tokens"]
        }

        if self.logger is not None:
            interaction = LLMInteraction(
                agent_type="LongTermReflectionAgent",
                timestamp=datetime.now(),
                iteration=iteration,
                round_index=round_index,
                messages_sent=messages_sent,
                response_received={"content": content, "type": "text"},
                processing_time=elapsed,
                model_name=self.reflection_llm.model_name,
                token_usage=token_usage
            )
            self.logger.log_llm_interaction(interaction)

        return content

def simulate_eco_iterations(
        num_iterations: int = None,
        num_rounds: int = 1,
        design: str = "NV_NVDLA_partition_m",
        resume: bool = False,
        checkpoint_path: str = None) -> List[Dict[str, Any]]:
    """Test the LangChain ECO system."""
    design_config = _apply_design_config(design)
    system = create_eco_system(int(design_runtime_budget[design]))
    trace_memory = TraceMemory()
    reflection_agent = ShortTermReflectionAgent(system.logger)
    long_term_reflection_agent = LongTermReflectionAgent(system.logger)

    agent_dir = get_agent_dir()
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
        system.short_term_reflection = checkpoint["short_term_reflection"]
        system.long_term_reflection = checkpoint["long_term_reflection"]
        start_round_index = checkpoint["next_round_index"]
        print(f"Resuming from checkpoint {checkpoint_path}")
        print(f"Loaded {len(trace_memory.traces)} previous traces; next round is {start_round_index}")

    if num_iterations is None:
        num_iterations = design_config["max_iterations_per_trace"]

    print(f"ECO LangChain System Test Started")
    print(f"Running {num_iterations} iterations")
    print("=" * 70)

    objectives = design_config["objectives"]
    end_round_index = start_round_index + num_rounds
    results = []

    for round_index in range(start_round_index, end_round_index):
        system.round_index = round_index
        system.persistent_state = None
        system.iteration_count = -1
        system.current_iteration_llm_time = 0.0
        system.iteration_llm_times = {}
        system.design_state_history = []
        system.unfixable_analysis_history = []
        if round_index == 0 and not resume:
            system.long_term_reflection = ""
            system.short_term_reflection = ""
        system.start_time = time.time()
        results = []
        last_command_type = None  # Track the last executed command type
        last_executed_command = ""  # Track the last executed command text
        reports = {}
        for i in range(num_iterations):
            print(f"\nIteration {i}")
            print("-" * 50)
            # Load reports for this iteration, using last command type and command text from previous iteration
            reports = load_reports_for_iteration(i, last_command_type=last_command_type, last_reports=reports, last_executed_command=last_executed_command, tool_using=TOOL_USING)
            
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
            checkpoint_path, trace_memory_path = _save_round_resume_state(
                agent_dir,
                design,
                round_index,
                num_iterations,
                trace_memory,
                results,
                system.short_term_reflection,
                system.long_term_reflection
            )
            print(f"Saved trace memory backup after round {round_index} to {trace_memory_path}")
            print(f"Saved resume checkpoint after round {round_index} to {checkpoint_path}")

    with open(trace_memory_path, "wb") as f:
        pickle.dump(trace_memory.traces, f)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run ECO agent.")
    parser.add_argument("--design", default="NV_NVDLA_partition_m")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-path")
    args = parser.parse_args()
    simulate_eco_iterations(
        args.iterations,
        args.rounds,
        design=args.design,
        resume=args.resume,
        checkpoint_path=args.checkpoint_path
    )


if __name__ == "__main__":
    main()
