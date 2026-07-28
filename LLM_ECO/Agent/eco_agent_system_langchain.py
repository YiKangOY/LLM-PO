#!/usr/bin/env python3
"""
ECO Agent System - LangChain/LangGraph Implementation
Multi-agent system for Engineering Change Order optimization
Reimplemented using LangChain and LangGraph for simplicity
"""

import copy
import re
import time
import json
import logging
import os
import requests
from datetime import datetime
from typing import Dict, List, Any, TypedDict, Annotated
from enum import Enum
from dataclasses import dataclass, asdict

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
# LangGraph imports
from langgraph.graph import StateGraph, END
# from langgraph.graph.message import add_messages  # Not needed with simplified state
# from langgraph.checkpoint.memory import MemorySaver  # Not needed with manual conversation memory

# Import existing report parsers
from report_parsers import ReportParserManager, PowerMetrics, QoRMetrics, extract_json_from_response
from eco_database import ReportParser
from configs import BUFFER_LIST, base_path, get_agent_logs_dir

def extract_json_from_thinking_response(content: str) -> str:
    """Extract JSON content from LLM response that may contain <think>...</think> tags"""
    import re

    # Remove thinking tags and their content
    cleaned_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

    # Try to extract JSON from the cleaned content
    # Look for content within ```json...``` blocks first
    json_match = re.search(r'```json\s*(.*?)\s*```', cleaned_content, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()

    # If no code block, try to find JSON-like content
    # Look for content that starts with { and ends with }
    json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
    if json_match:
        return json_match.group(0).strip()

    # Return cleaned content if no specific JSON pattern found
    return cleaned_content.strip()

# Configuration


OPENAI_CONFIG = {
    "api_key": os.environ.get("OPENAI_API_KEY"),
    "api_url": os.environ.get("OPENAI_BASE_URL")
}


class ECOType(Enum):
    """ECO fix types"""
    DRC = "drc"
    TIMING = "timing"
    POWER = "power"



class FixMethod(Enum):
    """Available fixing actions"""
    SIZE_CELL = "gate_sizing"
    INSERT_BUFFER = "buffer_insertion"
    INSERT_BUFFER_AT_LOAD_PINS = "buffer_insertion_at_load_pins"
    INSERT_BUFFER_AT_DRIVER_PINS = "buffer_insertion_at_driver_pins"
    REMOVE_BUFFER = "buffer_removal"
    BYPASS_BUFFER = "buffer_bypass"


class PhysicalMode(Enum):
    """Physical implementation modes"""
    OPEN_SITE = "open_slot"
    OCCUPIED_SITE = "occupied_slot"


# Pydantic models for structured output
class CommandOutput(BaseModel):
    """Structured output for ECO commands"""
    tcl_command: str = Field(description="Complete TCL command to execute")
    timeout: int = Field(description="Recommended timeout in seconds")
    confidence: float = Field(description="Confidence level 0-1")
    reasoning: str = Field(description="Reasoning for this command")


class DecisionOutput(BaseModel):
    """Structured output for command selection"""
    selected_key: str = Field(description="Selected command key (drc, timing, or power)")
    reasoning: str = Field(description="Reasoning for selection")
    confidence: float = Field(description="Confidence level 0-1")


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


@dataclass
class LLMResponse:
    """Structure for logging LLM responses"""
    agent_type: str
    timestamp: datetime
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    processing_time: float
    iteration: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class LLMInteraction:
    """Structure for logging individual LLM message/response pairs"""
    agent_type: str
    timestamp: datetime
    iteration: int
    messages_sent: List[Dict[str, Any]]
    response_received: Dict[str, Any]
    processing_time: float
    model_name: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class ECOLogger:
    """Centralized logging system for ECO agent responses and LLM interactions"""

    def __init__(self, log_file: str = "eco_agent_responses.json"):
        self.log_file = log_file
        self.responses = []
        self.llm_interactions = []
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = get_agent_logs_dir()
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, os.path.basename(self.log_file))
        self.llm_log_file = os.path.join(log_dir, "llm_interactions.json")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'eco_system.log')),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger('ECOSystem')

    def log_response(self, response: LLMResponse):
        """Log an LLM response"""
        self.responses.append(response)
        self.logger.info(f"Agent {response.agent_type} - Iteration {response.iteration} - Time: {response.processing_time:.2f}s")
        self._save_to_json()

    def log_llm_interaction(self, interaction: LLMInteraction):
        """Log an LLM interaction (messages sent and response received)"""
        self.llm_interactions.append(interaction)
        self.logger.info(f"LLM Interaction - {interaction.agent_type} - Iteration {interaction.iteration} - Time: {interaction.processing_time:.2f}s - Model: {interaction.model_name}")
        self._save_llm_interactions()

    def _save_to_json(self):
        """Save all responses to JSON file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump([resp.to_dict() for resp in self.responses], f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save responses to JSON: {e}")

    def _save_llm_interactions(self):
        """Save all LLM interactions to JSON file"""
        try:
            with open(self.llm_log_file, 'w') as f:
                json.dump([interaction.to_dict() for interaction in self.llm_interactions], f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save LLM interactions to JSON: {e}")


class ECOLangChainSystem:
    """
    ECO System implemented with LangChain and LangGraph
    """

    def __init__(self, total_runtime_budget: float = 3600, log_file: str = "eco_agent_responses_langchain.json"):
        self.logger = ECOLogger(log_file)
        self.total_runtime_budget = total_runtime_budget
        self.start_time = time.time()
        self.iteration_count = -1

        # Initialize persistent state for accumulating context across iterations
        self.persistent_state: ECOState = None

        # Initialize separate LLM instances for each node
        self.llm_summary = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            api_key=OPENAI_CONFIG["api_key"],
            base_url=OPENAI_CONFIG["api_url"]
        )

        self.llm_drc = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            api_key=OPENAI_CONFIG["api_key"],
            base_url=OPENAI_CONFIG["api_url"]
        )

        self.llm_timing = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            api_key=OPENAI_CONFIG["api_key"],
            base_url=OPENAI_CONFIG["api_url"]
        )

        self.llm_power = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            api_key=OPENAI_CONFIG["api_key"],
            base_url=OPENAI_CONFIG["api_url"]
        )

        self.llm_selector = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            api_key=OPENAI_CONFIG["api_key"],
            base_url=OPENAI_CONFIG["api_url"]
        )

        # Initialize message lists for conversation memory for each LLM node
        self.summary_messages = []
        self.drc_messages = []
        self.timing_messages = []
        self.power_messages = []
        self.selector_messages = []

        # Initialize report parser
        self.parser_manager = ReportParserManager()

        # Create the LangGraph workflow
        self.workflow = self._create_workflow()

        self.logger.logger.info(f"ECO LangChain System initialized with {total_runtime_budget}s budget")


    def _summarize_reports(self, reports: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of reports for archival in iteration history"""
        summary = {}

        if 'timing' in reports:
            qor = reports['timing']
            summary['timing'] = {
                'setup_violations': qor.get('setup_violating_paths', 0),
                'hold_violations': qor.get('hold_violating_paths', 0),
                'setup_slack': qor.get('setup_critical_path_slack', 0),
                'hold_slack': qor.get('hold_critical_path_slack', 0),
            }
        if 'area' in reports:
            area = reports['area']
            summary['area'] = {
                'total_area': area.get('design_area', 0),
            }

        if 'drc' in reports:
            drc = reports['drc']
            summary['drc'] = {
                'min_capacitance_count': drc.get('min_capacitance_count', 0),
                'max_transition_count': drc.get('max_transition_count', 0),
                'total_drc_cost': drc.get('total_drc_cost', 0)
            }

        if 'power' in reports:
            power = reports['power']
            summary['power'] = {
                'total_power': power.get('total_power', 0),
                'dynamic_power': power.get('total_power', 0) - power.get('leakage_power', 0),
                'leakage_power': power.get('leakage_power', 0)
            }


        if 'last_fix' in reports:
            fix_type = reports['last_fix'].get('type', None)
            eco_data = reports['last_fix']
            eco_unfix_data = reports['last_unfix']
            if fix_type == 'drc':
                summary['last_fix'] = {
                    'type': 'drc',
                    'detected_violations': eco_data.get('detected_violations', 0),
                    'remaining_violations': eco_data.get('remaining_violations', 0),
                    'elapsed_time': eco_data.get('elapsed_time_seconds', 0)
                }
                summary['last_unfix'] = {"fix_type": "drc", "violations_by_reason": eco_unfix_data.get('violations_by_reason', {})}
            elif fix_type == 'timing':
                summary['last_fix'] = {
                    'type': 'timing',
                    'endpoints_found': eco_data.get('total_violating_endpoints_found', 0),
                    'endpoints_fixed': eco_data.get('total_violating_endpoints_fixed', 0),
                    'fix_percentage': eco_data.get('percentage_violations_fixed', 0)
                }
                summary['last_unfix'] = {"fix_type": "timing", "violations_summary": eco_unfix_data.get('summary', {})}
            elif fix_type == 'power':
                summary['last_fix'] = {
                    'type': 'power',
                    'area_decreased_percent': eco_data.get('percentage_area_decreased', 0),
                    'elapsed_time': eco_data.get('elapsed_time_seconds', 0)
                }
                summary['last_unfix'] = eco_unfix_data

        return summary

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(ECOState)

        # Add nodes
        workflow.add_node("analyze_reports", self._analyze_reports_node)

        workflow.add_node("generate_drc_command", self._generate_drc_command_node)
        workflow.add_node("generate_timing_command", self._generate_timing_command_node)
        workflow.add_node("generate_power_command", self._generate_power_command_node)
        workflow.add_node("select_command", self._select_command_node)
        workflow.add_node("execute_command", self._execute_command_node)

        # # Define the flow
        workflow.set_entry_point("analyze_reports")
        workflow.add_edge("analyze_reports", "generate_drc_command")
        workflow.add_edge("analyze_reports", "generate_timing_command")
        workflow.add_edge("analyze_reports", "generate_power_command")

        workflow.add_edge("generate_drc_command", "select_command")
        workflow.add_edge("generate_timing_command", "select_command")
        workflow.add_edge("generate_power_command", "select_command")

        workflow.add_edge("select_command", "execute_command")
        workflow.add_edge("execute_command", END)

        # Compile workflow
        return workflow.compile()

    def _get_optimization_type_from_tcl_command(self, tcl_command: str) -> str:
        """Extract optimization type from executed TCL command"""
        if not tcl_command:
            return 'unknown'

        tcl_command_lower = tcl_command.lower()
        if 'opt_drc' in tcl_command_lower:
            return 'drc'
        elif 'opt_timing' in tcl_command_lower:
            return 'timing'
        elif 'opt_power' in tcl_command_lower:
            return 'power'
        else:
            return 'unknown'

    def _analyze_reports_node(self, state: ECOState) -> ECOState:
        """Summary agent: Analyze parsed reports and execution history"""
        start_time = time.time()
        reports = state["reports"]
        iteration = state["iteration"]

        try:
            # Extract structured data from parsed reports
            current_state = {}
            parsing_errors = []


            # Handle structured QoR data
            if 'timing' in reports:
                qor_data = reports['timing']
                current_state['timing'] = {
                        'setup_violating_paths': qor_data.get('setup_violating_paths', 0),
                        'hold_violating_paths': qor_data.get('hold_violating_paths', 0),
                        'setup_critical_path_slack': round(qor_data.get('setup_critical_path_slack', 0.0), 4),
                        'hold_critical_path_slack': round(qor_data.get('hold_critical_path_slack', 0.0), 4),
                }
            if 'drc' in reports:
                drc_data = reports['drc']
                current_state['drc'] = {
                    'min_capacitance_count': drc_data.get('min_capacitance_count', 0),
                    'max_transition_count': drc_data.get('max_transition_count', 0)
                }
            if 'area' in reports:
                area_data = reports['area']
                current_state['area'] = {
                    'design_area': round(area_data.get('design_area', 0.0), 4)
                }
            # Handle structured power data
            if 'power' in reports:
                power_data = reports['power']
                current_state['power'] = {
                    'total_power': round(power_data.get('total_power', 0.0), 4),
                    'dynamic_power': round(power_data.get('total_power', 0.0) - power_data.get('leakage_power', 0.0), 4),
                    'leakage_power': round(power_data.get('leakage_power', 0.0), 4)
                }



            # Handle ECO fixing reports for non-zero iterations
            fixing_results = {}
            unfixable_reasons = {}

            if iteration > 0:
                # Get the actual optimization type from the previous iteration's executed command
                # Look for the TCL command in the execution result from previous iteration
                last_optimization_type = 'unknown'
                prev_analysis = state['unified_analysis']
                prev_iteration_key = f'iteration_{iteration-1}'
                if prev_iteration_key in prev_analysis['iteration_history']:
                    prev_execution_result = prev_analysis['iteration_history'][prev_iteration_key].get('execution_result', {})
                    prev_tcl_command = prev_execution_result.get('tcl_command', '')
                    last_optimization_type = self._get_optimization_type_from_tcl_command(prev_tcl_command)

                # Process fixing reports
                last_fix = reports.get('last_fix', {})
                fix_type = last_fix.get('fix_type', None)
                last_unfix = reports.get('last_unfix', {})
                if fix_type == 'timing':
                    fixing_results = {'last_optimization_command': prev_tcl_command, 'elapsed_time_seconds': last_fix.get('elapsed_time_seconds', 0), 'detected_violations': last_fix.get('total_violating_endpoints_found', 0), 'total_violating_endpoints_fixed' : last_fix.get('total_violating_endpoints_fixed', 0), 'percentage_violations_fixed' : last_fix.get('percentage_violations_fixed', 0)}

                    unfixable_reasons = {
                        "violations_summary": last_unfix.get('summary', {}),
                    }

                elif fix_type == 'drc':
                    fixing_results = {'last_optimization_command': prev_tcl_command, 'elapsed_time_seconds': last_fix.get('elapsed_time_seconds', 0), 'detected_violations': last_fix.get('detected_violations', 0), 'remaining_violations' : last_fix.get('remaining_violations', 0)}
                    unfixable_reasons = {'violations_by_reason': last_unfix.get('violations_by_reason', {}), "total_unfixable_violations": last_unfix.get('total_unfixable_violations', 0)}

                # Process power fixing reports
                elif fix_type == 'power':
                    fixing_results = {
                        'last_optimization_command': prev_tcl_command,
                        'elapsed_time_seconds': last_fix.get('elapsed_time_seconds', 0),
                        'total_power_decreased': round(last_fix.get('total_power_decreased', 0.0), 4),
                        'leakage_power_decreased': round(last_fix.get('leakage_power_decreased', 0.0), 4),
                        'dynamic_power_decreased': round(last_fix.get('dynamic_power_decreased', 0.0), 4),
                        'total_area_decreased': round(last_fix.get('total_area_decreased', 0.0), 4),
                    }
                    unfixable_reasons = last_unfix

            # Get execution history for non-zero iterations
            # execution_history = {}
            # if iteration > 0 and hasattr(state, 'unified_analysis') and state.get('unified_analysis'):
            #     prev_analysis = state['unified_analysis']
            #     if 'iteration_history' in prev_analysis:
            #         execution_history = prev_analysis['iteration_history']

            # LangGraph automatically maintains message history in state["messages"]
            # No manual extraction needed - we'll use it directly in the prompt

            # Prepare comprehensive summary prompt
            system_prompt = """You are an expert IC design summary agent working on summarizing reports. Your role is to:
1. Analyze parsed timing, drc, area, and power reports to analyze current design states
2. For non-zero iterations, evaluate the effectiveness of the last executed command
3. Provide insights on fixing/unfixable reasons if you are provided with these reports for non-zero iterations
4. Analyze previous dialogue histories to summarize overall optimization progress and trends across iterations

Before giving your answer, use <think>...</think> to think about your approach. This thinking won't be shown to the user.

Provide comprehensive analysis in JSON format with keys:
- current_design_state: Summary of current timing, drc violations and power consumptions.
- execution_effectiveness: Analysis of last command results (for iteration == 0, leave empty)
- unfixable_analysis: Summary of unfixable violations (for iteration ==0, leave empty)
- optimization_trends: Progress of timing, drc, power, and area ECO trends across iterations based on dialogue history

**Definitions of Unfix Reasons:**
    A - There are available library cells outside area limit
    B - Delay improvement is too small to fix the violation
    C - The violation is in clock network
    D - Cell or net is located in high density area
    E - Physical information is incomplete or unavailable
    H - Logical and physical hierarchies are inconsistent
    I - Buffer insertion with given library cells cannot fix the violation
    L - Available physical area limits the use of one or more library cells
    O - No open free site is available
    R - No locations are available in parasitics or location transformation failed
    S - Cell sizing with alternative library cells cannot fix the violation
    T - Timing margin is too tight to fix the violation
    W - Fixing the violation might degrade DRC violations

**Example**
Human: "**ITERATION 0 SUMMARY**\nCurrent Design Metrics:\n{{\n  \"timing\": {{\n    \"setup_violating_paths\": 48,\n    \"hold_violating_paths\": 56,\n    \"setup_critical_path_slack\": -0.1678,\n    \"hold_critical_path_slack\": -0.0072\n  }},\n  \"drc\": {{\n    \"min_capacitance_count\": 1,\n    \"max_transition_count\": 3371\n  }},\n  \"area\": {{\n    \"design_area\": 90725.5469,\n    \"cell_area\": 0.0\n  }},\n  \"power\": {{\n    \"total_power\": 0.077,\n    \"dynamic_power\": 0.07141,\n    \"leakage_power\": 0.00559\n  }}\n}}\n\nProvide comprehensive analysis in JSON format with keys:\n- current_design_state: Summary of current timing, drc violations and power consumptions.\n- execution_effectiveness: Analysis of last command results (for iteration > 0)\n- unfixable_analysis: Summary of unfixable violations (for iteration > 0)\n- optimization_trends: Progress of Timing, DRC, Power ECO trends across iterations based on dialogue history\n\nRemaining Runtime Budget: 3600s"

AI: "<think>
</think>

```json
{{
  \"current_design_state\": {{
    \"timing\": \"At iteration 0, design shows both setup (48 paths) and hold (56 paths) violations, with setup slack more critical at -0.1678ns\",
    \"drc\": \"Significant max_transition violations (3371) dominate over minimal min_capacitance violations (1)\",
    \"power\": \"Baseline power consumption established at 0.077mW total\"
  }},
  \"execution_effectiveness\": {{}},
  \"unfixable_analysis\": {{}},
  \"optimization_trends\": {{}}
}}
```"

Human: "**ITERATION 1 SUMMARY**', 'Current Design Metrics:', '{{\n  "timing": {{\n    "setup_violating_paths": 36,\n    "hold_violating_paths": 5,\n    "setup_critical_path_slack": -0.1621,\n    "hold_critical_path_slack": -0.0035\n  }},\n  "drc": {{\n    "min_capacitance_count": 1,\n    "max_transition_count": 3371\n  }},\n  "area": {{\n    "design_area": 75821.3672\n  }},\n  "power": {{\n    "total_power": 0.07,\n    "dynamic_power": 0.0659,\n    "leakage_power": 0.0041\n  }}\n}}', '\nProvide comprehensive analysis in JSON format with keys:\n- current_design_state: Summary of current timing, drc violations and power consumptions.\n- execution_effectiveness: Analysis of last command results (for iteration > 0)\n- unfixable_analysis: Summary of unfixable violations (for iteration > 0)\n- optimization_trends: Progress of Timing, DRC, Power ECO trends across iterations based on dialogue history', '\nLast Executed Commands Results:', '{{\n  'last_optimization_command': 'opt_power -actions gate_sizing -power_scope total -timeout 200',\n  "elapsed_time_seconds": 125,\n  "total_power_decreased": 0.007,\n  "leakage_power_decreased": 0.0015,\n  "dynamic_power_decreased": 0.0055,\n  "total_area_decreased": 14903.91\n}}', '\nUnfixable Violations Analysis:', '{{\n  "fix_type": "power",\n  "fixes": "Power fixing does not have unfix report because power is not a constraint."\n}}', '\nRemaining Runtime Budget: 3422s"

AI: "<think>
</think>

```json
{{
  \"current_design_state\": {{
    \"timing\": \"At iteration 1, timing shows improvement - setup violations reduced from 48 to 36 paths, hold violations significantly improved from 56 to 5 paths\",
    \"drc\": \"DRC violations remain unchanged with same max_transition (3371) and min_capacitance (1) counts\",
    \"power\": \"Power successfully optimized from 0.077mW to 0.07mW, with significant area reduction\"
  }},
  \"execution_effectiveness\": {{
    \"command_type\": \"power optimization\",
    \"power_reduction\": \"7mW reduction achieved (9.1% improvement)\",
    \"area_improvement\": \"Significant area reduction of 14903.91 units\",
    \"runtime_efficiency\": \"Completed in 125s, good time efficiency\"
  }},
  \"unfixable_analysis\": {{
    \"power_constraints\": \"Power optimization has no unfixable violations as it's not a hard constraint\"
  }},
  \"optimization_trends\": {{
    \"power_trend\": \"Power optimization showing good results, further reduction possible\",
    \"timing_trend\": \"Timing improved as side effect of power optimization\",
    \"drc_trend\": \"DRC violations not addressed yet, require dedicated optimization\",
    \"strategy\": \"Continue with targeted DRC or timing optimization in next iteration\"
  }}
}}
```"

"""

            # Prepare JSON strings (escape braces for f-string compatibility)
            current_state_json = json.dumps(current_state, indent=2).replace("{", "{{").replace("}", "}}")
            fixing_results_json = json.dumps(fixing_results, indent=2).replace("{", "{{").replace("}", "}}")
            unfixable_reasons_json = json.dumps(unfixable_reasons, indent=2).replace("{", "{{").replace("}", "}}")
            # execution_history_json = json.dumps(execution_history, indent=2).replace("{", "{{").replace("}", "}}")

            human_prompt_parts = [
                f"**ITERATION {iteration} SUMMARY**",
                f"Current Design Metrics:",
                current_state_json,
                """\nProvide comprehensive analysis in JSON format with keys:
- current_design_state: Summary of current timing, drc violations and power consumptions.
- execution_effectiveness: Analysis of last command results (for iteration > 0)
- unfixable_analysis: Summary of unfixable violations (for iteration > 0)
- optimization_trends: Progress of Timing, DRC, Power ECO trends across iterations based on dialogue history"""
            ]

            if iteration > 0:
                human_prompt_parts.extend([
                    f"\nLast Executed Commands Results:",
                    fixing_results_json,
                    f"\nUnfixable Violations Analysis:",
                    unfixable_reasons_json,
                    # f"\nExecution History:",
                    # execution_history_json
                ])

            human_prompt_parts.append(f"\nRemaining Runtime Budget: {state['remaining_budget']:.0f}s")

            # Use system prompt only in first iteration, LangGraph memory handles subsequent ones
            if state["iteration"] == 0:
                # First iteration: establish system context
                analysis_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "\n".join(human_prompt_parts))
                ])
            else:
                # Subsequent iterations: only human input, LangGraph handles history
                analysis_prompt = ChatPromptTemplate.from_messages([
                    ("human", "\n".join(human_prompt_parts))
                ])

            # Invoke LLM directly with logging using conversation memory
            llm_start_time = time.time()
            try:
                # Get formatted messages from prompt
                new_messages = analysis_prompt.format_messages()

                # Append new messages to conversation history
                for msg in new_messages:
                    self.summary_messages.append(msg)

                messages_sent = [{"role": msg.type, "content": msg.content} for msg in self.summary_messages]

                # Invoke LLM with full conversation history
                response = self.llm_summary.invoke(self.summary_messages)

                # Append LLM response to conversation history
                from langchain_core.messages import AIMessage
                self.summary_messages.append(AIMessage(content=response.content))

                # Parse JSON response, skipping thinking content
                json_content = extract_json_from_thinking_response(response.content)
                parser = JsonOutputParser()
                llm_analysis = parser.parse(json_content)

                processing_time = time.time() - llm_start_time

                # Log the interaction
                interaction = LLMInteraction(
                    agent_type="SummaryAgent",
                    timestamp=datetime.now(),
                    iteration=iteration,
                    messages_sent=messages_sent,
                    response_received={"content": response.content, "type": "json"},
                    processing_time=processing_time,
                    model_name=self.llm_summary.model_name
                )
                self.logger.log_llm_interaction(interaction)

            except Exception as e:
                processing_time = time.time() - llm_start_time
                error_response = {"error": str(e), "type": "error"}

                # Log the failed interaction
                interaction = LLMInteraction(
                    agent_type="SummaryAgent",
                    timestamp=datetime.now(),
                    iteration=iteration,
                    messages_sent=messages_sent if 'messages_sent' in locals() else [{"role": "user", "content": "Error extracting messages"}],
                    response_received=error_response,
                    processing_time=processing_time,
                    model_name=self.llm_summary.model_name
                )
                self.logger.log_llm_interaction(interaction)
                raise e

            unified_analysis = {
                'iteration': iteration,
                'current_state': current_state,
                'fixing_results': fixing_results,
                'unfixable_reasons': unfixable_reasons,
                'llm_summary': llm_analysis,
                'parsing_errors': parsing_errors,
                'processing_time': time.time() - start_time
            }

            # Preserve iteration history from previous state
            if hasattr(state, 'unified_analysis') and state.get('unified_analysis') and 'iteration_history' in state['unified_analysis']:
                unified_analysis['iteration_history'] = state['unified_analysis']['iteration_history']

            # Log response
            response = LLMResponse(
                agent_type="SummaryAgent",
                timestamp=datetime.now(),
                input_data={
                    'reports_processed': list(reports.keys()),
                    'iteration': iteration,
                    'has_dialogue_history': len(state.get('messages', [])) > 0
                },
                output_data=unified_analysis,
                processing_time=time.time() - start_time,
                iteration=iteration
            )
            self.logger.log_response(response)

            state["unified_analysis"] = unified_analysis

            # Add the LLM analysis response to message history for future reference
            # This creates the conversation flow: human prompt -> assistant response
            from langchain_core.messages import AIMessage

            # Escape braces in JSON to avoid template parsing issues
            raw_json = json.dumps(llm_analysis, indent=2)
            escaped_json = raw_json.replace("{", "{{").replace("}", "}}")

            # Store as AIMessage (will map to "assistant" when processed by prompt template)
            state["messages"].append(AIMessage(content=escaped_json))

        except Exception as e:
            raise RuntimeError(f"Failed to analyze reports: {e}")


        return state

    def _generate_drc_command_node(self, state: ECOState) -> ECOState:
        """Generate DRC command using LangChain"""
        start_time = time.time()

        try:
            # Get comprehensive analysis from summary agent
            unified_analysis = state["unified_analysis"]
            current_state = unified_analysis.get('current_state', {})
            llm_summary = unified_analysis.get('llm_summary', {})
            fixing_results = unified_analysis.get('fixing_results', {})
            unfixable_reasons = unified_analysis.get('unfixable_reasons', {})


            current_state['drc']
            drc_violations = (current_state['drc'].get('min_capacitance_count', 0) +
                            current_state['drc'].get('max_transition_count', 0))

            if drc_violations == 0:
                command_result = {
                    'command_type': ECOType.DRC.value,
                    'tcl_command': '',
                    'reasoning': 'No DRC violations detected',
                    'recommended_timeout': 0,
                }
            else:
                # Build prompt messages with conversation history
                prompt_messages = [
                    ("system", """You are an expert in IC design DRC fixing. Use the comprehensive summary analysis to generate optimal opt_drc commands that consider:
                    1. Current design DRC violation status
                    2. Previous DRC fixing effectiveness and trends if any
                    3. Previous DRC unfixable violation reasons to avoid ineffective approaches
                    4. Optimization trend analysis
                    5. Runtime budget

                    Before giving your answer, use <think>...</think> to think about your approach. This thinking won't be shown to the user.

                    Command format: opt_drc -type [max_transition | max_capacitance | max_fanout] -actions {{method1 method2 ...}} -site_mode [mode] -timeout [seconds] -Vth {{HVT LVT RVT}}
                    Available actions: gate_sizing, buffer_insertion, buffer_insertion_at_load_pins, buffer_insertion_at_driver_pins, buffer_removal, buffer_bypass
                    Available physical modes: open_slot, occupied_slot
                    Avaialble Vths: [LVT HVT RVT] you can choose any one or more from them.
                    Note: buffer_removal cannot be used with other actions and the timeout option.

                    **Strategy guidelines:**
                    - Refer to previous fixing results to avoid actions that proved ineffective
                    - Consider unfixable reasons to choose appropriate actions and modes
                    - Balance the timeout option vs quality based on remaining budget and trends:
                        -- The more actions you choose, the more optimization opportunities there will be but slower the command will be
                        -- At earlier optimization iterations, spend less efforts to fix easily-fixable issues to leave enough runtime budget for further optimization. At the middle of optimization iterations, try larger efforts to fix hard-to-fix issues. When the runtime budget is about to use up, try less efforts to fix remaining issues on time.
                        -- Physical mode "open_slot": faster, uses free layout areas
                        -- Physical mode "occupied_slot": slower but more powerful, modifies existing layout, breaks physical constraints
                        -- Vth stands for threshold voltage mode. Use more VTHs may lead to better fixing results but cost more runtime. LVT cells help reduce transition time, while HVT cells have lower capacitance, and RVT cells provide balance between transition and capacitance.

                    **Definitions of Unfix Reasons:**
                        A - There are available library cells outside area limit
                        B - Delay improvement is too small to fix the violation
                        C - The violation is in clock network
                        D - Cell or net is located in high density area
                        E - Physical information is incomplete or unavailable
                        H - Logical and physical hierarchies are inconsistent
                        I - Buffer insertion with given library cells cannot fix the violation
                        L - Available physical area limits the use of one or more library cells
                        O - No open free site is available
                        R - No locations are available in parasitics or location transformation failed
                        S - Cell sizing with alternative library cells cannot fix the violation
                        T - Timing margin is too tight to fix the violation
                        W - Fixing the violation might degrade DRC violations
                     """)
                ]

                # Current iteration input - LangGraph automatically maintains conversation history
                user_input = f"""
CURRENT DRC VIOLATIONS: {drc_violations} violations
REMAINING BUDGET: {state["remaining_budget"]:.0f}s
ITERATION: {state["iteration"]}

SUMMARY ANALYSIS:
{json.dumps(llm_summary, indent=2).replace('{', '{{').replace('}', '}}')}

PREVIOUS FIXING RESULTS:
{json.dumps(fixing_results, indent=2).replace('{', '{{').replace('}', '}}')}

UNFIXABLE REASONS:
{json.dumps(unfixable_reasons, indent=2).replace('{', '{{').replace('}', '}}')}

Generate an optimal opt_drc command considering the above analysis.
Command format: opt_drc -type [max_transition | max_capacitance | max_fanout] -actions {{{{method1 method2 ...}}}} -site_mode [mode] -timeout [seconds]
Available actions: gate_sizing, buffer_insertion, buffer_insertion_at_load_pins, buffer_insertion_at_driver_pins, buffer_removal, buffer_bypass
Available physical modes: open_slot, occupied_slot
Note: buffer_removal cannot be used with other actionsm, nor the -timeout parameter.

If you think no more optimization is needed, return and empty command string and reason.
Respond with JSON format:
{{{{
    "command": "complete opt_drc command string",
    "reason": "detailed reasoning for this command choice"
}}}}
"""
                # Use system prompt only in first iteration, LangGraph memory handles subsequent ones
                if state["iteration"] == 0:
                    # First iteration: establish system context
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", prompt_messages[0][1]),
                        ("human", user_input)
                    ])
                else:
                    # Subsequent iterations: only human input, LangGraph handles history
                    prompt = ChatPromptTemplate.from_messages([
                        ("human", user_input)
                    ])

                # Invoke LLM directly with logging using conversation memory
                llm_start_time = time.time()
                try:
                    # Get formatted messages from prompt
                    new_messages = prompt.format_messages()

                    # Append new messages to conversation history
                    for msg in new_messages:
                        self.drc_messages.append(msg)

                    messages_sent = [{"role": msg.type, "content": msg.content} for msg in self.drc_messages]

                    # Invoke LLM with full conversation history
                    response = self.llm_drc.invoke(self.drc_messages)

                    # Append LLM response to conversation history
                    from langchain_core.messages import AIMessage
                    self.drc_messages.append(AIMessage(content=response.content))

                    # Parse JSON response, skipping thinking content
                    json_content = extract_json_from_thinking_response(response.content)
                    parser = JsonOutputParser()
                    result = parser.parse(json_content)

                    processing_time = time.time() - llm_start_time

                    # Log the interaction
                    interaction = LLMInteraction(
                        agent_type="DRCCommandGenerator",
                        timestamp=datetime.now(),
                        iteration=state["iteration"],
                        messages_sent=messages_sent,
                        response_received={"content": response.content, "type": "json"},
                        processing_time=processing_time,
                        model_name=self.llm_drc.model_name
                    )
                    self.logger.log_llm_interaction(interaction)

                except Exception as e:
                    processing_time = time.time() - llm_start_time
                    error_response = {"error": str(e), "type": "error"}

                    # Log the failed interaction
                    interaction = LLMInteraction(
                        agent_type="DRCCommandGenerator",
                        timestamp=datetime.now(),
                        iteration=state["iteration"],
                        messages_sent=messages_sent if 'messages_sent' in locals() else [{"role": "user", "content": "Error extracting messages"}],
                        response_received=error_response,
                        processing_time=processing_time,
                        model_name=self.llm_drc.model_name
                    )
                    self.logger.log_llm_interaction(interaction)
                    raise e

                command_result = {
                    'command_type': ECOType.DRC.value,
                    'tcl_command': result.get('command', ''),
                    'reasoning': result.get('reason', '')
                }


            # Log response
            response = LLMResponse(
                agent_type="DRCCommandGenerator",
                timestamp=datetime.now(),
                input_data={'drc_violations': drc_violations},
                output_data=command_result,
                processing_time=time.time() - start_time,
                iteration=state["iteration"]
            )
            self.logger.log_response(response)

            # Return command proposal as partial update for parallel processing
            return {"command_proposals": {"drc": command_result}}

        except Exception as e:
            self.logger.logger.error(f"DRC command generation failed: {e}")
            error_command = {
                'command_type': ECOType.DRC.value,
                'tcl_command': '',
                'reasoning': f'Command generation failed: {e}'
            }
            return {"command_proposals": {"drc": error_command}}

    def _generate_timing_command_node(self, state: ECOState) -> ECOState:
        """Generate timing command using LangChain"""
        start_time = time.time()

        try:
            # Get comprehensive analysis from summary agent
            unified_analysis = state["unified_analysis"]
            current_state = unified_analysis.get('current_state', {})
            llm_summary = unified_analysis.get('llm_summary', {})
            fixing_results = unified_analysis.get('fixing_results', {})
            unfixable_reasons = unified_analysis.get('unfixable_reasons', {})
            execution_history = unified_analysis.get('execution_history', {})


            setup_violations = current_state['timing'].get('setup_violating_paths', 0)
            hold_violations = current_state['timing'].get('hold_violating_paths', 0)
            total_violations = setup_violations + hold_violations

            if total_violations == 0:
                command_result = {
                    'command_type': ECOType.TIMING.value,
                    'tcl_command': '',
                    'reasoning': 'No timing violations detected',
                    'recommended_timeout': 0,
                }
            else:
                # Build prompt messages with conversation history
                prompt_messages = [
                    ("system", """You are an expert in IC design timing fixing. Use the comprehensive summary analysis to generate optimal opt_timing commands that consider:
                    1. Current design timing violation status
                    2. Previous timing fixing effectiveness and trends if any
                    3. Profile of unfixable timing paths and their reasons
                    4. Optimization trend analysis
                    5. Runtime budget

                    Before giving your answer, use <think>...</think> to think about your approach. This thinking won't be shown to the user.

                    Command format: opt_timing -type [setup|hold] -actions {{method1 method2 ...}} -site_mode [mode] -timeout [seconds] -Vth {{HVT LVT RVT}} [-area_cap x]
                    Available actions for setup: gate_sizing, gate_sizing_side_load, buffer_insertion
                    Available actions for hold: gate_sizing, buffer_insertion
                    Available physical modes: open_slot, occupied_slot
                    Avaialble Vths: [LVT HVT RVT] you can choose any one or more from them.
                    Optional -area_cap: Specify area increment limit for gate_sizing, only use it when gate_sizing option selected, defalut value is 2.
                    Optional -slack_above slack_limit: By setting the slack limit to a positive value, the command will try to improve the timing of paths with a positive slack_limit (means no timing violation).

                    **Strategy guidelines:**
                    - Prioritize setup over hold violation
                    - Use previous timing fix results to select effective actions
                    - Balance the timeout option vs quality based on remaining budget and trends:
                        -- The more **actions** you choose, the more optimization opportunities there will be but slower the command will be
                        -- At earlier optimization iterations, spend less efforts to fix easily-fixable issues to leave enough runtime budget for further optimization. At the middle of optimization iterations, try larger efforts to fix hard-to-fix issues. When the runtime budget is about to use up, try less efforts to fix remaining issues on time.
                        -- Physical mode "open_slot": faster, uses free layout areas
                        -- Physical mode "occupied_slot": slower but more powerful, modifies existing layout, breaks physical constraints, could be used when timing paths are unfixable due to "L - Available physical area limits the use of one or more library cells"
                        -- Vth stands for threshold voltage mode, which affects cell performance and power consumption: Use more VTHs may lead to better fixing results but cost more runtime. LVT is good for timing fixing but increase power consumption. HVT reduces power consumption but has little timing fixing effect. RVT is a trade-off between LVT and HVT.
                    - For extremely unfixable timing constraints, consider append "-ignore_drc" into the command to allow optimizing timing but sacrifice DRC integrity.
                    - If the timing paths are unfixable due to "S - Cell sizing with alternative library cells cannot fix the violation", try append "-area_cap x" to the command, the default value for x is 2. But upsclaing this limit sacrifies power and area, use it only when timings cannoted be fixed mainly due to area limit of available cells.
                    - Only set the *-slack_above slack_limit* at the later stage of optimization and no other metrics, like power, DRC, can be optimized. Set the slack_limit starting from small values, otherwise it will take much runtime.

                    **Definitions of Unfix Reasons:**
                        A - There are available library cells outside area limit
                        B - Delay improvement is too small to fix the violation
                        C - The violation is in clock network
                        D - Cell or net is located in high density area
                        E - Physical information is incomplete or unavailable
                        H - Logical and physical hierarchies are inconsistent
                        I - Buffer insertion with given library cells cannot fix the violation
                        L - Available physical area limits the use of one or more library cells
                        O - No open free site is available
                        R - No locations are available in parasitics or location transformation failed
                        S - Cell sizing with alternative library cells cannot fix the violation
                        T - Timing margin is too tight to fix the violation
                        W - Fixing the violation might degrade DRC violations
                    """)
                ]

                # Current iteration input
                user_input = f"""
CURRENT TIMING VIOLATIONS:
- Setup violations: {setup_violations}
- Hold violations: {hold_violations}
- Total violations: {total_violations}

REMAINING BUDGET: {state["remaining_budget"]:.0f}s
ITERATION: {state["iteration"]}

SUMMARY ANALYSIS:
{json.dumps(llm_summary, indent=2).replace('{', '{{').replace('}', '}}')}

PREVIOUS FIXING RESULTS:
{json.dumps(fixing_results, indent=2).replace('{', '{{').replace('}', '}}')}

UNFIXABLE REASONS:
{json.dumps(unfixable_reasons, indent=2).replace('{', '{{').replace('}', '}}')}

Generate an optimal opt_timing command considering the above analysis.
Command format: opt_timing -type [setup|hold] -actions {{{{method1 method2 ...}}}} -site_mode [mode] -timeout [seconds]
Available actions for setup: gate_sizing, gate_sizing_side_load, buffer_insertion
Available actions for hold: gate_sizing, buffer_insertion
Available physical modes: open_slot, occupied_slot

Respond with JSON format:
{{{{
    "command": "complete opt_timing command string",
    "reason": "detailed reasoning for this command choice"
}}}}
"""
                # Use system prompt only in first iteration, LangGraph memory handles subsequent ones
                if state["iteration"] == 0:
                    # First iteration: establish system context
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", prompt_messages[0][1]),
                        ("human", user_input)
                    ])
                else:
                    # Subsequent iterations: only human input, LangGraph handles history
                    prompt = ChatPromptTemplate.from_messages([
                        ("human", user_input)
                    ])

                # Invoke LLM directly with logging using conversation memory
                llm_start_time = time.time()
                try:
                    # Get formatted messages from prompt
                    new_messages = prompt.format_messages()

                    # Append new messages to conversation history
                    for msg in new_messages:
                        self.timing_messages.append(msg)

                    messages_sent = [{"role": msg.type, "content": msg.content} for msg in self.timing_messages]

                    # Invoke LLM with full conversation history
                    response = self.llm_timing.invoke(self.timing_messages)

                    # Append LLM response to conversation history
                    from langchain_core.messages import AIMessage
                    self.timing_messages.append(AIMessage(content=response.content))

                    # Parse JSON response, skipping thinking content
                    json_content = extract_json_from_thinking_response(response.content)
                    parser = JsonOutputParser()
                    result = parser.parse(json_content)

                    processing_time = time.time() - llm_start_time

                    # Log the interaction
                    interaction = LLMInteraction(
                        agent_type="TimingCommandGenerator",
                        timestamp=datetime.now(),
                        iteration=state["iteration"],
                        messages_sent=messages_sent,
                        response_received={"content": response.content, "type": "json"},
                        processing_time=processing_time,
                        model_name=self.llm_timing.model_name
                    )
                    self.logger.log_llm_interaction(interaction)

                except Exception as e:
                    processing_time = time.time() - llm_start_time
                    error_response = {"error": str(e), "type": "error"}

                    # Log the failed interaction
                    interaction = LLMInteraction(
                        agent_type="TimingCommandGenerator",
                        timestamp=datetime.now(),
                        iteration=state["iteration"],
                        messages_sent=messages_sent if 'messages_sent' in locals() else [{"role": "user", "content": "Error extracting messages"}],
                        response_received=error_response,
                        processing_time=processing_time,
                        model_name=self.llm_timing.model_name
                    )
                    self.logger.log_llm_interaction(interaction)
                    raise e

                command_result = {
                    'command_type': ECOType.TIMING.value,
                    'tcl_command': result.get('command', ''),
                    'reasoning': result.get('reason', '')
                }


            # Log response
            response = LLMResponse(
                agent_type="TimingCommandGenerator",
                timestamp=datetime.now(),
                input_data={'setup_violations': setup_violations, 'hold_violations': hold_violations},
                output_data=command_result,
                processing_time=time.time() - start_time,
                iteration=state["iteration"]
            )
            self.logger.log_response(response)

            # Return command proposal as partial update for parallel processing
            return {"command_proposals": {"timing": command_result}}

        except Exception as e:
            self.logger.logger.error(f"Timing command generation failed: {e}")
            error_command = {
                'command_type': ECOType.TIMING.value,
                'tcl_command': '',
                'reasoning': f'Command generation failed: {e}'
            }
            return {"command_proposals": {"timing": error_command}}

    def _generate_power_command_node(self, state: ECOState) -> ECOState:
        """Generate power command using LangChain"""
        start_time = time.time()

        try:
            # Get comprehensive analysis from summary agent
            unified_analysis = state["unified_analysis"]
            current_state = unified_analysis.get('current_state', {})
            llm_summary = unified_analysis.get('llm_summary', {})
            fixing_results = unified_analysis.get('fixing_results', {})
            unfixable_reasons = unified_analysis.get('unfixable_reasons', {})
            execution_history = unified_analysis.get('execution_history', {})

            power_metrics = current_state.get('power', {}).get('metrics', {})

            total_power = power_metrics.get('total_power', 0.0)
            clock_power = power_metrics.get('clock_tree_power', 0.0)
            register_power = power_metrics.get('register_power', 0.0)
            combinational_power = power_metrics.get('combinational_power', 0.0)
            leakage_power = power_metrics.get('leakage_power', 0.0)

            # Build prompt messages with conversation history
            prompt_messages = [
                ("system", """You are an expert in IC design power optimization. Use the comprehensive summary analysis to generate optimal opt_power commands that consider:
                 1. Current design power consumption breakdown
                 2. Previous power optimization effectiveness and trends if any
                 3. Optimization trend analysis
                 4. Runtime budget
                 Note: The reduction of power will usually lead to the reduction of area.

                 Before giving your answer, use <think>...</think> to think about your approach. This thinking won't be shown to the user.

                Command format: opt_power -actions [method] -power_scope [mode] -timeout [seconds] -Vth {{HVT LVT RVT}}
                Available actions: gate_sizing, buffer_removal
                Available power_scope: total | dynamic | leakage
                Avaialble threshold voltage modes: [LVT HVT RVT] you can choose any one or more from them.
                Note: buffer_removal cannot be used with other actions and the timeout option.

                Strategy guidelines:
                - Use power breakdown to target specific power components based on analysis and trends
                - Consider previous power fix results to select effective actions and modes
                - Select proper runtime budget based on remaining time and optimization trends:
                    -- At earlier optimization iterations, spend less time to fix easily-fixable issues to leave enough runtime budget for further optimization. At the middle of optimization iterations, try larger timeout to fix hard-to-fix issues. When the runtime budget is about to use up, try less runtime to fix remaining issues on time.
                    -- Vth stands for threshold voltage mode, which affects cell performance and power consumption: Use more VTHs may lead to better fixing results but cost more runtime. LVT is good for timing fixing but increase power consumption. HVT reduces power consumption but has little timing fixing effect. RVT is a trade-off between LVT and HVT.
                 """)
            ]

            # Current iteration input - LangGraph automatically maintains conversation history
            user_input = f"""
CURRENT POWER BREAKDOWN:
- Total power: {total_power:.2f}mW
- Clock network power: {clock_power:.2f}mW
- Register power: {register_power:.2f}mW
- Combinational power: {combinational_power:.2f}mW
- Leakage power: {leakage_power:.2f}mW

REMAINING BUDGET: {state["remaining_budget"]:.0f}s
ITERATION: {state["iteration"]}

SUMMARY ANALYSIS:
{json.dumps(llm_summary, indent=2).replace('{', '{{').replace('}', '}}')}

PREVIOUS FIXING RESULTS:
{json.dumps(fixing_results, indent=2).replace('{', '{{').replace('}', '}}')}

UNFIXABLE REASONS:
{json.dumps(unfixable_reasons, indent=2).replace('{', '{{').replace('}', '}}')}

Generate an optimal opt_power command considering the above analysis.
Command format: opt_power -actions [method] -power_scope [mode] -timeout [seconds]
Available actions: gate_sizing, buffer_removal
Available power_scope: total | dynamic | leakage
Note: buffer_removal cannot be used with other actions and the timeout option.

Respond with JSON format:
{{{{
    "command": "complete opt_power command string",
    "reason": "detailed reasoning for this command choice"
}}}}
"""
            # Use system prompt only in first iteration, LangGraph memory handles subsequent ones
            if state["iteration"] == 0:
                # First iteration: establish system context
                prompt = ChatPromptTemplate.from_messages([
                    ("system", prompt_messages[0][1]),
                    ("human", user_input)
                ])
            else:
                # Subsequent iterations: only human input, LangGraph handles history
                prompt = ChatPromptTemplate.from_messages([
                    ("human", user_input)
                ])

            # Invoke LLM directly with logging using conversation memory
            llm_start_time = time.time()
            try:
                # Get formatted messages from prompt
                new_messages = prompt.format_messages()

                # Append new messages to conversation history
                for msg in new_messages:
                    self.power_messages.append(msg)

                messages_sent = [{"role": msg.type, "content": msg.content} for msg in self.power_messages]

                # Invoke LLM with full conversation history
                response = self.llm_power.invoke(self.power_messages)

                # Append LLM response to conversation history
                from langchain_core.messages import AIMessage
                self.power_messages.append(AIMessage(content=response.content))

                # Parse JSON response, skipping thinking content
                json_content = extract_json_from_thinking_response(response.content)
                parser = JsonOutputParser()
                result = parser.parse(json_content)

                processing_time = time.time() - llm_start_time

                # Log the interaction
                interaction = LLMInteraction(
                    agent_type="PowerCommandGenerator",
                    timestamp=datetime.now(),
                    iteration=state["iteration"],
                    messages_sent=messages_sent,
                    response_received={"content": response.content, "type": "json"},
                    processing_time=processing_time,
                    model_name=self.llm_power.model_name
                )
                self.logger.log_llm_interaction(interaction)

            except Exception as e:
                processing_time = time.time() - llm_start_time
                error_response = {"error": str(e), "type": "error"}

                # Log the failed interaction
                interaction = LLMInteraction(
                    agent_type="PowerCommandGenerator",
                    timestamp=datetime.now(),
                    iteration=state["iteration"],
                    messages_sent=messages_sent if 'messages_sent' in locals() else [{"role": "user", "content": "Error extracting messages"}],
                    response_received=error_response,
                    processing_time=processing_time,
                    model_name=self.llm_power.model_name
                )
                self.logger.log_llm_interaction(interaction)
                raise e

            command_result = {
                'command_type': ECOType.POWER.value,
                'tcl_command': result.get('command', ''),
                'reasoning': result.get('reason', '')
            }


            # Log response
            response = LLMResponse(
                agent_type="PowerCommandGenerator",
                timestamp=datetime.now(),
                input_data={'total_power': total_power},
                output_data=command_result,
                processing_time=time.time() - start_time,
                iteration=state["iteration"]
            )
            self.logger.log_response(response)

            # Return command proposal as partial update for parallel processing
            return {"command_proposals": {"power": command_result}}

        except Exception as e:
            self.logger.logger.error(f"Power command generation failed: {e}")
            error_command = {
                'command_type': ECOType.POWER.value,
                'tcl_command': '',
                'reasoning': f'Command generation failed: {e}'
            }
            return {"command_proposals": {"power": error_command}}

    def _select_command_node(self, state: ECOState) -> ECOState:
        """Select optimal command using comprehensive analysis and optimization trends"""
        start_time = time.time()

        try:
            # Get comprehensive analysis from summary agent
            unified_analysis = state["unified_analysis"]
            current_state = unified_analysis.get('current_state', {})
            llm_summary = unified_analysis.get('llm_summary', {})
            fixing_results = unified_analysis.get('fixing_results', {})
            unfixable_reasons = unified_analysis.get('unfixable_reasons', {})
            execution_history = unified_analysis.get('execution_history', {})
            command_proposals = state["command_proposals"]

            # Use system prompt only in first iteration, LangGraph memory handles subsequent ones
            human_input = f"""
                Analyze the provided optimization process profile and runtime budget, and the proposed DRC, Timing, and Power optimization commands with their reasoning.
                Select the command key ('drc', 'timing', or 'power') that should be executed next.

                ITERATION: {state["iteration"]}
                REMAINING BUDGET: {state["remaining_budget"]:.0f}s

                CURRENT DESIGN STATE:
                {json.dumps(current_state, indent=2).replace('{', '{{').replace('}', '}}')}

                OPTIMIZATION TRENDS & SUMMARY ANALYSIS:
                {json.dumps(llm_summary, indent=2).replace('{', '{{').replace('}', '}}')}

                PREVIOUS FIXING RESULTS:
                {json.dumps(fixing_results, indent=2).replace('{', '{{').replace('}', '}}')}

                UNFIXABLE REASONS:
                {json.dumps(unfixable_reasons, indent=2).replace('{', '{{').replace('}', '}}')}

                COMMAND PROPOSALS WITH REASONING:
                {json.dumps(command_proposals, indent=2).replace('{', '{{').replace('}', '}}')}

                Respond with JSON format:
                {{{{
                    "selected_key": "drc|timing|power",
                    "reasoning": "detailed reasoning for this selection"
                }}}}
                """

            if state["iteration"] == 0:
                # First iteration: establish system context
                #                     5. Strategic ECO flow priorities (Power → DRC → Timing).
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are an expert IC design optimization strategist. You will be given available ECO options for fixing DRC, Timing, and Power.
                    For DRC and Timing, the less violation is preferred, while for Power, the less power consumption is preferred. The optimization of power usually also optimizes area.

                    Before giving your answer, use <think>...</think> to think about your approach. This thinking won't be shown to the user.

                    At each iteration, you will be given:
                    1. Current design Timing, DRC, Power summaries.
                    2. Optimization trends and summary analysis from the summary agent.
                    3. The previous iteration's fixing results (if not the first iteration).
                    4. The DRC, Timing, Power ECO commands proposed by engineers.

                    You need to choose one of the proposed DRC, Timing, and Power commands for executing, while considering:
                    1. Current design state and violation severity.
                    2. Optimization trends and convergence patterns from summary analysis.
                    3. Previous fixing effectiveness and unfixable reasons to avoid wasting time on impossible fixes.
                    4. Consider remaining runtime budget and used runtime to explore new fixing possibilities or keep optimizing current target with larger efforts.
                    5. The coherence between the reasons of [DRC|Timing|Power] command proposal and corresponding profiles related to [DRC|Timing|Power].
                    6. Allow timing fixing proposal with option *slack_above* even if there is no timing violation to keep increase timing.
                    """),
                    ("human", human_input)
                ])
            else:
                # Subsequent iterations: only human input, LangGraph handles history
                prompt = ChatPromptTemplate.from_messages([
                    ("human", human_input)
                ])

            # Invoke LLM directly with logging using conversation memory
            llm_start_time = time.time()
            try:
                # Get formatted messages from prompt
                new_messages = prompt.format_messages()

                # Append new messages to conversation history
                for msg in new_messages:
                    self.selector_messages.append(msg)

                messages_sent = [{"role": msg.type, "content": msg.content} for msg in self.selector_messages]

                # Invoke LLM with full conversation history
                response = self.llm_selector.invoke(self.selector_messages)

                # Append LLM response to conversation history
                from langchain_core.messages import AIMessage
                self.selector_messages.append(AIMessage(content=response.content))

                # Parse JSON response, skipping thinking content
                json_content = extract_json_from_thinking_response(response.content)
                parser = JsonOutputParser()
                result = parser.parse(json_content)

                processing_time = time.time() - llm_start_time

                # Log the interaction
                interaction = LLMInteraction(
                    agent_type="CommandSelector",
                    timestamp=datetime.now(),
                    iteration=state["iteration"],
                    messages_sent=messages_sent,
                    response_received={"content": response.content, "type": "json"},
                    processing_time=processing_time,
                    model_name=self.llm_selector.model_name
                )
                self.logger.log_llm_interaction(interaction)

            except Exception as e:
                processing_time = time.time() - llm_start_time
                error_response = {"error": str(e), "type": "error"}

                # Log the failed interaction
                interaction = LLMInteraction(
                    agent_type="CommandSelector",
                    timestamp=datetime.now(),
                    iteration=state["iteration"],
                    messages_sent=messages_sent if 'messages_sent' in locals() else [{"role": "user", "content": "Error extracting messages"}],
                    response_received=error_response,
                    processing_time=processing_time,
                    model_name=self.llm_selector.model_name
                )
                self.logger.log_llm_interaction(interaction)
                raise e

            selected_key = result.get('selected_key', 'power').lower()

            if selected_key in command_proposals:
                selected_command = command_proposals[selected_key].copy()
                selected_command['selection_reasoning'] = result.get('reasoning', 'LLM selection')
            else:
                # Fallback to power if invalid selection
                selected_command = command_proposals.get('power', {})
                selected_command['selection_reasoning'] = f'Invalid selection {selected_key}, defaulted to power'

            state["selected_command"] = selected_command

            # Log response
            response = LLMResponse(
                agent_type="CommandSelector",
                timestamp=datetime.now(),
                input_data={
                    'available_commands': list(command_proposals.keys()),
                    'iteration': state["iteration"],
                    'has_optimization_trends': 'llm_summary' in unified_analysis,
                    'has_fixing_history': len(fixing_results) > 0
                },
                output_data={'selected_key': selected_key, 'reasoning': result.get('reasoning', '')},
                processing_time=time.time() - start_time,
                iteration=state["iteration"]
            )
            self.logger.log_response(response)

        except Exception as e:
            self.logger.logger.error(f"Command selection failed: {e}")
            # Default to power command
            state["selected_command"] = command_proposals.get('power', {
                'command_type': ECOType.POWER.value,
                'tcl_command': '',
                'reasoning': f'Selection failed: {e}'
            })

        return state

    def _execute_command_node(self, state: ECOState) -> ECOState:
        """Execute the selected command"""
        start_time = time.time()

        selected_command = state["selected_command"]
        tcl_command = selected_command.get('tcl_command', '')
        command_type = selected_command.get('command_type', ECOType.POWER.value)

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

    def _formulate_tcl_command(self, tcl_command: str, command_type: str, iteration: int) -> str:
        # Fill in the command and iteration variables
        # Use a different approach - replace the {} with command first, then handle ${} variables
        tcl_command = re.sub(r'\[(.*?)\]', lambda m: '{' + ' '.join(m.group(1).split(', ')) + '}', tcl_command)
        if "drc" in tcl_command:
            tcl_command+= " -verbose -unfixable_reasons_format text -unfixable_reasons_prefix {}/reports/unfix_drc_{}".format(base_path, iteration)
        elif "timing" in tcl_command:
            tcl_command+= " -verbose -unfixable_reasons_format text -unfixable_reasons_prefix {}/reports/unfix_timing_{}".format(base_path, iteration)
        # process buffer lists
        if "buffer_insertion" in tcl_command:
            tcl_command += f" -buffer_list {BUFFER_LIST} "
        # add redirecting output
        if "opt_power" in tcl_command:
            tcl_command += " > {}/reports/fix_power_{}.txt;".format(base_path, iteration)
        elif "opt_timing" in tcl_command:
            tcl_command += " > {}/reports/fix_timing_{}.txt;".format(base_path, iteration)
        elif "opt_drc" in tcl_command:
            tcl_command += " > {}/reports/fix_drc_{}.txt;".format(base_path, iteration)

        vth_lib_mapping = {
            'RVT': 'tcbn28hpcplusbwp30p140tt0p9v85c',
            'LVT': 'tcbn28hpcplusbwp30p140lvttt0p9v85c',
            'HVT': 'tcbn28hpcplusbwp30p140hvttt0p9v85c'
        }
        all_vth_types = set(vth_lib_mapping.keys())

        # Extract VTH values from the string using regex
        vth_pattern = r'-Vth\s+\{([^}]+)\}'
        vth_match = re.search(vth_pattern, tcl_command)

        if vth_match:
            # Get the VTH values from the match
            vth_values_str = vth_match.group(1)
            # Split and clean the VTH values
            present_vth = set(vth_values_str.split())

            # Remove the entire -Vth {...} portion from the string
            processed_string = re.sub(vth_pattern, '', tcl_command).strip()
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

        # Add area limit commands if needed
        result = area_cap_prefix + result + area_cap_suffix
        result += '\n remove_target_library_subset -top; \n'
        return result
    def _execute_tcl_command(self, tcl_command: str, command_type: str, iteration: int) -> Dict[str, Any]:
        """Execute TCL command using pt_shell"""
        os.makedirs(os.path.join(base_path, 'logs'), exist_ok=True)

        # Read the execute_command.tcl template
        template_path = os.path.join(base_path, 'EDA_scripts/execute_command.tcl')
        try:
            with open(template_path, 'r') as f:
                template_content = f.read()
        except FileNotFoundError:
            return {
                'success': False,
                'message': 'Template file not found',
                'execution_time': 0,
                'command_type': command_type,
                'output': f'Could not find template: {template_path}'
            }
        ori_tcl_command = copy.deepcopy(tcl_command)
        tcl_command = self._formulate_tcl_command(tcl_command, command_type, iteration)
        filled_content = template_content.replace('{}', tcl_command)
        filled_content = filled_content.replace('${i}', str(iteration))
        filled_content = filled_content.replace('${i+1}', str(iteration + 1))

        # Write the filled script
        script_path = os.path.join(base_path, f'Run_scripts_{iteration}.tcl')
        with open(script_path, 'w') as f:
            f.write(filled_content)

        # Execute pt_shell
        start_execution_time = time.time()
        log_path = os.path.join(base_path, f'logs/pt_{iteration}.log')
        command = f'cd {base_path} && pt_shell -f {script_path} | tee {log_path}'

        self.logger.logger.info(f"Executing pt_shell command: {command}")
        exit_code = os.system(command)
        exit_code = 0 # for debugging
        end_execution_time = time.time()
        execution_time = end_execution_time - start_execution_time

        # Read the log output
        try:
            with open(log_path, 'r') as f:
                output = f.read()
        except FileNotFoundError:
            output = "Log file not found"

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

    def run_iteration(self, reports: Dict[str, str]) -> Dict[str, Any]:
        """Run one complete ECO iteration using persistent state"""
        self.iteration_count += 1
        iteration_start = time.time()
        elapsed_runtime = time.time() - self.start_time
        remaining_budget = self.total_runtime_budget - elapsed_runtime

        self.logger.logger.info(f"Starting ECO iteration {self.iteration_count}")

        # First iteration: Create initial persistent state
        if self.persistent_state is None:
            self.logger.logger.info("Creating initial ECO state for first iteration")
            self.persistent_state = ECOState(
                iteration=self.iteration_count,
                reports=reports,
                unified_analysis={},
                command_proposals={},
                selected_command={},
                execution_result={},
                elapsed_runtime=elapsed_runtime,
                remaining_budget=remaining_budget,
                messages=[],
                system_status="running"
            )
        else:
            # Subsequent iterations: Update persistent state while preserving accumulated context
            self.logger.logger.info(f"Updating persistent state for iteration {self.iteration_count}")

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
                'elapsed_runtime': elapsed_runtime,
                'remaining_budget': remaining_budget,
                'system_status': "running",
                'messages': preserved_messages  # Explicitly preserve messages
                # NOTE: unified_analysis, command_proposals are preserved!
                # selected_command and execution_result will be updated by workflow
            })

        # Run the workflow with persistent state
        # config = {"configurable": {"thread_id": f"eco_optimization_persistent"}}  # Use same thread for continuity

        try:
            final_state = self.workflow.invoke(self.persistent_state)

            # Update persistent state with final results
            self.persistent_state = final_state

            iteration_time = time.time() - iteration_start
            total_elapsed = elapsed_runtime + iteration_time

            result = {
                "iteration": self.iteration_count,
                "unified_analysis": final_state["unified_analysis"],
                "command_proposals": final_state["command_proposals"],
                "selected_command": final_state["selected_command"],
                "execution_result": final_state["execution_result"],
                "iteration_time": iteration_time,
                "total_elapsed": total_elapsed,
                "remaining_budget": self.total_runtime_budget - total_elapsed,
                "system_status": "healthy" if total_elapsed < self.total_runtime_budget else "budget_exceeded"
            }

            self.logger.logger.info(f"ECO iteration {self.iteration_count} completed in {iteration_time:.2f}s")
            return result

        except Exception as e:
            self.logger.logger.error(f"ECO iteration {self.iteration_count} failed: {e}")
            return {
                "iteration": self.iteration_count,
                "system_failure": True,
                "error_message": str(e),
                "iteration_time": time.time() - iteration_start,
                "total_elapsed": elapsed_runtime + (time.time() - iteration_start),
                "remaining_budget": remaining_budget,
                "system_status": "failed"
            }

    def should_continue_optimization(self) -> bool:
        """Determine if optimization should continue"""
        elapsed_time = time.time() - self.start_time
        remaining_budget = self.total_runtime_budget - elapsed_time

        if remaining_budget <= 0:
            self.logger.logger.info("Optimization stopped: Runtime budget exhausted")
            return False

        if self.iteration_count >= 20:  # Arbitrary limit
            self.logger.logger.info("Optimization stopped: Maximum iterations reached")
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


def load_reports_for_iteration(iteration: int, last_command_type: ECOType = None, last_reports: dict = {}) -> Dict[str, Any]:
    """Load and parse reports for a given iteration using eco_database ReportParser"""
    if iteration > 0:
        last_power_data = copy.deepcopy(last_reports.get("power", {}))
    parsed_reports = {}

    if iteration == 0:
        # First iteration: baseline reports only
        report_files = {
            "qor": os.path.join(base_path, "reports/report_qor_0.txt"),
            "power": os.path.join(base_path, "reports/report_power_0.txt")
        }
        print(f"Loading baseline reports for iteration {iteration}...")
    else:
        # Subsequent iterations: baseline + ECO fixing reports based on last command
        report_files = {
            "qor": os.path.join(base_path, f"reports/report_qor_{iteration}.txt"),
            "power": os.path.join(base_path, f"reports/report_power_{iteration}.txt")
        }

        # Add ECO fixing reports based on last command type
        # Use previous iteration index for ECO fix reports
        prev_iteration = iteration - 1
        if last_command_type == ECOType.DRC:
            report_files["drc_fix"] = os.path.join(base_path, f"reports/fix_drc_{prev_iteration}.txt")
            # Check for unfixing report
            unfix_file = os.path.join(base_path, f"reports/unfix_drc_{prev_iteration}_eco_drc.txt")
            if os.path.exists(unfix_file):
                report_files["drc_unfix"] = unfix_file
        elif last_command_type == ECOType.TIMING:
            report_files["timing_fix"] = os.path.join(base_path, f"reports/fix_timing_{prev_iteration}.txt")
            # Check for unfixing report
            unfix_file = os.path.join(base_path, f"reports/unfix_timing_{prev_iteration}_eco_tim.txt")
            if os.path.exists(unfix_file):
                report_files["timing_unfix"] = unfix_file
        elif last_command_type == ECOType.POWER:
            report_files["power_fix"] = os.path.join(base_path, f"reports/fix_power_{prev_iteration}.txt")

        print(f"Loading reports for iteration {iteration} (last command: {last_command_type.value if last_command_type else 'None'})")
        print(f"Report files: {list(report_files.keys())}")


    # Load and parse each report
    for report_type, file_path in report_files.items():
        with open(file_path, 'r') as f:
            content = f.read()

        # Parse reports using eco_database ReportParser
        if report_type == "qor":
            parsed_qor = ReportParser.parse_qor_report(content)
            parsed_reports["timing"] = {"setup_violating_paths": parsed_qor.get("setup_violating_paths", 0), "hold_violating_paths": parsed_qor.get("hold_violating_paths", 0), "setup_critical_path_slack": parsed_qor.get("setup_critical_path_slack", 0.0), "hold_critical_path_slack": parsed_qor.get("hold_critical_path_slack", 0.0)}
            parsed_reports['area'] = {"design_area": parsed_qor.get("design_area", 0.0)}
            parsed_reports['drc'] = {"min_capacitance_count": parsed_qor.get("min_capacitance_count", 0), "max_transition_count": parsed_qor.get("max_transition_count", 0), "total_drc_cost": parsed_qor.get("total_drc_cost", 0.0)}

        elif report_type == "power":
            parsed_reports["power"] = ReportParser.parse_power_report(content)
    if iteration > 0:
        match last_command_type:
            case ECOType.TIMING:
                content = open(report_files['timing_fix'], 'r').read() if 'timing_fix' in report_files else ''
                parsed_reports["last_fix"] = ReportParser.parse_timing_log(content) if 'timing_fix' in report_files else {}
                unfix_content = open(report_files['timing_unfix'], 'r').read() if 'timing_unfix' in report_files else ''
                parsed_reports["last_unfix"] = ReportParser.parse_timing_unfixing(unfix_content) if 'timing_unfix' in report_files else {}
            case ECOType.DRC:
                content = open(report_files['drc_fix'], 'r').read() if 'drc_fix' in report_files else ''
                parsed_reports["last_fix"] = ReportParser.parse_drc_log(content) if 'drc_fix' in report_files else {}
                unfix_content = open(report_files['drc_unfix'], 'r').read() if 'drc_unfix' in report_files else ''
                parsed_reports["last_unfix"] = ReportParser.parse_drc_unfixing(report_files['drc_unfix']) if 'drc_unfix' in report_files else {}
            case ECOType.POWER:
                content = open(report_files['power_fix'], 'r').read() if 'power_fix' in report_files else ''
                power_fix = ReportParser.parse_power_log(content) if 'power_fix' in report_files else {}
                elapsed_time_seconds =  power_fix.get('elapsed_time_seconds', 0)
                parsed_reports['last_fix'] = {'fix_type': 'power', 'elapsed_time_seconds': elapsed_time_seconds,'total_power_decreased': last_power_data['total_power'] - parsed_reports['power']['total_power'], 'leakage_power_decreased': last_power_data['leakage_power'] - parsed_reports['power']['leakage_power'], 'dynamic_power_decreased': (last_power_data['total_power'] - last_power_data['leakage_power']) - (parsed_reports['power']['total_power'] - parsed_reports['power']['leakage_power']), 'total_area_decreased': power_fix.get('total_area_decreased', 0.0), 'fixes': power_fix.get('fixes', '')}
                parsed_reports["last_unfix"] = {"fix_type": "power", "fixes": "Power fixing does not have unfix report because power is not a constraint."}


    return parsed_reports


def create_eco_system(runtime_budget: float = 3600, log_file: str = "eco_agent_responses_langchain.json") -> ECOLangChainSystem:
    """Factory function to create a configured ECO system"""
    return ECOLangChainSystem(runtime_budget, log_file)




def simulate_eco_iterations(num_iterations: int = 3):
    """Test the LangChain ECO system"""
    system = create_eco_system(2000)

    print(f"ECO LangChain System Test Started")
    print(f"Running {num_iterations} iterations")
    print("=" * 70)

    results = []
    last_command_type = None  # Track the last executed command type
    reports = {}
    for i in range(num_iterations):
        if not system.should_continue_optimization():
            print("Optimization stopped due to budget or convergence criteria")
            break

        print(f"\nIteration {i}")
        print("-" * 50)


        # Load reports for this iteration, using last command type from previous iteration
        reports = load_reports_for_iteration(i, last_command_type=last_command_type, last_reports=reports)

        if not reports:
            print(f"No reports found for iteration {i}, stopping")
            break

        try:
            result = system.run_iteration(reports)
            results.append(result)

            if result.get('system_failure'):
                print(f"Iteration {i} failed: {result.get('error_message', 'Unknown error')}")
                break
            else:
                selected_command = result.get('selected_command', {})
                execution_result = result.get('execution_result', {})

                print(f"✅ Iteration {i} Results:")
                print(f"  • Selected Command: {selected_command.get('command_type', 'Unknown')}")
                print(f"  • Execution Success: {execution_result.get('success', False)}")
                print(f"  • Iteration Time: {result['iteration_time']:.1f}s")
                print(f"  • System Status: {result['system_status']}")

                # Update last_command_type for next iteration's report loading
                command_type_str = selected_command.get('command_type', '')
                if command_type_str:
                    # Convert string to ECOType enum
                    if command_type_str.lower() == 'drc':
                        last_command_type = ECOType.DRC
                    elif command_type_str.lower() == 'timing':
                        last_command_type = ECOType.TIMING
                    elif command_type_str.lower() == 'power':
                        last_command_type = ECOType.POWER
                    else:
                        last_command_type = None
                    print(f"  • Next iteration will load reports for: {last_command_type.value if last_command_type else 'None'}")
                else:
                    last_command_type = None

        except Exception as e:
            print(f"Iteration {i} failed with exception: {e}")
            break

    print(f"\n🎯 Test completed - {len(results)} iterations")
    return results


if __name__ == "__main__":
    simulate_eco_iterations(5)
