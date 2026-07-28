"""
RAG helper functions for ECO optimization system
Phase 3: Node Integration helper functions
"""

import json
import time
from typing import Dict, List, Any
from datetime import datetime

from langchain_core.output_parsers import JsonOutputParser

def extract_current_state(reports: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structured data from parsed reports"""
    current_state = {}
    
    # Handle structured QoR data
    qor_data = reports['timing']
    current_state['timing'] = {
        'setup_violating_paths': qor_data['setup_violating_paths'],
        'hold_violating_paths': qor_data['hold_violating_paths'],
        'setup_critical_path_slack': round(qor_data['setup_critical_path_slack'], 4),
        'hold_critical_path_slack': round(qor_data['hold_critical_path_slack'], 4),
        'setup_total_negative_slack': round(qor_data['setup_total_negative_slack'], 4),
        'hold_total_negative_slack': round(qor_data['hold_total_negative_slack'], 4),
    }

    area_data = reports['area']
    current_state['area'] = {
        'design_area': round(area_data['design_area'], 4)
    }
    
    # Handle structured power data
    power_data = reports['power']
    current_state['power'] = {
        'total_power': round(power_data['total_power'], 4),
        'dynamic_power': round(power_data['total_power'] - power_data['leakage_power'], 4),
        'leakage_power': round(power_data['leakage_power'], 4),
        'clock_tree_power': round(power_data['clock_tree_power'], 4),
        'combinational_power': round(power_data['combinational_power'], 4),
        'register_power': round(power_data['register_power'], 4)
    }
    
    return current_state

def extract_fixing_results(reports: Dict[str, Any], state: Dict, iteration: int):
    """Extract fixing results and unfixable reasons for non-zero iterations"""
    fixing_results = {}
    unfixable_reasons = {}
    
    if iteration > 0:
        # Get the actual optimization type from the previous iteration's executed command
        prev_analysis = state['unified_analysis']
        prev_iteration_key = f'iteration_{iteration-1}'
        prev_execution_result = prev_analysis['iteration_history'][prev_iteration_key]['execution_result']
        prev_tcl_command = prev_execution_result['tcl_command']
        buffer_removal_used = 'buffer_removal' in prev_tcl_command.lower()
        
        # Process fixing reports based on available fix reports
        if 'timing_fix' in reports:
            timing_fix = reports['timing_fix']
            timing_unfix = reports['timing_unfix']
            fixing_results = {
                'last_optimization_command': prev_tcl_command,
                'elapsed_time_seconds': timing_fix['elapsed_time_seconds'], 
                'detected_violations': timing_fix['total_violating_endpoints_found'], 
                'total_violating_endpoints_fixed': timing_fix['total_violating_endpoints_fixed'], 
                'percentage_violations_fixed': timing_fix['percentage_violations_fixed']
            }
            
            # Filter out raw_content from timing_unfix before extracting summary
            timing_unfix_filtered = {k: v for k, v in timing_unfix.items() if k not in ['raw_content', 'timing_paths', 'paths_sorted_by_slack']}
            unfixable_reasons = {
                "fix_type": timing_unfix_filtered['fix_type'],
                "violations_summary": timing_unfix_filtered['summary'],
            }

        elif 'area_fix' in reports:
            area_fix = reports['area_fix']
            area_unfix = reports['area_unfix']
            fixing_results = {
                'last_optimization_command': prev_tcl_command,
                'elapsed_time_seconds': area_fix['elapsed_time_seconds']
            }

            # Add fields based on report format identifier (mutually exclusive)
            if area_fix['report_format'] == 'eco_summary':
                # "Final ECO Summary" format - has area decrease metrics
                fixing_results['total_area_decreased'] = area_fix['total_area_decreased']
                fixing_results['percentage_area_decreased'] = area_fix['percentage_area_decreased']
            elif area_fix['report_format'] == 'fixing_summary' and buffer_removal_used:
                # "Fixing Summary" format - has buffer removal metrics
                fixing_results['buffers_removed'] = area_fix['buffers_removed']
                fixing_results['percentage_buffers_removed'] = area_fix['percentage_buffers_removed']
                fixing_results['percentage_cells_removed'] = area_fix['percentage_cells_removed']

            unfixable_reasons = {k: v for k, v in area_unfix.items() if k not in ['raw_content', 'reason_definitions']}
        
        # Process power fixing reports
        elif 'power_fix' in reports:
            power_fix = reports['power_fix']
            power_unfix = reports['power_unfix']
            fixing_results = {
                'last_optimization_command': prev_tcl_command,
                'elapsed_time_seconds': power_fix['elapsed_time_seconds']
            }

            # Add fields based on report format identifier (mutually exclusive)
            # NOTE: Power fixing reports only show area metrics, not power metrics
            # Power reduction is achieved by reducing cell sizes, which reduces area
            if power_fix['report_format'] == 'eco_summary':
                # "Final ECO Summary" format - has area decrease metrics
                fixing_results['total_area_decreased'] = round(power_fix['total_area_decreased'], 4)
                fixing_results['percentage_area_decreased'] = round(power_fix['percentage_area_decreased'], 4)
            elif power_fix['report_format'] == 'fixing_summary' and buffer_removal_used:
                # "Fixing Summary" format - has buffer removal metrics
                fixing_results['buffers_removed'] = power_fix['buffers_removed']
                fixing_results['percentage_buffers_removed'] = power_fix['percentage_buffers_removed']
                fixing_results['percentage_cells_removed'] = power_fix['percentage_cells_removed']

            unfixable_reasons = {k: v for k, v in power_unfix.items() if k not in ['raw_content', 'reason_definitions']}
            
    return fixing_results, unfixable_reasons

def generate_target_selection_queries(TOOL_USING: bool, current_state: Dict, fixing_results: Dict, unfixable_reasons: Dict, iteration: int, round_index: int, state: Dict, llm, design_state_history: List, unfixable_analysis_history: List, rag_messages: List, logger) -> List[str]:
    """LLM Call #1: Generate informed queries for knowledge retrieval based on current design state"""
    from utils import extract_content_from_llm_response, extract_json_from_thinking_response
    from agent_logs import LLMInteraction
    from langchain_core.messages import SystemMessage, HumanMessage

    # Define system prompt for query generation
    system_prompt = """You are an expert IC design ECO (Engineering Change Order) optimization engineer responsible for comprehensive analysis.
You will work as an reviewer engineer to evaluate the current design state and optimization history regarding timing, power, and area. Then generate queries to retrieve relevant knowledge to assist in decision-making for the ECO iterations.

**ECO Background**: ECO is an incremental design optimization process that iteratively improves the design by fixing violations and optimizing metrics such as timing, area, and power. Each iteration involves analyzing the current design state and optimization histories and the optimization of one target metric with specific optimization options. The success of ECO lies in: 1. Extensively explore optimization targets (timing, area, power); 2. Keep track of the tradeoff among 3 targets to ensure balanced optimization and do recovery of a metrics if it is deteriorated due to optimizing other metrics; 3. Do not waste time on unpromissing targets.
The timing slacks are the more positive the better, while area and power are the more lower the better. The optimization benefit tend to decrease as the proceeding of ECO iterations. The goal is to achieve a balanced optimization across timing, area, and power while adhering to a total iteration budget.

**Report Content**:
- CURRENT DESIGN STATE: The current design state includes timing, power, and area metrics after the most recent optimization iteration.
- OPTIMIZATION HISTORY: The optimization history includes the design states and already performed optimization commands from all previous iterations.
- OBJECTIVES: The objectives describe optimization priorities for this run.

**Task**: Based on your analysis, generate the following content (Do not repeat design states and history in your evaluation with detailed values.):
1. Your evaluation on the optimization trend of timing, power, area from previous iterations.
2. Generate exactly 1 query to retrieve relevant knowledge to assist in decision-making. Focus on: 
    - Optimization target strategy: What is the optimization target selection strategy based on your evaluation of optimization trends? 
    - Do not repeat any detailed values in your query.
    - Focus on a general strategy.
    (Some possible query contents: "We have tried optimize target timing for x iterations, power for y iterations", "We have optimized timing for xx, but increased power and area for xx", "We have not explore the optimization of power, area, timing thoroughly", "The iteration budget has used xx percent, while we have not optimize xx target") You can combine them, think of new ones and make your own query.

**Evaluation Content**:
1. Evaluate the trend of timing, area, power metrics. Use the best achieved value (the default is iteration 0) as the optimization baseline.
2. Evaluate the spent optimization iterations on optimizing all three targets (timing, area, power). The iteration budget at iteration 0 will be the total budget.
3. Evaluate the iterations that they have major optimization effect, e.g., the improving timing at the cost of increasing power and area.
4. Evaluate the effectiveness of previous optimization histories based on the metrics' change and the feedback comment from executors (for iteration > 0).
**Evaluation Guidelines**: 
1. Only make evaluation on given data, do not make any conjecture and comments on optimization strategy or priorities.
2. The command at each iteration will affect the design state at the next iteration, not the current one.
3. For timing, you should have seperate evaluation for setup and hold violations.

**RESPONSE FORMAT**: Always respond in valid JSON format.
Example:
{
  "evaluation": "Your evaluation here, no more than 200 words.",
  "query": "query on combined target and option selection strategy, use plain text, no more than 40 words."
}
        """

    timing_data = current_state['timing']
    power_data = current_state['power']
    area_data = current_state['area']
    remaining_iterations = design_state_history[-1]['remaining_iterations']

    # Build comprehensive report analysis prompt
    current_state_section = [
        f"Iteration: {iteration}, Remaining iterations: {remaining_iterations}",
        f"- Setup violations: {timing_data['setup_violating_paths']} paths",
        f"- Hold violations: {timing_data['hold_violating_paths']} paths",
        f"- Setup critical slack: {timing_data['setup_critical_path_slack']}",
        f"- Hold critical slack: {timing_data['hold_critical_path_slack']}",
        f"- Setup total negative slack: {timing_data['setup_total_negative_slack']}",
        f"- Hold total negative slack: {timing_data['hold_total_negative_slack']}",
        f"- Total power: {power_data['total_power']:.3e}W",
        f"- Dynamic power: {power_data['dynamic_power']:.3e}W",
        f"- Leakage power: {power_data['leakage_power']:.3e}W",
        f"- Design area: {area_data['design_area']} um^2",
    ]

    objectives = state["objectives"]
    analysis_prompt_parts = [
        f"**TASK: Analyze current design state and generate strategic queries for knowledge retrieval**",
        f"**OBJECTIVES:** {objectives}",
        f"**CURRENT DESIGN STATE:**\t" + "\t".join(current_state_section)
    ]

    analysis_prompt_parts.append(f"\n**OPTIMIZATION HISTORY:**")
    if len(design_state_history) <=1:
        analysis_prompt_parts.append("\tNo previous optimization history available.")
    # Add optimization history if available (only previous iterations, not current)
    elif len(design_state_history) > 1:  # More than just current iteration
        analysis_prompt_parts.append("Optimization history listed in chronological order.")
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
        hist_remaining_budgets = []
        hist_execution_times = []
        hist_executed_commands = []
        for hist_entry in design_state_history[:-1]:  # Exclude current iteration (already shown above)
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
            hist_dynamic_powers.append(str(hist_state['power']['dynamic_power']))
            hist_leakage_powers.append(str(hist_state['power']['leakage_power']))
            hist_design_areas.append(str(hist_state['area']['design_area']))
            hist_remaining_budgets.append(f"{hist_entry['remaining_budget']:.0f}s")
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
            f"- Dynamic power: {hist_dynamic_powers}W",
            f"- Leakage power: {hist_leakage_powers}W",
            f"- Design area: {hist_design_areas} um^2",
            f"- Actual execution times: {hist_execution_times} s",
            f"- Executed commands: {hist_executed_commands}"
        ]
        analysis_prompt_parts.append("\t".join(hist_parts))

    # Do not use unfixable analysis for now
    # if unfixable_analysis_history:
    #     analysis_prompt_parts.append(f"\n**UNFIXABLE ISSUES HISTORY:**")
    #     if TOOL_USING:
    #         analysis_prompt_parts.append("\tExplanation for reason_distribution: The unfixable reasons are weighted by the negative slack they causes. Explanation for slack_distribution: The number of paths in each negative slack range.\n")
    #     for unfixable_entry in unfixable_analysis_history:
    #         unfixable_iteration = unfixable_entry['iteration']
    #         unfixable_data = unfixable_entry['unfixable_reasons']
    #         # Use json.dumps and escape curly braces for f-string
    #         import json
    #         unfixable_str = json.dumps(unfixable_data).replace('{', '{{').replace('}', '}}')
    #         analysis_prompt_parts.append(f"Iteration {unfixable_iteration}: {unfixable_str}")

    analysis_prompt_parts.extend([
        f"\n**Now You Can generate the response with JSON resonse:**",
        f"{{{{",
        f'  "evaluation": "Your evaluation here, no more than 200 words.",',
        f'  "query": "query on optimization target, use plain text, no more than 40 words."',
        f"}}}}"
    ])
    
    # Add system prompt only if this is the first message in RAG conversation
    if not rag_messages:
        rag_messages.append(SystemMessage(content=system_prompt))

    # Add query generation human message
    rag_messages.append(HumanMessage(content="\n".join(analysis_prompt_parts)))

    # LLM Call #1: Generate queries
    llm_start_time = time.time()
    messages_sent = [{"role": msg.type, "content": msg.content} for msg in rag_messages]

    # Invoke LLM with conversation history
    response = llm.invoke(rag_messages)
    processing_time = time.time() - llm_start_time
    # Extract content from response (handles both string and list formats)
    content_text = extract_content_from_llm_response(response.content)
    token_usage = {
        "input_tokens": response.usage_metadata["input_tokens"],
        "output_tokens": response.usage_metadata["output_tokens"]
    }

    # Add AI response to maintain conversation
    from langchain_core.messages import AIMessage
    rag_messages.append(AIMessage(content=content_text))

    # Parse JSON response - expecting: {"queries": ["query1", "query2", "query3"]}
    import re
    json_content = extract_json_from_thinking_response(content_text)
    json_content = re.sub(r'\n\s*\n+', '\n', json_content)
    json_content = re.sub(
        r'("evaluation"\s*:\s*"(?:[^"\\]|\\.)*")\s*("query"\s*:)',
        r'\1,\n  \2',
        json_content,
        flags=re.DOTALL
    )
    parser = JsonOutputParser()
    analysis_result = parser.parse(json_content)

    # Extract queries
    queries = analysis_result['query']
    evaluation = analysis_result['evaluation']
    

    # Log analysis and query generation interaction
    interaction = LLMInteraction(
        agent_type="SummaryAgent_AnalysisAndQueryGeneration",
        timestamp=datetime.now(),
        iteration=iteration,
        round_index=round_index,
        messages_sent=messages_sent,
        response_received={
            "content": content_text,
            "type": "json",
            "parsed_decision": {
                "evaluation": evaluation,
                "query": queries
            }
        },
        processing_time=processing_time,
        model_name=llm.model_name,
        token_usage=token_usage
    )
    logger.log_llm_interaction(interaction)

    # Return queries and timing
    return queries, evaluation, processing_time

def generate_selection_queries(TOOL_USING: bool, current_state: Dict, fixing_results: Dict, unfixable_reasons: Dict, iteration: int, round_index: int, state: Dict, llm, design_state_history: List, unfixable_analysis_history: List, rag_messages: List, logger) -> List[str]:
    """LLM Call #1: Generate combined target/option selection query for knowledge retrieval"""
    from utils import extract_content_from_llm_response, extract_json_from_thinking_response
    from agent_logs import LLMInteraction
    from langchain_core.messages import SystemMessage, HumanMessage

    system_prompt = """You are an expert IC design ECO (Engineering Change Order) optimization engineer responsible for comprehensive analysis and scheduling.
You will work as an reviewer engineer to evaluate the current design state, optimization history, and unfixable reasons regarding timing, power, and area. Then generate a single query to retrieve relevant knowledge for both optimization target selection and option selection strategy.

**ECO Background**: ECO is an incremental design optimization process that iteratively improves the design by fixing violations and optimizing metrics such as timing, area, and power. Each iteration involves analyzing the current design state and optimization histories and the optimization of one target metric with specific optimization options. The success of ECO lies in: 1. Extensively explore optimization targets (timing, area, power) and optimization options for each target; 2. Keep track of the tradeoff among 3 targets to ensure balanced optimization and do recovery of a metrics if it is deteriorated due to optimizing other metrics; 3. Identify unfixable reasons from previous optimizations and select options that can avoid those unfixable reasons or fix them; 4. Do not waste time on unpromissing targets.
The timing slacks are the more positive the better, while area and power are the more lower the better. The optimization benefit tend to decrease as the proceeding of ECO iterations. The goal is to achieve a balanced optimization across timing, area, and power while adhering to a total iteration budget.

**Report Content**:
- CURRENT DESIGN STATE: The current design state includes timing, power, and area metrics after the most recent optimization iteration.
- OPTIMIZATION HISTORY: The optimization history includes the design states and already performed optimization commands from all previous iterations.
- UNFIXABLE ISSUES HISTORY: The unfixable issues history includes the design states and reasons for unfixable violations from all previous iterations.
- OBJECTIVES: The objectives describe optimization priorities for this run.

**Task**: Based on your analysis, generate the following content (Do not repeat design states, history, or unfixable reasons in your evaluation with detailed values.):
1. Your evaluation on the optimization trend of timing, power, area from previous iterations.
2. Your evaluation on unfixable reasons of timing, power, area from previous iterations. If at iteration 0 without unfixable reasons, mark the unfixable reasons evaluation as "N/A".
3. Generate exactly 1 query to retrieve relevant knowledge to assist in decision-making. Focus on:
    - Optimization target selection strategy based on your trend evaluation.
    - Optimization option selection strategy (Exploration/Exploitation) based on unfixable reasons.
    - Do not repeat any detailed values in your query.
    - Focus on a general strategy.
    (Some possible query contents: "We have tried optimize target timing for x iterations, power for y iterations", "We have optimized timing for xx, but increased power and area for xx", "New unfixable reasons are emerging after optimizing timing", "Previous unfixable reasons are diminishing but timing still lags") You can combine them, think of new ones and make your own query.

**Evaluation Content**:
1. Evaluate the trend of timing, area, power metrics. Use the best achieved value (the default is iteration 0) as the optimization baseline.
2. Evaluate the spent optimization iterations on optimizing all three targets (timing, area, power). The iteration budget at iteration 0 will be the total budget.
3. Evaluate the iterations that they have major optimization effect, e.g., improving timing at the cost of increasing power and area.
4. Evaluate the effectiveness of previous optimization histories based on the metrics' change and the feedback comment from executors (for iteration > 0).
5. Evaluate the optimization benefit on each target (power/area/timing) and the emerged unfixable reasons.
6. Based on the evaluation, correlate the optimization target with the emergence of new unfixable reasons and fixed unfixable reasons.
**Evaluation Guidelines**: 
1. Only make evaluation on given data, do not make any conjecture and comments on optimization strategy or priorities.
2. The command at each iteration will affect the design state at the next iteration, not the current one.
3. For timing, you should have seperate evaluation for setup and hold violations.
4. Do not mention any detailed optimization actions like gate sizing.

Definitions for Unfixable Reasons for Timing:
A - There are available library cells outside area limit.       B - Delay improvement is too small to fix the violation.        C - The violation is in clock tree.  D - Cell or net is located in high density area.        I - Buffer insertion with given library cells cannot fix the violation. L - Available physical area limits the use of one or more library cells.        O - No open free site is available.     R - No locations are available in parasitics or location transformation failed. S - Cell sizing with alternative library cells cannot fix the violation.        T - Timing margin is too tight to fix the violation.    W - Fixing the violation might degrade DRC violations.

Definitions for Unfixable Reasons for Power/Area:
B - Benefit from sizing cell is too small       L - Physical constraints restrict sizing        S - Cell has no alternate library cell with better power        T - Sizing cell might degrade timing    W - Sizing cell might degrade DRC       X - Cell is unusable for ECO    Z - Cell is sized

**RESPONSE FORMAT**: Always respond in valid JSON format.
Example:
{
  "evaluation": "Your evaluation here, no more than 200 words.",
  "query": "query on combined target and option selection strategy, use plain text, no more than 40 words."
}
        """

    timing_data = current_state['timing']
    power_data = current_state['power']
    area_data = current_state['area']
    remaining_iterations = design_state_history[-1]['remaining_iterations']

    current_state_section = [
        f"Iteration: {iteration}, Remaining iterations: {remaining_iterations}",
        f"- Setup violations: {timing_data['setup_violating_paths']} paths",
        f"- Hold violations: {timing_data['hold_violating_paths']} paths",
        f"- Setup critical slack: {timing_data['setup_critical_path_slack']}",
        f"- Hold critical slack: {timing_data['hold_critical_path_slack']}",
        f"- Setup total negative slack: {timing_data['setup_total_negative_slack']}",
        f"- Hold total negative slack: {timing_data['hold_total_negative_slack']}",
        f"- Total power: {power_data['total_power']:.3e}W",
        f"- Dynamic power: {power_data['dynamic_power']:.3e}W",
        f"- Leakage power: {power_data['leakage_power']:.3e}W",
        f"- Design area: {area_data['design_area']} um^2",
    ]

    objectives = state["objectives"]
    analysis_prompt_parts = [
        f"**TASK: Analyze current design state, optimization history, and unfixable reasons to generate strategic queries for knowledge retrieval**",
        f"**OBJECTIVES:** {objectives}",
        f"**CURRENT DESIGN STATE:**\t" + "\t".join(current_state_section)
    ]

    analysis_prompt_parts.append(f"\n**OPTIMIZATION HISTORY:**")
    if len(design_state_history) <= 1:
        analysis_prompt_parts.append("No previous optimization history available.")
    elif len(design_state_history) > 1:
        analysis_prompt_parts.append("Optimization history listed in chronological order.")
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
        hist_remaining_budgets = []
        hist_execution_times = []
        hist_executed_commands = []
        for hist_entry in design_state_history[:-1]:
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
            hist_dynamic_powers.append(str(hist_state['power']['dynamic_power']))
            hist_leakage_powers.append(str(hist_state['power']['leakage_power']))
            hist_design_areas.append(str(hist_state['area']['design_area']))
            hist_remaining_budgets.append(f"{hist_entry['remaining_budget']:.0f}s")
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
            f"- Dynamic power: {hist_dynamic_powers}W",
            f"- Leakage power: {hist_leakage_powers}W",
            f"- Design area: {hist_design_areas} um^2",
            f"- Actual execution times: {hist_execution_times} s",
            f"- Executed commands: {hist_executed_commands}"
        ]
        analysis_prompt_parts.append("\t".join(hist_parts))

    analysis_prompt_parts.append(f"\n**UNFIXABLE REASON HISTORY:**")
    if unfixable_analysis_history:
        if TOOL_USING:
            analysis_prompt_parts.append("Explanation for reason_distribution: The unfixable reasons are weighted by the negative slack they causes. Explanation for slack_distribution: The number of paths in each negative slack range.\n")
        for unfixable_entry in unfixable_analysis_history:
            unfixable_iteration = unfixable_entry['iteration']
            unfixable_data = unfixable_entry['unfixable_reasons']
            unfixable_str = json.dumps(unfixable_data).replace('{', '{{').replace('}', '}}')
            analysis_prompt_parts.append(f"Iteration {unfixable_iteration}: {unfixable_str}")
    else:
        analysis_prompt_parts.append("\tNo unfixable reason history available.")

    analysis_prompt_parts.append("\nNow generate the response in the required JSON format.")

    if not rag_messages:
        rag_messages.append(SystemMessage(content=system_prompt))

    rag_messages.append(HumanMessage(content="\n".join(analysis_prompt_parts)))

    llm_start_time = time.time()
    messages_sent = [{"role": msg.type, "content": msg.content} for msg in rag_messages]

    response = llm.invoke(rag_messages)
    processing_time = time.time() - llm_start_time
    content_text = extract_content_from_llm_response(response.content)
    token_usage = {
        "input_tokens": response.usage_metadata["input_tokens"],
        "output_tokens": response.usage_metadata["output_tokens"]
    }

    from langchain_core.messages import AIMessage
    rag_messages.append(AIMessage(content=content_text))

    json_content = extract_json_from_thinking_response(content_text)
    parser = JsonOutputParser()
    analysis_result = parser.parse(json_content)

    queries = analysis_result['query']
    evaluation = analysis_result['evaluation']

    interaction = LLMInteraction(
        agent_type="SummaryAgent_AnalysisAndQueryGeneration",
        timestamp=datetime.now(),
        iteration=iteration,
        round_index=round_index,
        messages_sent=messages_sent,
        response_received={
            "content": content_text,
            "type": "json",
            "parsed_decision": {
                "evaluation": evaluation,
                "query": queries
            }
        },
        processing_time=processing_time,
        model_name=llm.model_name,
        token_usage=token_usage
    )
    logger.log_llm_interaction(interaction)

    return queries, evaluation, processing_time

def retrieve_knowledge(queries, rag_system, k: int = 1) -> str:
    """Retrieve knowledge using RAG system"""
    if not queries:
        raise ValueError("No queries provided for knowledge retrieval")
    if isinstance(queries, str):
        queries = [queries]

    rag_results = rag_system.query_knowledge(queries, k=k)

    raw_contents = []
    for result in rag_results:
        for doc in result["retrieved_docs"]:
            raw_contents.append(doc["content"])

    return "\n\n".join(raw_contents)

def generate_option_selection_queries(TOOL_USING: bool, evaluation: str, iteration: int, round_index: int, state: Dict, llm, design_state_history: List, unfixable_analysis_history: List, rag_messages: List, logger):
    """LLM Call #2: Make final optimization decision using consolidated system prompt with optimization history and retrieved knowledge"""
    from utils import extract_content_from_llm_response, extract_json_from_thinking_response
    from agent_logs import LLMInteraction
    from langchain_core.messages import SystemMessage, HumanMessage

    # Build comprehensive system prompt with role, history, and retrieved knowledge
    system_prompt_parts = [
        """You are an expert IC design ECO (Engineering Change Order) optimization engineer responsible for optimization scheduling.
You will work as an reviewer engineer to evaluate the unfixable reasons regarding timing, power, and area. Then generate queries to retrieve relevant knowledge to assist in option-selection strategy for the ECO iterations.

**ECO Background**: ECO is an incremental design optimization process that iteratively improves the design by fixing violations and optimizing metrics such as timing, area, and power. Each iteration involvesthe optimization of one target metric with optimization option. The success of ECO lies in: 1. Extensively explore optimization target (timing, area, power) and optimization options for each target; 2. Identify unfixable reasons from previous optimizations; 3. Select optimization options that can avoid those unfixable reasons or can target the fixing of those unfixable reasons.
The timing slacks are the more positive the better, while area and power are the more lower the better. The optimization benefit tend to decrease as the proceeding of ECO iterations. The goal is to achieve a balanced optimization across timing, area, and power while adhering to a total iteration budget.""",
"""**Report Content**:
- CURRENT DESIGN STATE: The current design state includes timing, power, and area metrics after the most recent optimization iteration.
- OPTIMIZATION HISTORY: The optimization history includes the design states and already performed optimization commands from all previous iterations.
- UNFIXABLE ISSUES HISTORY: The unfixable issues history includes the design states and reasons for unfixable violations from all previous iterations.
- OBJECTIVES: The objectives describe optimization priorities for this run.
""",
"""
**Task**: Based on your analysis, generate the following content (Do not repeat unfixable reasons in your evaluation with detailed values.):
1. Your evaluation on unfixable reasons of timing, power, area from previous iterations. If at iteration 0 without unfixable reasons, you can directly reply "N/A" withour other contents.
2. Generate exactly 1 query to retrieve relevant knowledge to assist in decision-making. Focus on:
    - Optimization option strategy: What are the optimization option selection strategies based on your evaluation of unfixable reasons?
    - Do not repeat any detailed values in your query.
    - Focus on a general strategy (exploration or exploitation) withtou too much details.
    (Some possible query focuses: "New unfixable reasons are emerging", "Previous unfixable reasons are diminishing", "There is no enough report on unfixable reasons", etc.) You can combine them, think of new ones and make your own query.
"""
"""**Evaluation Content**:
1. Evaluate the optimization benefit on each target (power/area/timing) and the emerged unfixable reasons.
2. Based on the evaluation, correlate the optimization target with the emergence of new unfixable reasons and fixed unfixable reasons, i.e., how does the target value and unfixable reasons change across iterations.
"""
"**Evaluation Guidelines**:",
"1. You will receive evaluation from an ECO expert on the optimization trend.",
"2. Only make evaluation on given data, do not make any conjecture and comments on optimization strategy or priorities.",
"3. For timing, you should have seperate evaluation for setup and hold violations.",
"4. Based on the evaluation, correlate the optimization trend with the emergence of new unfixable reasons and fixed unfixable reasons.",
"5. Do not mention any detailed optimization actions like gate sizing."
"",
"""Definitions for Unfixable Reasons for Timing:
A - There are available library cells outside area limit.       B - Delay improvement is too small to fix the violation.        C - The violation is in clock tree.  D - Cell or net is located in high density area.        I - Buffer insertion with given library cells cannot fix the violation. L - Available physical area limits the use of one or more library cells.        O - No open free site is available.     R - No locations are available in parasitics or location transformation failed. S - Cell sizing with alternative library cells cannot fix the violation.        T - Timing margin is too tight to fix the violation.    W - Fixing the violation might degrade DRC violations.
""",
"""Definitions for Unfixable Reasons for Power/Area:
B - Benefit from sizing cell is too small       L - Physical constraints restrict sizing        S - Cell has no alternate library cell with better power        T - Sizing cell might degrade timing    W - Sizing cell might degrade DRC       X - Cell is unusable for ECO    Z - Cell is sized
"""

"**RESPONSE FORMAT**: Always respond in valid JSON format with evaluation and query keys.",
    ]


    objectives = state["objectives"]
    user_prompt_parts = []
    user_prompt_parts.append("**OBJECTIVES:**")
    user_prompt_parts.append(objectives)
    user_prompt_parts.append("")
    # Add current design state to system prompt
    current_hist = design_state_history[-1] if design_state_history else None
    if current_hist:
        current_state = current_hist['design_state']
        user_prompt_parts.append("**CURRENT DESIGN STATE:**")
        user_prompt_parts.append(f"Iteration: {current_hist['iteration']}")
        user_prompt_parts.append(f"  - Setup violations: {current_state['timing']['setup_violating_paths']} paths")
        user_prompt_parts.append(f"  - Hold violations: {current_state['timing']['hold_violating_paths']} paths")
        user_prompt_parts.append(f"  - Setup critical slack: {current_state['timing']['setup_critical_path_slack']}")
        user_prompt_parts.append(f"  - Hold critical slack: {current_state['timing']['hold_critical_path_slack']}")
        user_prompt_parts.append(f"  - Setup total negative slack: {current_state['timing']['setup_total_negative_slack']}")
        user_prompt_parts.append(f"  - Hold total negative slack: {current_state['timing']['hold_total_negative_slack']}")
        user_prompt_parts.append(f"  - Total power: {current_state['power']['total_power']:.3e}W")
        user_prompt_parts.append(f"  - Dynamic power: {current_state['power']['dynamic_power']:.3e}W")
        user_prompt_parts.append(f"  - Leakage power: {current_state['power']['leakage_power']:.3e}W")
        user_prompt_parts.append(f"  - Design area: {current_state['area']['design_area']} um^2")
        user_prompt_parts.append("")

    # Add optimization history to system prompt (only previous iterations, not current)
    if len(design_state_history) > 1:
        user_prompt_parts.append("**OPTIMIZATION HISTORY:**\n")
        user_prompt_parts.append("Optimization history listed in chronological order.")

        # Collect all historical data into lists for compact table-like display
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
        hist_remaining_budgets = []
        hist_execution_times = []
        hist_executed_commands = []

        for hist_entry in design_state_history[:-1]:  # Exclude current iteration (already shown above)
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
            hist_remaining_budgets.append(f"{hist_entry['remaining_budget']:.0f}s")
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
            f"- Dynamic power: {hist_dynamic_powers}W",
            f"- Leakage power: {hist_leakage_powers}W",
            f"- Design area: {hist_design_areas} um^2",
            f"- Executed commands: {hist_executed_commands}"
        ]
        user_prompt_parts.append("\t".join(hist_parts))

    # Add unfixable history to system prompt
    if unfixable_analysis_history:
        user_prompt_parts.append("**UNFIXABLE REASON HISTORY:**")
        if TOOL_USING:
            user_prompt_parts.append("Explanation for reason_distribution: The unfixable reasons are weighted by the negative slack they causes. Explanation for slack_distribution: The number of paths in each negative slack range.\n")
        for unfixable_entry in unfixable_analysis_history:
            unfixable_iteration = unfixable_entry['iteration']
            unfixable_data = unfixable_entry['unfixable_reasons']
            # Use json.dumps and escape curly braces for f-string
            import json
            unfixable_str = json.dumps(unfixable_data).replace('{', '{{').replace('}', '}}')
            user_prompt_parts.append(f"Iteration {unfixable_iteration}: {unfixable_str}")
        user_prompt_parts.append("")



    user_prompt_parts.extend([
        "Evaluation from ECO expert:",
        evaluation,
        "",
        ])
    # Build user prompt with current iteration data
    user_prompt_parts.extend([
        f"\n**Now You Can generate the response with JSON resonse:**",
        f"{{{{",
        f'  "evaluation": "Your evaluation here, no more than 200 words.",',
        f'  "query": "query on optimization option selection, use plain text, no more than 40 words."',
        f"}}}}"
    ])

    # Create fresh message list for final decision (no conversation history)
    final_messages = [
        SystemMessage(content="\n".join(system_prompt_parts)),
        HumanMessage(content="\n".join(user_prompt_parts))
    ]

    # LLM Call #2: Final analysis
    llm_start_time = time.time()
    messages_sent = [{"role": msg.type, "content": msg.content} for msg in final_messages]

    # Invoke LLM with fresh prompt
    response = llm.invoke(final_messages)

    # Extract content from response (handles both string and list formats)
    content_text = extract_content_from_llm_response(response.content)
    token_usage = {
        "input_tokens": response.usage_metadata["input_tokens"],
        "output_tokens": response.usage_metadata["output_tokens"]
    }

    # Parse JSON response - expecting: {"selected_command_type": "...", "optimization_strategy": "...", "timeout": ..., "reasoning": "..."}
    json_content = extract_json_from_thinking_response(content_text)
    parser = JsonOutputParser()
    llm_analysis = parser.parse(json_content)

    # Extract routing information
    queries = llm_analysis['query']
    evaluation = llm_analysis['evaluation']

    processing_time = time.time() - llm_start_time

    # Log final analysis interaction
    interaction = LLMInteraction(
        agent_type="SummaryAgent_FinalAnalysis",
        timestamp=datetime.now(),
        iteration=iteration,
        round_index=round_index,
        messages_sent=messages_sent,
        response_received={
            "content": content_text,
            "type": "json",
            "parsed_decision": {
                "evaluation": evaluation,
                "query": queries
            }
        },
        processing_time=processing_time,
        model_name=llm.model_name,
        token_usage=token_usage
    )
    logger.log_llm_interaction(interaction)

    return queries, evaluation, processing_time


def generate_final_decision(TOOL_USING: bool, rag_content: str, evaluation: str, long_term_reflection: str, iteration: int, round_index: int, state: Dict, llm, design_state_history: List, unfixable_analysis_history: List, rag_messages: List, logger):
    """LLM Call #3: Make final optimization decision using consolidated system prompt with optimization history and retrieved knowledge"""
    from utils import extract_content_from_llm_response, extract_json_from_thinking_response
    from agent_logs import LLMInteraction
    from langchain_core.messages import SystemMessage, HumanMessage

    # Build comprehensive system prompt with role, history, and retrieved knowledge
    system_prompt_parts = [
        """You are an expert IC design ECO (Engineering Change Order) optimization engineer responsible for optimization scheduling.

**TASK**: You will work as a scheduler to select the optimization target and optimization option strategy for the current ECO iteration based on (with Priority):
1. The evaluation on optimization trends and unfixable reasons, 
2. The refelection over strategy.
3. The template target/option selection strategy (you can refer to them but override them with your situation).
You need to schedule in following way:
1. Select the optimization target for current iteration (timing/power/area) and choose the optimization selection strategy (Exploration/Exploitation).
2. Summarize your analysis process for final decision making.

**ECO Background**: ECO is an incremental design optimization process that iteratively improves the design by fixing violations and optimizing metrics such as timing, area, and power. Each iteration involves analyzing the current design state, selecting an optimization target, generating and executing ECO commands, and evaluating the results. The goal is to achieve a balanced optimization across timing, area, and power while adhering to a total iteration budget.
As the proceeding of ECO iterations, the design gets optimized and the optimization benefit tend to decrease. The success of ECO lies in: 1. Extensively explore optimization target (timing, area, power) and optimization options for each target; 2. Exploit the optimization history and reports for unfixable reasons to select targeted optimization options or avoid those unfixable reasons. 3. Always keep the iteration budget in mind.""",
        "",

        """**Report Content**:
- CURRENT DESIGN STATE: The current design state includes timing, power, and area metrics after the most recent optimization iteration.
- OPTIMIZATION HISTORY: The optimization history includes the design states and already performed optimization commands from all previous iterations.
- UNFIXABLE ISSUES HISTORY: The unfixable issues history includes the design states and reasons for unfixable violations from all previous iterations.
- OBJECTIVES: The objectives describe optimization priorities for this run.
        """,

        "**GUIDELINES**:",
        "1. Please combine the evaluation and reference strategy for final decision.",
        "2. For detailed optimization trends and unfixable reasons that are not clear in the evaluation, you can refer to the detailed attatched report.",
        "3. Do not mention any detailed optimization actions like gate_sizing.",
        "4. Do not mention any detailed values of metrics or unfixable reasons.",
        "**RESPONSE FORMAT**: Always respond in valid JSON format."
    ]


    objectives = state["objectives"]
    user_prompt_parts = []
    user_prompt_parts.append("**OBJECTIVES:**")
    user_prompt_parts.append(objectives)
    user_prompt_parts.append("")
    # Add current design state to system prompt
    current_hist = design_state_history[-1] if design_state_history else None
    if current_hist:
        current_state = current_hist['design_state']
        user_prompt_parts.append("**CURRENT DESIGN STATE:**")
        user_prompt_parts.append(f"Iteration: {current_hist['iteration']}")
        user_prompt_parts.append(f"  - Setup violations: {current_state['timing']['setup_violating_paths']} paths")
        user_prompt_parts.append(f"  - Hold violations: {current_state['timing']['hold_violating_paths']} paths")
        user_prompt_parts.append(f"  - Setup critical slack: {current_state['timing']['setup_critical_path_slack']}")
        user_prompt_parts.append(f"  - Hold critical slack: {current_state['timing']['hold_critical_path_slack']}")
        user_prompt_parts.append(f"  - Setup total negative slack: {current_state['timing']['setup_total_negative_slack']}")
        user_prompt_parts.append(f"  - Hold total negative slack: {current_state['timing']['hold_total_negative_slack']}")
        user_prompt_parts.append(f"  - Total power: {current_state['power']['total_power']:.3e}W")
        user_prompt_parts.append(f"  - Dynamic power: {current_state['power']['dynamic_power']:.3e}W")
        user_prompt_parts.append(f"  - Leakage power: {current_state['power']['leakage_power']:.3e}W")
        user_prompt_parts.append(f"  - Design area: {current_state['area']['design_area']} um^2")
        user_prompt_parts.append("")

    # Add optimization history to system prompt (only previous iterations, not current)
    if len(design_state_history) > 1:
        user_prompt_parts.append("**OPTIMIZATION HISTORY:**")
        user_prompt_parts.append("Optimization history listed in chronological order.")

        # Collect all historical data into lists for compact table-like display
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
        hist_remaining_budgets = []
        hist_execution_times = []
        hist_executed_commands = []

        for hist_entry in design_state_history[:-1]:  # Exclude current iteration (already shown above)
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
            hist_remaining_budgets.append(f"{hist_entry['remaining_budget']:.0f}s")
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
            f"- Dynamic power: {hist_dynamic_powers}W",
            f"- Leakage power: {hist_leakage_powers}W",
            f"- Design area: {hist_design_areas} um^2",
            f"- Actual execution times: {hist_execution_times} s",
            f"- Executed commands: {hist_executed_commands}"
        ]
        user_prompt_parts.append("\t".join(hist_parts))

    # Add unfixable history to system prompt
    if unfixable_analysis_history:
        user_prompt_parts.append("**UNFIXABLE REASON HISTORY:**")
        if TOOL_USING:
            user_prompt_parts.append("Explanation for reason_distribution: The unfixable reasons are weighted by the negative slack they causes. Explanation for slack_distribution: The number of paths in each negative slack range.\n")
        for unfixable_entry in unfixable_analysis_history:
            unfixable_iteration = unfixable_entry['iteration']
            unfixable_data = unfixable_entry['unfixable_reasons']
            # Use json.dumps and escape curly braces for f-string
            import json
            unfixable_str = json.dumps(unfixable_data).replace('{', '{{').replace('}', '}}')
            user_prompt_parts.append(f"Iteration {unfixable_iteration}: {unfixable_str}")
        user_prompt_parts.append("")

    user_prompt_parts.extend([
        "Evaluation on ECO target and option selection:",
        evaluation,
        "Reflection on Strategy:",
        long_term_reflection,
        "Template Strategy on ECO target and option selection (Could be overidded):",
        rag_content,
        ])
    # Build user prompt with current iteration data
    user_prompt_parts.extend( [
        f"**CURRENT ITERATION:** {iteration}",
        "**RESPONSE FORMAT**: Always respond in valid JSON format:",
        "{{",
        '  "target": "timing|power|area",',
        '  "option": "Exploration|Exploitation",',
        '  "reasoning": "A short summary of the reason you choose this target and option",',
        "}}"
    ])

    # Create fresh message list for final decision (no conversation history)
    final_messages = [
        SystemMessage(content="\n".join(system_prompt_parts)),
        HumanMessage(content="\n".join(user_prompt_parts))
    ]

    # LLM Call #2: Final analysis
    llm_start_time = time.time()
    messages_sent = [{"role": msg.type, "content": msg.content} for msg in final_messages]

    # Invoke LLM with fresh prompt
    response = llm.invoke(final_messages)

    # Extract content from response (handles both string and list formats)
    content_text = extract_content_from_llm_response(response.content)

    # Parse JSON response - expecting: {"selected_command_type": "...", "optimization_strategy": "...", "timeout": ..., "reasoning": "..."}
    json_content = extract_json_from_thinking_response(content_text)
    parser = JsonOutputParser()
    llm_analysis = parser.parse(json_content)

    token_usage = {
        "input_tokens": response.usage_metadata["input_tokens"],
        "output_tokens": response.usage_metadata["output_tokens"]
    }

    # Extract routing information
    target = llm_analysis['target']
    option = llm_analysis['option']
    reasoning = llm_analysis['reasoning']

    processing_time = time.time() - llm_start_time

    # Log final analysis interaction
    interaction = LLMInteraction(
        agent_type="SummaryAgent_FinalAnalysis",
        timestamp=datetime.now(),
        iteration=iteration,
        round_index=round_index,
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
        model_name=llm.model_name,
        token_usage=token_usage
    )
    logger.log_llm_interaction(interaction)

    return llm_analysis, target, option, reasoning, processing_time
