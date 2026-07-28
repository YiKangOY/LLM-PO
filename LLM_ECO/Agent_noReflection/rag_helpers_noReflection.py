"""
RAG helper functions for ECO optimization system (no reflection prompts)
"""

import time
from datetime import datetime
import json

from langchain_core.output_parsers import JsonOutputParser

from rag_helpers import extract_current_state
from rag_helpers import extract_fixing_results as _extract_fixing_results
from rag_helpers import retrieve_knowledge
from rag_helpers import generate_selection_queries as _generate_selection_queries

CONFIDENTIAL_UNFIXABLE = "removed due to confidential reasons."


def extract_fixing_results(reports, state, iteration):
    fixing_results, _ = _extract_fixing_results(reports, state, iteration)
    return fixing_results, {"status": CONFIDENTIAL_UNFIXABLE}


def generate_selection_queries(TOOL_USING, current_state, fixing_results, unfixable_reasons, iteration, round_index, state, llm, design_state_history, unfixable_analysis_history, rag_messages, logger):
    sanitized_reasons = {"status": CONFIDENTIAL_UNFIXABLE}
    sanitized_history = []
    if unfixable_analysis_history:
        sanitized_history = [{"iteration": entry.get("iteration"), "unfixable_reasons": CONFIDENTIAL_UNFIXABLE} for entry in unfixable_analysis_history]
    return _generate_selection_queries(
        TOOL_USING,
        current_state,
        fixing_results,
        sanitized_reasons,
        iteration,
        round_index,
        state,
        llm,
        design_state_history,
        sanitized_history,
        rag_messages,
        logger,
    )


def generate_final_decision(TOOL_USING, rag_content, evaluation, long_term_reflection, iteration, round_index, state, llm, design_state_history, unfixable_analysis_history, rag_messages, logger):
    """LLM Call #3: Make final optimization decision without reflection context"""
    from utils import extract_content_from_llm_response
    from utils import extract_json_from_thinking_response
    from agent_logs import LLMInteraction
    from langchain_core.messages import SystemMessage
    from langchain_core.messages import HumanMessage

    system_prompt_parts = [
        """You are an expert IC design ECO (Engineering Change Order) optimization engineer responsible for optimization scheduling.

**TASK**: You will work as a scheduler to select the optimization target and optimization option strategy for the current ECO iteration based on (with Priority):
1. The evaluation on optimization trends. Unfixable reasons: removed due to confidential reasons.
2. The template target/option selection strategy (you can refer to them but override them with your situation).
You need to schedule in following way:
1. Select the optimization target for current iteration (timing/power/area) and choose the optimization selection strategy (Exploration/Exploitation).
2. Summarize your analysis process for final decision making.

**ECO Background**: ECO is an incremental design optimization process that iteratively improves the design by fixing violations and optimizing metrics such as timing, area, and power. Each iteration involves analyzing the current design state, selecting an optimization target, generating and executing ECO commands, and evaluating the results. The goal is to achieve a balanced optimization across timing, area, and power while adhering to a total iteration budget.
As the proceeding of ECO iterations, the design gets optimized and the optimization benefit tend to decrease. The success of ECO lies in: 1. Extensively explore optimization target (timing, area, power) and optimization options for each target; 2. Exploit the optimization history and reports to select targeted optimization options. 3. Always keep the iteration budget in mind.""",
        "",
        """**Report Content**:
- CURRENT DESIGN STATE: The current design state includes timing, power, and area metrics after the most recent optimization iteration.
- OPTIMIZATION HISTORY: The optimization history includes the design states and already performed optimization commands from all previous iterations.
- UNFIXABLE ISSUES HISTORY: removed due to confidential reasons.
- OBJECTIVES: The objectives describe optimization priorities for this run.
        """,
        "**GUIDELINES**:",
        "1. Please combine the evaluation and reference strategy for final decision.",
        "2. For detailed optimization trends that are not clear in the evaluation, you can refer to the detailed attached report.",
        "3. Do not mention any detailed optimization actions like gate_sizing.",
        "4. Do not mention any detailed values of metrics. Unfixable reasons are removed due to confidential reasons.",
        "**RESPONSE FORMAT**: Always respond in valid JSON format."
    ]

    objectives = state["objectives"]
    user_prompt_parts = []
    user_prompt_parts.append("**OBJECTIVES:**")
    user_prompt_parts.append(objectives)
    user_prompt_parts.append("")

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

    if len(design_state_history) > 1:
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
            hist_total_powers.append(f"{hist_state['power']['total_power']:.3e}")
            hist_dynamic_powers.append(f"{hist_state['power']['dynamic_power']:.3e}")
            hist_leakage_powers.append(f"{hist_state['power']['leakage_power']:.3e}")
            hist_design_areas.append(str(hist_state['area']['design_area']))
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

    if unfixable_analysis_history:
        user_prompt_parts.append("**UNFIXABLE REASON HISTORY:** removed due to confidential reasons.")
        user_prompt_parts.append("")
    user_prompt_parts.extend([
        "Evaluation on ECO target and option selection:",
        evaluation,
        "Template Strategy on ECO target and option selection (Could be overidded):",
        rag_content,
    ])

    user_prompt_parts.extend([
        f"**CURRENT ITERATION:** {iteration}",
        "**RESPONSE FORMAT**: Always respond in valid JSON format:",
        "{{",
        '  "target": "timing|power|area",',
        '  "option": "Exploration|Exploitation",',
        '  "reasoning": "A short summary of the reason you choose this target and option",',
        "}}"
    ])

    final_messages = [
        SystemMessage(content="\n".join(system_prompt_parts)),
        HumanMessage(content="\n".join(user_prompt_parts))
    ]

    llm_start_time = time.time()
    messages_sent = [{"role": msg.type, "content": msg.content} for msg in final_messages]

    response = llm.invoke(final_messages)

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

    processing_time = time.time() - llm_start_time

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
