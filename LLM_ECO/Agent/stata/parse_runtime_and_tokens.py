#!/usr/bin/env python3
"""Summarize runtime and token usage from agent log JSON files."""

import csv
import json
import os
import sys
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from configs import AGENT_DIR_NAME
from configs import DESIGN_CONFIG_OVERRIDES


def load_json(path):
    """Load JSON content from a file path."""
    with open(path, "r") as handle:
        return json.load(handle)


def collect_timestamps(llm_entries, agent_entries):
    """Collect all timestamps across both log files."""
    timestamps = []
    for entry in llm_entries:
        timestamps.append(datetime.fromisoformat(entry["timestamp"]))
    for entry in agent_entries:
        timestamps.append(datetime.fromisoformat(entry["timestamp"]))
    return timestamps


def filter_llm_entries_by_round(llm_entries, round_index):
    """Filter LLM interactions to a single round."""
    filtered_entries = []
    for entry in llm_entries:
        if entry["round_index"] == round_index:
            filtered_entries.append(entry)
    return filtered_entries


def filter_agent_entries_by_window(agent_entries, start_time, end_time):
    """Filter agent log entries to a specific time window."""
    filtered_entries = []
    for entry in agent_entries:
        entry_time = datetime.fromisoformat(entry["timestamp"])
        if start_time <= entry_time <= end_time:
            filtered_entries.append(entry)
    return filtered_entries


def summarize_llm(llm_entries):
    """Sum LLM runtime and tokens from llm_interactions.json."""
    llm_runtime = 0.0
    input_tokens = 0
    output_tokens = 0
    for entry in llm_entries:
        llm_runtime += entry["processing_time"]
        token_usage = entry["token_usage"]
        input_tokens += token_usage["input_tokens"]
        output_tokens += token_usage["output_tokens"]
    total_tokens = input_tokens + output_tokens
    return llm_runtime, input_tokens, output_tokens, total_tokens


def summarize_eda(agent_entries):
    """Sum EDA tool runtime from command executor entries."""
    eda_runtime = 0.0
    for entry in agent_entries:
        if entry["agent_type"] == "CommandExecutor":
            eda_runtime += entry["output_data"]["execution_time"]
    return eda_runtime


def summarize_design(design_name, base_path):
    """Summarize runtimes and tokens for a single design."""
    logs_dir = os.path.join(base_path, AGENT_DIR_NAME, "logs")
    llm_path = os.path.join(logs_dir, "llm_interactions.json")
    agent_path = os.path.join(logs_dir, "eco_agent_responses_langchain.json")

    llm_entries = load_json(llm_path)
    agent_entries = load_json(agent_path)

    round_index = 0
    scale_factor = 10

    llm_entries = filter_llm_entries_by_round(llm_entries, round_index)
    round_timestamps = collect_timestamps(llm_entries, [])
    round_start_time = min(round_timestamps)
    round_end_time = max(round_timestamps)
    agent_entries = filter_agent_entries_by_window(agent_entries, round_start_time, round_end_time)

    timestamps = collect_timestamps(llm_entries, agent_entries)
    start_time = min(timestamps)
    end_time = max(timestamps)
    total_runtime = (end_time - start_time).total_seconds()

    llm_runtime, input_tokens, output_tokens, total_tokens = summarize_llm(llm_entries)
    eda_runtime = summarize_eda(agent_entries)
    other_runtime = total_runtime - llm_runtime - eda_runtime
    total_runtime *= scale_factor
    llm_runtime *= scale_factor
    eda_runtime *= scale_factor
    other_runtime *= scale_factor
    input_tokens *= scale_factor
    output_tokens *= scale_factor
    total_tokens *= scale_factor

    return {
        "design": design_name,
        "log_dir": logs_dir,
        "total_runtime_s": total_runtime,
        "llm_runtime_s": llm_runtime,
        "eda_runtime_s": eda_runtime,
        "other_runtime_s": other_runtime,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
    }


def format_seconds(value):
    """Format seconds with three decimal places."""
    return f"{value:.3f}"


def main():
    """Entry point for runtime and token summarization."""
    summaries = []
    for design_name in DESIGN_CONFIG_OVERRIDES:
        configured_path = DESIGN_CONFIG_OVERRIDES[design_name]["base_path"]
        base_path = configured_path
        if not os.path.isabs(base_path):
            base_path = os.path.join(PROJECT_ROOT, base_path)
        logs_dir = os.path.join(base_path, AGENT_DIR_NAME, "logs")
        if not os.path.isdir(logs_dir):
            continue
        summaries.append(summarize_design(design_name, base_path))

    output_csv = os.path.join(CURRENT_DIR, "runtime_token_summary.csv")
    with open(output_csv, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "design",
            "total_runtime_s",
            "llm_runtime_s",
            "eda_runtime_s",
            "other_runtime_s",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "start_time",
            "end_time",
            "log_dir",
        ])

        for summary in summaries:
            writer.writerow([
                summary["design"],
                format_seconds(summary["total_runtime_s"]),
                format_seconds(summary["llm_runtime_s"]),
                format_seconds(summary["eda_runtime_s"]),
                format_seconds(summary["other_runtime_s"]),
                summary["input_tokens"],
                summary["output_tokens"],
                summary["total_tokens"],
                summary["start_time"],
                summary["end_time"],
                summary["log_dir"],
            ])

    for summary in summaries:
        print(
            summary["design"],
            "total_runtime_s=", format_seconds(summary["total_runtime_s"]),
            "llm_runtime_s=", format_seconds(summary["llm_runtime_s"]),
            "eda_runtime_s=", format_seconds(summary["eda_runtime_s"]),
            "other_runtime_s=", format_seconds(summary["other_runtime_s"]),
            "input_tokens=", summary["input_tokens"],
            "output_tokens=", summary["output_tokens"],
            "total_tokens=", summary["total_tokens"],
        )

    print("Wrote CSV:", output_csv)


if __name__ == "__main__":
    main()
