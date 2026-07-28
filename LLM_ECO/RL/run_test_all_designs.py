#!/usr/bin/env python3
"""
Generate a nohup bash script to run test mode for all designs in parallel.
"""

import os

from design_configs import DESIGN_CONFIG_OVERRIDES, list_design_names


def _write_nohup_script(output_path, python_path):
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    for index, design_name in enumerate(list_design_names()):
        gpu_id = index % 8
        base_path = DESIGN_CONFIG_OVERRIDES[design_name]["environment"]["base_path"]
        log_file = "{}/rl_test_{}.log".format(base_path, design_name)
        lines.append(
            "nohup python {} --design {} --use-cuda --gpu-id {} > {} 2>&1 &".format(
                python_path, design_name, gpu_id, log_file
            )
        )
        lines.append(
            "echo \"Started {} on GPU {} -> {}\"".format(
                design_name, gpu_id, log_file
            )
        )
    lines.append("")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    script_dir = os.path.abspath(os.path.dirname(__file__))
    output_path = os.path.join(script_dir, "run_test_all_designs_nohup.sh")
    python_path = os.path.join(script_dir, "run_test_single_design.py")
    _write_nohup_script(output_path, python_path)
    print("Wrote {}".format(output_path))


if __name__ == "__main__":
    main()
