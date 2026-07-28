#!/usr/bin/env python3
"""
Run train_eco_a3c.py in test mode for a single design, using the latest model.
"""

import argparse
import os
import sys

import train_eco_a3c
from design_configs import DESIGN_CONFIG_OVERRIDES, list_design_names


def _model_timestamp(path):
    stat_result = os.stat(path)
    if hasattr(stat_result, "st_birthtime"):
        return stat_result.st_birthtime
    return stat_result.st_ctime


def _find_latest_model(model_dir):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError("Model directory does not exist: {}".format(model_dir))
    latest_path = ""
    latest_time = None
    for filename in os.listdir(model_dir):
        if not filename.endswith(".pt"):
            continue
        path = os.path.join(model_dir, filename)
        timestamp = _model_timestamp(path)
        if latest_time is None or timestamp > latest_time:
            latest_time = timestamp
            latest_path = path
    if latest_time is None:
        raise FileNotFoundError("No .pt files found under {}".format(model_dir))
    return latest_path


def _set_latest_model_path(design_name):
    overrides = DESIGN_CONFIG_OVERRIDES[design_name]
    base_path = overrides["environment"]["base_path"]
    model_dir = os.path.join(base_path, "RL", "models")
    latest_model = _find_latest_model(model_dir)
    overrides["environment"]["model_path"] = latest_model
    return latest_model


def main():
    parser = argparse.ArgumentParser(
        description="Run ECO test for a single design with latest model."
    )
    parser.add_argument(
        "--design",
        choices=list_design_names(),
        required=True,
        help="Design name for config overrides"
    )
    parser.add_argument("--use-cuda", action="store_true", help="Enable CUDA")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device id")
    args = parser.parse_args()

    latest_model = _set_latest_model_path(args.design)
    print(
        "Testing design {} with model {} on GPU {}".format(
            args.design, latest_model, args.gpu_id
        )
    )

    script_path = os.path.abspath(train_eco_a3c.__file__)
    sys.argv = [script_path, "--mode", "test", "--design", args.design]
    if args.use_cuda:
        sys.argv.extend(["--use-cuda", "--gpu-id", str(args.gpu_id)])
    train_eco_a3c.main()


if __name__ == "__main__":
    main()
