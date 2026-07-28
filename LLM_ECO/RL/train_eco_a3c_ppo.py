#!/usr/bin/env python3
"""
Entry point to train/test the ECO PPO implementation.
This mirrors `train_eco_a3c.py` but forces `algo=ppo`.
"""

import argparse
import os
import re

from eco_a3c import a3c, continue_inference_all_models, inference_all_models
from eco_a3c_config import default_eco_a3c_config
from design_configs import DEFAULT_DESIGN_NAME, list_design_names
from rl_config import ENV_CONFIG
from utils.utils import remove_suffix


def _append_suffix_before_extension(path, suffix):
    root, ext = os.path.splitext(path)
    return "{}_{}{}".format(root, suffix, ext)


def _find_latest_checkpoint(model_dir, base_name):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError("Model directory does not exist: {}".format(model_dir))
    pattern = re.compile(r"^{}_episode_(\d+)\.pt$".format(re.escape(base_name)))
    last_episode = None
    last_path = ""
    for filename in os.listdir(model_dir):
        match = pattern.match(filename)
        if match:
            episode = int(match.group(1))
            if last_episode is None or episode > last_episode:
                last_episode = episode
                last_path = os.path.join(model_dir, filename)
    return last_episode, last_path


def main():
    parser = argparse.ArgumentParser(description="Train/Test ECO PPO agent")
    parser.add_argument(
        "--mode",
        choices=["train", "test", "inference", "continue-inference"],
        default="train",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override max episodes")
    parser.add_argument("--use-cuda", action="store_true", help="Enable CUDA if available")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device id (default: 0)")
    parser.add_argument(
        "--log-runtime-breakdown",
        action="store_true",
        help="Log per-episode runtime breakdowns for EDA tool, RL, and other time"
    )
    parser.add_argument(
        "--continue",
        dest="resume",
        action="store_true",
        help="Resume training from the latest checkpoint; use --mode continue-inference for inference"
    )
    parser.add_argument(
        "--design",
        choices=list_design_names(),
        default=DEFAULT_DESIGN_NAME,
        help="Design name for config overrides"
    )
    args = parser.parse_args()

    cfg = default_eco_a3c_config(design_name=args.design)
    cfg["algo"]["mode"] = args.mode
    cfg["algo"]["algo"] = "ppo"
    cfg["log-path"] = _append_suffix_before_extension(cfg["log-path"], "ppo")
    if args.episodes is not None:
        cfg["algo"]["max-episode"] = args.episodes
    cfg["algo"]["use-cuda"] = args.use_cuda
    cfg["algo"]["gpu-id"] = args.gpu_id
    cfg["algo"]["train"]["eval-frequency"] = 10
    cfg["algo"]["train"]["log-runtime-breakdown"] = args.log_runtime_breakdown
    cfg["algo"]["train"]["save-interval"] = 1
    if args.resume and args.mode != "train":
        raise ValueError(
            "--continue is only for --mode train; use --mode continue-inference "
            "to resume inference."
        )
    if args.resume:
        base_name = remove_suffix(os.path.basename(cfg["log-path"]), ".log")
        model_dir = cfg["model-path"]
        last_episode, resume_path = _find_latest_checkpoint(model_dir, base_name)
        if last_episode is None:
            fallback_dir = os.path.join(ENV_CONFIG["base_path"], "models")
            if fallback_dir != model_dir and os.path.isdir(fallback_dir):
                last_episode, resume_path = _find_latest_checkpoint(fallback_dir, base_name)
                if last_episode is not None:
                    cfg["model-path"] = fallback_dir
                    model_dir = fallback_dir
        if last_episode is None:
            raise FileNotFoundError(
                "No checkpoints found under {} for base name '{}'.".format(
                    model_dir, base_name
                )
            )
        print(
            "Resuming from checkpoint: {} (episode {})".format(
                resume_path, last_episode
            )
        )
        resume_episode = last_episode + 1
        if resume_episode >= cfg["algo"]["max-episode"]:
            raise ValueError(
                "Resume episode {} exceeds max-episode {}.".format(
                    resume_episode, cfg["algo"]["max-episode"]
                )
            )
        remaining_episodes = cfg["algo"]["max-episode"] - resume_episode
        print(
            "Remaining episodes: {} (max-episode {}, start {})".format(
                remaining_episodes, cfg["algo"]["max-episode"], resume_episode
            )
        )
        cfg["algo"]["train"]["resume"] = True
        cfg["algo"]["train"]["resume-episode"] = resume_episode
        cfg["algo"]["train"]["resume-path"] = resume_path

    if args.mode == "inference":
        inference_all_models(cfg)
    elif args.mode == "continue-inference":
        continue_inference_all_models(cfg)
    else:
        a3c(cfg)


if __name__ == "__main__":
    main()
