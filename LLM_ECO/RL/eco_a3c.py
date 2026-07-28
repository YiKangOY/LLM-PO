#!/usr/bin/env python3
"""
Lightweight A3C training loop for the ECO task that mirrors the structure
of the reference `dse.algo.a3c.a3c` implementation.
"""

import copy
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)
_AGENT_ROOT = os.path.join(_ROOT, "Agent")
if _AGENT_ROOT not in sys.path:
    sys.path.append(_AGENT_ROOT)

from eco_env import ECOEnv
from eco_agent import ECOAgent
from rl_config import ENV_CONFIG, RL_CONFIG, load_config
from utils.utils import remove_suffix


class EpisodeMetricsLogger:
    """
    Append per-episode reward breakdowns to a CSV for quick visualization.
    """

    fieldnames = (
        "episode",
        "avg_reward",
        "avg_setup_tns",
        "avg_hold_tns",
        "avg_area",
        "avg_power",
    )

    def __init__(self, csv_path):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, episode, averages):
        row = {
            "episode": episode,
            "avg_reward": averages["reward"],
            "avg_setup_tns": averages["setup_tns_reward"],
            "avg_hold_tns": averages["hold_tns_reward"],
            "avg_area": averages["area_reward"],
            "avg_power": averages["power_reward"],
        }
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


class InferenceMetricsLogger:
    """
    Append per-checkpoint inference reward breakdowns to a CSV.
    """

    fieldnames = (
        "model_name",
        "model_path",
        "avg_reward",
        "avg_setup_tns",
        "avg_hold_tns",
        "avg_area",
        "avg_power",
    )

    def __init__(self, csv_path):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, model_name, model_path, averages):
        row = {
            "model_name": model_name,
            "model_path": model_path,
            "avg_reward": averages["reward"],
            "avg_setup_tns": averages["setup_tns_reward"],
            "avg_hold_tns": averages["hold_tns_reward"],
            "avg_area": averages["area_reward"],
            "avg_power": averages["power_reward"],
        }
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


class TrainingRunLogger:
    """
    Append structured per-episode events to a JSONL log for later analysis.
    """

    def __init__(self, log_dir, base_name):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{base_name}_{timestamp}.log")
        self._fh = open(self.log_path, "a", encoding="utf-8")

    def log_event(self, event, payload):
        record = {
            "event": event,
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        record.update(payload)
        self._fh.write(json.dumps(record, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


def _append_suffix_before_extension(filename, suffix):
    root, ext = os.path.splitext(filename)
    return "{}_{}{}".format(root, suffix, ext)


def _extract_reward_metrics(metrics_snapshot):
    timing = metrics_snapshot["timing"]
    setup = timing["setup"]
    hold = timing["hold"]
    area = metrics_snapshot["area"]
    power = metrics_snapshot["power"]
    return {
        "setup_wns": float(setup["wns"]),
        "setup_tns": float(setup["tns"]),
        "hold_wns": float(hold["wns"]),
        "hold_tns": float(hold["tns"]),
        "area": float(area["design_area"]),
        "power": {
            "internal": float(power["internal"]),
            "leakage": float(power["leakage"]),
            "switching": float(power["switching"]),
            "total": float(power["total"]),
        },
    }


def _get_max_runtime_value(runtime_stats, key):
    if not runtime_stats:
        return 0.0
    return max(float(stats[key]) for stats in runtime_stats)


def _build_episode_runtime_payload(
    episode,
    runtime_stats,
    episode_wall_time_s,
    rl_training_inference_runtime_s,
    log_runtime_breakdown,
):
    if not log_runtime_breakdown:
        return {
            "episode": episode,
            "runtime_s": _get_max_runtime_value(runtime_stats, "elapsed_runtime_s"),
        }

    eda_tool_runtime_s = _get_max_runtime_value(runtime_stats, "tool_time_s")
    other_runtime_s = (
        episode_wall_time_s
        - eda_tool_runtime_s
        - rl_training_inference_runtime_s
    )
    if other_runtime_s < 0.0:
        other_runtime_s = 0.0

    return {
        "episode": episode,
        "runtime_s": episode_wall_time_s,
        "eda_tool_runtime_s": eda_tool_runtime_s,
        "rl_training_inference_runtime_s": rl_training_inference_runtime_s,
        "other_runtime_s": other_runtime_s,
    }


def _merge_terminal_episode_metrics(envs, terminal_episode_metrics):
    episode_metrics = envs.env_method("get_episode_metrics")
    for idx, metrics in enumerate(terminal_episode_metrics):
        if metrics is not None:
            episode_metrics[idx] = metrics
    return episode_metrics


def _checkpoint_sort_key(model_path):
    filename = os.path.basename(model_path)
    match = re.search(r"_episode_(\d+)\.pt$", filename)
    if match:
        return (0, int(match.group(1)), filename)
    return (1, 0, filename)


def _checkpoint_epoch(model_path):
    filename = os.path.basename(model_path)
    match = re.search(r"_episode_(\d+)\.pt$", filename)
    if match:
        return int(match.group(1))
    return None


def _find_model_checkpoints(model_dir, exclude_ppo=False):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError("Model directory does not exist: {}".format(model_dir))
    model_paths = []
    for filename in os.listdir(model_dir):
        if exclude_ppo and "ppo" in filename:
            continue
        if filename.endswith(".pt"):
            model_paths.append(os.path.join(model_dir, filename))
    if len(model_paths) == 0:
        raise FileNotFoundError("No .pt model checkpoints found under {}".format(model_dir))
    model_paths.sort(key=_checkpoint_sort_key)
    return model_paths


def _find_latest_inference_log(log_dir, log_basename):
    if not os.path.isdir(log_dir):
        return ""
    prefix = "{}_inference_".format(log_basename)
    latest_path = ""
    latest_mtime = None
    for filename in os.listdir(log_dir):
        if filename.startswith(prefix) and filename.endswith(".log"):
            log_path = os.path.join(log_dir, filename)
            log_mtime = os.path.getmtime(log_path)
            if latest_mtime is None or log_mtime > latest_mtime:
                latest_mtime = log_mtime
                latest_path = log_path
    return latest_path


def _find_inference_logs(log_dir, log_basename):
    if not os.path.isdir(log_dir):
        return []
    prefix = "{}_inference_".format(log_basename)
    log_paths = []
    for filename in os.listdir(log_dir):
        if filename.startswith(prefix) and filename.endswith(".log"):
            log_paths.append(os.path.join(log_dir, filename))
    log_paths.sort(key=os.path.getmtime, reverse=True)
    return log_paths


def _get_last_inferenced_model_index(log_path):
    last_model_index = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            record = json.loads(stripped_line)
            if record["event"] == "episode_reward":
                last_model_index = int(record["model_index"])
    return last_model_index


def _get_inference_start_index(log_dir, log_basename):
    log_paths = _find_inference_logs(log_dir, log_basename)
    if len(log_paths) == 0:
        return 0, ""
    for log_path in log_paths:
        last_model_index = _get_last_inferenced_model_index(log_path)
        if last_model_index is not None:
            return last_model_index + 1, log_path
    return 0, log_paths[0]


def _safe_run_dir_token(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def train_a3c_impl(agent, fixed_w, episode):
    updated_w = agent.preference.generate_preference(agent.sample_size, fixed_w)
    total_updated_w = updated_w.repeat(agent.num_parallel * agent.num_step, axis=0)
    agent.buffer.generate_batch_with_n_step()
    value, next_value, policy = agent.forward_transition(total_updated_w)
    discounted_reward = agent.calc_discounted_reward(value, next_value)
    adv, discounted_reward = agent.envelope_operator(updated_w, discounted_reward, value, episode)
    agent.optimize_actor_critic(total_updated_w, discounted_reward, adv)
    agent.schedule_lr(episode)
    agent.save(episode)
    agent.sync_critic(episode)


def train_ppo_impl(agent, fixed_w, episode):
    updated_w = agent.preference.generate_preference(agent.sample_size, fixed_w)
    total_updated_w = updated_w.repeat(agent.num_parallel * agent.num_step, axis=0)
    agent.buffer.generate_batch_with_n_step()
    value, next_value, _ = agent.forward_transition(total_updated_w)
    discounted_reward = agent.calc_discounted_reward(value, next_value)
    adv, discounted_reward = agent.envelope_operator(updated_w, discounted_reward, value, episode)
    agent.buffer.attach_postprocess(advantage=adv, returns=discounted_reward)
    agent.optimize_ppo(total_updated_w, discounted_reward, adv)
    agent.schedule_lr(episode)
    agent.save(episode)


def eval_a3c(
    agent,
    configs,
    eval_episode,
    metrics_logger,
    run_logger,
    run_dir_override=None,
    event_payload=None,
    model_path=None,
):
    eval_configs = copy.deepcopy(configs)
    eval_configs["algo"]["mode"] = "train"
    eval_configs["algo"]["num-parallel"] = 1
    eval_configs["env"]["sim"]["idx"] = 0
    if run_dir_override is None:
        eval_configs["run_dir_override"] = "run_eval_{}_{}".format(
            eval_configs["algo"]["algo"], eval_episode
        )
    else:
        eval_configs["run_dir_override"] = run_dir_override

    eval_agent = ECOAgent(eval_configs, ECOEnv)
    if model_path is None:
        eval_agent.model.load_state_dict(agent.model.state_dict())
        eval_agent._model.load_state_dict(agent._model.state_dict())
    else:
        eval_agent.load(model_path)
    eval_agent.model.eval()
    eval_agent._model.eval()
    eval_agent.training = False

    preference = eval_agent.preference
    fixed_w = preference.init_preference()
    explore_w = preference.generate_preference(eval_agent.num_parallel, fixed_w)

    max_rounds = eval_configs["algo"]["test"]["max-search-round"]
    total_reward_sums = np.zeros(eval_agent.num_parallel, dtype=np.float64)
    total_component_sums = {
        "setup_tns_reward": np.zeros(eval_agent.num_parallel, dtype=np.float64),
        "hold_tns_reward": np.zeros(eval_agent.num_parallel, dtype=np.float64),
        "area_reward": np.zeros(eval_agent.num_parallel, dtype=np.float64),
        "power_reward": np.zeros(eval_agent.num_parallel, dtype=np.float64),
    }
    log_runtime_breakdown = configs["algo"]["train"]["log-runtime-breakdown"]

    for eval_round in range(max_rounds):
        episode_t0 = time.perf_counter()
        rl_training_inference_runtime_s = 0.0
        state = eval_agent.envs.reset()
        terminal_episode_metrics = [None] * eval_agent.num_parallel
        reward_sums = np.zeros(eval_agent.num_parallel, dtype=np.float64)
        component_sums = {
            "setup_tns_reward": np.zeros(eval_agent.num_parallel, dtype=np.float64),
            "hold_tns_reward": np.zeros(eval_agent.num_parallel, dtype=np.float64),
            "area_reward": np.zeros(eval_agent.num_parallel, dtype=np.float64),
            "power_reward": np.zeros(eval_agent.num_parallel, dtype=np.float64),
        }

        for _ in range(eval_agent.num_step):
            inference_t0 = time.perf_counter()
            action, _, _ = eval_agent.get_action(state, explore_w)
            rl_training_inference_runtime_s += time.perf_counter() - inference_t0
            action = np.asarray(action, dtype=np.int64).reshape(eval_agent.num_parallel)
            next_state, reward, done, info = eval_agent.envs.step(action)

            for i in range(1, eval_agent.num_parallel):
                if done[i]:
                    explore_w = preference.renew_preference(explore_w, i)
            state = next_state

            reward_arr = np.asarray(reward).reshape(eval_agent.num_parallel, -1)
            reward_sums += reward_arr[:, 0]

            if isinstance(info, (list, tuple)):
                for idx in range(min(eval_agent.num_parallel, len(info))):
                    terminal_metrics = info[idx]["episode_metrics"]
                    if terminal_metrics is not None:
                        terminal_episode_metrics[idx] = terminal_metrics
                    rc = info[idx]["reward_components"]
                    if rc is None:
                        continue
                    component_sums["setup_tns_reward"][idx] += float(rc["setup_tns_reward"])
                    component_sums["hold_tns_reward"][idx] += float(rc["hold_tns_reward"])
                    component_sums["area_reward"][idx] += float(rc["area_reward"])
                    component_sums["power_reward"][idx] += float(rc["power_reward"])

        total_reward_sums += reward_sums
        for key in total_component_sums:
            total_component_sums[key] += component_sums[key]

        episode_wall_time_s = time.perf_counter() - episode_t0
        episode_metrics = _merge_terminal_episode_metrics(
            eval_agent.envs,
            terminal_episode_metrics,
        )
        runtime_stats = [metrics["runtime_stats"] for metrics in episode_metrics]
        runtime_payload = _build_episode_runtime_payload(
            eval_episode,
            runtime_stats,
            episode_wall_time_s,
            rl_training_inference_runtime_s,
            log_runtime_breakdown,
        )
        run_logger.log_event(
            "episode_runtime",
            _merge_event_payload(runtime_payload, event_payload),
        )
        max_runtime_s = runtime_payload["runtime_s"]

        for env_id, metrics in enumerate(episode_metrics):
            initial_metrics = _extract_reward_metrics(metrics["initial_metrics"])
            final_metrics = _extract_reward_metrics(metrics["final_metrics"])
            env_runtime_s = float(
                runtime_stats[env_id]["elapsed_runtime_s"]
            ) if env_id < len(runtime_stats) else 0.0
            run_logger.log_event(
                "episode_reward",
                _merge_event_payload({
                    "episode": eval_episode,
                    "env_id": env_id,
                    "reward": float(reward_sums[env_id]),
                    "initial": initial_metrics,
                    "final": final_metrics,
                    "episode_runtime_s": max_runtime_s,
                    "env_runtime_s": env_runtime_s,
                }, event_payload),
            )

    avg_metrics = {
        "reward": float(np.mean(total_reward_sums) / max_rounds),
        "setup_tns_reward": float(np.mean(total_component_sums["setup_tns_reward"]) / max_rounds),
        "hold_tns_reward": float(np.mean(total_component_sums["hold_tns_reward"]) / max_rounds),
        "area_reward": float(np.mean(total_component_sums["area_reward"]) / max_rounds),
        "power_reward": float(np.mean(total_component_sums["power_reward"]) / max_rounds),
    }
    if metrics_logger is not None:
        metrics_logger.log(eval_episode, avg_metrics)
    eval_agent.envs.close()
    return avg_metrics


def _merge_event_payload(payload, extra_payload):
    if extra_payload is None:
        return payload
    merged_payload = dict(payload)
    merged_payload.update(extra_payload)
    return merged_payload


def inference_all_models(configs):
    _run_inference_models(configs, 0)


def continue_inference_all_models(configs):
    log_dir = os.path.dirname(configs["log-path"])
    log_basename = os.path.splitext(os.path.basename(configs["log-path"]))[0]
    start_model_index, latest_log_path = _get_inference_start_index(
        log_dir,
        log_basename,
    )
    if latest_log_path == "":
        print("No previous inference log found. Starting from model index 0.")
    else:
        print(
            "Continuing inference from model index {} after log {}".format(
                start_model_index,
                latest_log_path,
            )
        )
    _run_inference_models(configs, start_model_index)


def _run_inference_models(configs, start_model_index):
    log_dir = os.path.dirname(configs["log-path"])
    log_basename = os.path.splitext(os.path.basename(configs["log-path"]))[0]
    model_paths = _find_model_checkpoints(configs["model-path"], exclude_ppo=True)
    if start_model_index >= len(model_paths):
        print(
            "No remaining models to inference: start index {} with {} checkpoints.".format(
                start_model_index,
                len(model_paths),
            )
        )
        return

    inference_configs = copy.deepcopy(configs)
    inference_configs["algo"]["mode"] = "train"
    inference_configs["algo"]["num-parallel"] = 1
    inference_configs["env"]["sim"]["idx"] = 0

    metrics_logger = InferenceMetricsLogger(
        os.path.join(log_dir, "{}_metrics_inference.csv".format(log_basename))
    )
    run_logger = TrainingRunLogger(log_dir, "{}_inference".format(log_basename))

    for model_idx in range(start_model_index, len(model_paths)):
        model_path = model_paths[model_idx]
        model_name = os.path.basename(model_path)

        model_token = _safe_run_dir_token(remove_suffix(model_name, ".pt"))
        run_dir_override = "run_inference_{}_{}".format(
            inference_configs["algo"]["algo"],
            model_token,
        )
        run_dir_path = os.path.join(
            ENV_CONFIG["base_path"],
            "RL",
            "run_dir",
            run_dir_override,
        )
        event_payload = {
            "model_name": model_name,
            "model_path": model_path,
            "model_index": model_idx,
            "model_epoch": _checkpoint_epoch(model_path),
            "run_dir": run_dir_path,
            "run_dir_name": run_dir_override,
            "report_iteration": RL_CONFIG["max_iterations_per_episode"],
        }
        avg_metrics = eval_a3c(
            None,
            inference_configs,
            model_idx,
            None,
            run_logger,
            run_dir_override=run_dir_override,
            event_payload=event_payload,
            model_path=model_path,
        )
        metrics_logger.log(model_name, model_path, avg_metrics)

    run_logger.close()


def train_a3c(agent, configs):
    envs = agent.envs
    assert agent.num_step == envs.safe_get_attr("dims_of_tunable_state"), (
        f"num_step {agent.num_step} vs tunable state: {envs.safe_get_attr('dims_of_tunable_state')}"
    )
    if not ENV_CONFIG:
        load_config()
    base_path = ENV_CONFIG["base_path"]

    preference = agent.preference
    fixed_w = preference.init_preference()
    explore_w = preference.generate_preference(agent.num_parallel, fixed_w)

    log_dir = os.path.dirname(
        configs.get("log-path", os.path.join(base_path, "RL", "logs", "eco_a3c.log"))
    )
    log_basename = os.path.splitext(
        os.path.basename(configs.get("log-path", "eco_a3c.log"))
    )[0]
    metrics_logger = EpisodeMetricsLogger(
        os.path.join(log_dir, f"{log_basename}_metrics.csv")
    )
    run_logger = TrainingRunLogger(log_dir, f"{log_basename}_train")
    eval_frequency = configs["algo"]["train"]["eval-frequency"]
    eval_metrics_logger = None
    eval_run_logger = None
    if eval_frequency > 0:
        eval_metrics_name = _append_suffix_before_extension(
            f"{log_basename}_metrics.csv",
            "eval"
        )
        eval_metrics_logger = EpisodeMetricsLogger(os.path.join(log_dir, eval_metrics_name))
        eval_run_logger = TrainingRunLogger(log_dir, f"{log_basename}_eval")
    resume = configs["algo"]["train"]["resume"]
    resume_path = configs["algo"]["train"]["resume-path"]
    resume_episode = configs["algo"]["train"]["resume-episode"]
    log_runtime_breakdown = configs["algo"]["train"]["log-runtime-breakdown"]
    start_episode = 0
    if resume:
        if resume_episode < 0:
            raise ValueError("resume-episode must be >= 0.")
        agent.load(resume_path)
        start_episode = resume_episode
    remaining_episodes = agent.max_episode - start_episode
    if remaining_episodes <= 0:
        raise ValueError(
            "No remaining episodes to train: resume-episode {} with max-episode {}.".format(
                start_episode, agent.max_episode
            )
        )
    best_eval_reward = None

    for episode_idx in range(remaining_episodes):
        episode = start_episode + episode_idx
        episode_t0 = time.perf_counter()
        rl_training_inference_runtime_s = 0.0
        agent.buffer.reset()
        state = envs.reset()
        action_prob = []
        old_explore_w = explore_w
        terminal_episode_metrics = [None] * agent.num_parallel

        component_sums = {
            "setup_tns_reward": np.zeros(agent.num_parallel, dtype=np.float64),
            "hold_tns_reward": np.zeros(agent.num_parallel, dtype=np.float64),
            "area_reward": np.zeros(agent.num_parallel, dtype=np.float64),
            "power_reward": np.zeros(agent.num_parallel, dtype=np.float64),
        }
        reward_sums = np.zeros(agent.num_parallel, dtype=np.float64)

        for _ in range(agent.num_step):
            inference_t0 = time.perf_counter()
            action, policy, log_prob = agent.get_action(state, explore_w)
            rl_training_inference_runtime_s += time.perf_counter() - inference_t0
            # Keep actions as a flat int64 array so SubprocVecEnv pickling never
            # sees object dtypes (which trigger PicklingError).
            action = np.asarray(action, dtype=np.int64).reshape(agent.num_parallel)
            next_state, reward, done, info = envs.step(action)

            terminal_safe_next_state = np.stack([
                info[idx].get("terminal_observation", next_state[idx]) if done[idx] else next_state[idx]
                for idx in range(agent.num_parallel)
            ])
            agent.buffer.insert(state, action, terminal_safe_next_state, reward, done, log_prob)

            action_prob.append(policy)

            for i in range(1, agent.num_parallel):
                if done[i]:
                    explore_w = preference.renew_preference(explore_w, i)
            state = next_state

            reward_arr = np.asarray(reward).reshape(agent.num_parallel, -1)
            reward_sums += reward_arr[:, 0]

            if isinstance(info, (list, tuple)):
                for idx in range(min(agent.num_parallel, len(info))):
                    terminal_metrics = info[idx]["episode_metrics"]
                    if terminal_metrics is not None:
                        terminal_episode_metrics[idx] = terminal_metrics
                    rc = info[idx].get("reward_components")
                    if rc is None:
                        continue
                    if not isinstance(rc, dict):
                        raise TypeError(
                            f"reward_components must be a dict, got {type(rc).__name__}"
                        )
                    missing_keys = [
                        key for key in ("setup_tns_reward", "hold_tns_reward", "area_reward", "power_reward")
                        if key not in rc
                    ]
                    if missing_keys:
                        raise KeyError(
                            f"reward_components missing keys: {', '.join(missing_keys)}"
                        )
                    component_sums["setup_tns_reward"][idx] += float(rc["setup_tns_reward"])
                    component_sums["hold_tns_reward"][idx] += float(rc["hold_tns_reward"])
                    component_sums["area_reward"][idx] += float(rc["area_reward"])
                    component_sums["power_reward"][idx] += float(rc["power_reward"])

        train_t0 = time.perf_counter()
        agent.anneal()
        if agent.is_ppo:
            train_ppo_impl(agent, fixed_w, episode)
        else:
            train_a3c_impl(agent, fixed_w, episode)
        rl_training_inference_runtime_s += time.perf_counter() - train_t0

        avg_metrics = {
            "reward": float(np.mean(reward_sums)),
            "setup_tns_reward": float(np.mean(component_sums["setup_tns_reward"])),
            "hold_tns_reward": float(np.mean(component_sums["hold_tns_reward"])),
            "area_reward": float(np.mean(component_sums["area_reward"])),
            "power_reward": float(np.mean(component_sums["power_reward"])),
        }
        episode_wall_time_s = time.perf_counter() - episode_t0
        metrics_logger.log(episode, avg_metrics)

        episode_metrics = _merge_terminal_episode_metrics(
            envs,
            terminal_episode_metrics,
        )
        runtime_stats = [metrics["runtime_stats"] for metrics in episode_metrics]
        runtime_payload = _build_episode_runtime_payload(
            episode,
            runtime_stats,
            episode_wall_time_s,
            rl_training_inference_runtime_s,
            log_runtime_breakdown,
        )
        run_logger.log_event(
            "episode_runtime",
            runtime_payload,
        )
        max_runtime_s = runtime_payload["runtime_s"]

        for env_id, metrics in enumerate(episode_metrics):
            initial_metrics = _extract_reward_metrics(metrics["initial_metrics"])
            final_metrics = _extract_reward_metrics(metrics["final_metrics"])
            env_runtime_s = float(
                runtime_stats[env_id]["elapsed_runtime_s"]
            ) if env_id < len(runtime_stats) else 0.0
            run_logger.log_event(
                "episode_reward",
                {
                    "episode": episode,
                    "env_id": env_id,
                    "reward": float(reward_sums[env_id]),
                    "initial": initial_metrics,
                    "final": final_metrics,
                    "episode_runtime_s": max_runtime_s,
                    "env_runtime_s": env_runtime_s,
                },
            )
        if eval_frequency > 0 and eval_metrics_logger is not None and eval_run_logger is not None:
            if episode % eval_frequency == 0:
                eval_metrics = eval_a3c(
                    agent,
                    configs,
                    episode,
                    eval_metrics_logger,
                    eval_run_logger,
                )
                eval_reward = eval_metrics["reward"]
                if best_eval_reward is None or eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    best_model_path = os.path.join(
                        configs["model-path"], "{}_best.pt".format(log_basename)
                    )
                    torch.save(agent.model.state_dict(), best_model_path)
    run_logger.close()
    if eval_run_logger is not None:
        eval_run_logger.close()


def test_a3c(agent, configs):
    envs = agent.envs
    assert agent.num_step == envs.safe_get_attr("dims_of_tunable_state"), (
        f"num_step {agent.num_step} vs tunable state: {envs.safe_get_attr('dims_of_tunable_state')}"
    )

    preference = agent.preference
    fixed_w = preference.init_preference()
    explore_w = preference.generate_preference(agent.num_parallel, fixed_w)

    for episode in range(configs["algo"]["test"]["max-search-round"]):
        agent.buffer.reset()
        state = envs.reset()

        for _ in range(agent.num_step):
            action, _, log_prob = agent.get_action(state, explore_w)
            action = np.asarray(action, dtype=np.int64).reshape(agent.num_parallel)
            next_state, reward, done, info = envs.step(action)

            terminal_safe_next_state = np.stack([
                info[idx].get("terminal_observation", next_state[idx]) if done[idx] else next_state[idx]
                for idx in range(agent.num_parallel)
            ])
            agent.buffer.insert(state, action, terminal_safe_next_state, reward, done, log_prob)

            for i in range(1, agent.num_parallel):
                if done[i]:
                    explore_w = preference.renew_preference(explore_w, i)
            state = next_state


def a3c(configs):
    # In test mode we run a single environment so each episode maps to
    # run_test_{idx} under base_path/RL/run_dir.
    if configs["algo"]["mode"] == "test" and configs["algo"].get("num-parallel", 1) != 1:
        configs = copy.deepcopy(configs)
        configs["algo"]["num-parallel"] = 1

    agent = ECOAgent(configs, ECOEnv)
    if configs["algo"]["mode"] == "train":
        train_a3c(agent, configs)
    else:
        test_a3c(agent, configs)
