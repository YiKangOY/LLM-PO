# Simple smoke test for the RL-backed BO sequence evaluator.
# Runs entirely in RL simulation mode to avoid tool dependencies.

import os
import sys
import yaml
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from algo.problem import DesignSpaceProblem


def build_test_config():
    with open("configs/boom-explorer.yml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "online"
    rl_cfg = cfg["rl-bo"]
    # Short episode and tiny pool for quick validation
    rl_cfg["episode-length"] = 2
    rl_cfg["initial-samples"] = 2
    rl_cfg["candidate-pool"] = 2
    rl_cfg["num-envs"] = 2
    rl_cfg["use-simulation"] = True
    # Keep fallback ref point neutral for 4-objective simulation tests
    rl_cfg["ref-point"] = [0.0, 0.0, 0.0, 0.0]
    return cfg


def test_parallel_sequence_eval():
    cfg = build_test_config()
    problem = DesignSpaceProblem(cfg)
    # Seed initial samples (parallel evaluation)
    x0, y0 = problem.initial_samples()
    assert x0.shape[0] == 2
    assert y0.shape[0] == 2
    assert y0.shape[1] == 4
    assert problem.rl_runner.action_dim == 24

    # Evaluate a fresh batch of 2 sequences in parallel
    batch = problem.rl_runner.sample_candidates(pool_size=2)
    seq_low = torch.from_numpy(problem.rl_runner.sequence_low).to(batch.dtype)
    seq_high = torch.from_numpy(problem.rl_runner.sequence_high).to(batch.dtype)
    assert torch.all(batch >= seq_low.unsqueeze(0))
    assert torch.all(batch <= seq_high.unsqueeze(0))
    # Explicitly verify sequence-level clipping against design-config bounds
    clipped = problem.clip_to_bounds(batch + 10.0)
    assert torch.all(clipped >= seq_low.unsqueeze(0))
    assert torch.all(clipped <= seq_high.unsqueeze(0))
    y = problem.evaluate_true(batch)
    assert y.shape[0] == 2
    assert y.shape[1] == 4
    assert problem.rl_runner.action_dim == 24

    # All simulated rewards should be finite
    assert torch.isfinite(y).all()


if __name__ == "__main__":
    cfg = build_test_config()
    problem = DesignSpaceProblem(cfg)
    x0, y0 = problem.initial_samples()
    print("Initial X shape:", x0.shape, "Y shape:", y0.shape, "action_dim:", problem.rl_runner.action_dim)
    batch = problem.rl_runner.sample_candidates(pool_size=2)
    y = problem.evaluate_true(batch)
    print("Batch X shape:", batch.shape, "Y shape:", y.shape, "action_dim:", problem.rl_runner.action_dim)
