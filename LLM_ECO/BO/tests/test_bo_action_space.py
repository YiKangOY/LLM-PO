import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
UTILS_ROOT = os.path.join(ROOT, "utils")
if UTILS_ROOT not in sys.path:
    sys.path.insert(0, UTILS_ROOT)
ALGO_ROOT = os.path.join(ROOT, "algo")
if ALGO_ROOT not in sys.path:
    sys.path.insert(0, ALGO_ROOT)

from algo.bo_action_space import BOActionSpace
from algo.solver import BOOMExplorerSolver
from configs.design_configs import DESIGN_CONFIGS


def test_bo_action_space_uses_rl_discrete_threshold_values():
    action_spaces = DESIGN_CONFIGS["NV_NVDLA_partition_m"]["action_spaces"]
    action_space = BOActionSpace(action_spaces)

    slack_lesser_values = sorted(set(float(val) for val in action_space.actions[:, 20]))
    slack_greater_values = sorted(set(float(val) for val in action_space.actions[:, 21]))
    setup_guard_values = sorted(set(float(val) for val in action_space.actions[:, 22]))
    target_values = sorted(set(float(val) for val in action_space.actions[:, 0]))

    assert slack_lesser_values == [0.0, 0.5, 1.0]
    assert slack_greater_values == [0.0, 0.5, 1.0]
    assert setup_guard_values == [0.0, 0.5, 1.0]
    assert target_values == [0.0, 0.5, 1.0]
    assert int((action_space.actions[:, 23] == 1.0).sum()) == 1


def test_bo_action_space_sampling_returns_exact_rl_actions():
    action_spaces = DESIGN_CONFIGS["NV_NVDLA_partition_m"]["action_spaces"]
    action_space = BOActionSpace(action_spaces)
    sampled = action_space.sample(32)

    sampled_rows = {tuple(row.tolist()) for row in sampled}
    all_rows = {tuple(row.tolist()) for row in action_space.actions}

    assert sampled_rows.issubset(all_rows)


class DummyRunner(object):
    def __init__(self):
        self.configs = {"design-name": "demo"}
        self.env_config = {
            "reward_weights": {
                "area": 2.0,
                "hold_tns": 4.0,
                "power": 5.0,
                "setup_tns": 7.0,
            }
        }


class DummyProblem(object):
    def __init__(self, report_path: str):
        self.rl_runner = DummyRunner()
        self.configs = {
            "rl-bo": {
                "candidate-pool": 1,
                "initial-samples": 0,
            },
            "bo": {"max-bo-steps": 1},
            "report": {"path": report_path},
        }
        self._ref_point = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float32)


def test_solver_uses_rl_reward_weights_for_ehvi(tmp_path):
    solver = BOOMExplorerSolver(DummyProblem(str(tmp_path)))
    weights = solver.objective_weights.detach().cpu().tolist()
    solver.run_logger.close()

    assert weights == [2.0, 4.0, 5.0, 7.0]
