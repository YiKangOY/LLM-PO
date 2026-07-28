import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from algo.bo_report_parser import PowerMetrics, QoRMetrics
from algo.problem import (
    _coerce_ref_point_to_worst_direction,
    _directional_ref_point_from_ranges,
)
from algo.solver import BOOMExplorerSolver, _coerce_normalized_ref_point


def test_directional_ref_point_from_ranges_uses_worst_direction():
    normalization_ranges = {
        "design_area": {"min": 0.0, "max": 5000.0},
        "hold_total_negative_slack": {"min": -25000.0, "max": 0.0},
        "total_power": {"min": 0.0, "max": 0.02},
        "setup_total_negative_slack": {"min": -30000.0, "max": 0.0},
    }

    ref_point = _directional_ref_point_from_ranges(normalization_ranges)

    assert ref_point == [5000.0, -25000.0, 0.02, -30000.0]


def test_invalid_config_ref_point_is_coerced_by_direction():
    directional_ref = [5000.0, -25000.0, 0.02, -25000.0]

    coerced = _coerce_ref_point_to_worst_direction(
        [0.0, 0.0, 0.0, 0.0],
        directional_ref,
    )

    assert coerced == directional_ref


def test_normalized_ref_point_is_forced_below_observations():
    normalized_y = torch.tensor(
        [
            [0.40, 0.60, 0.50, 0.52],
            [0.36, 0.80, 0.40, 0.68],
        ],
        dtype=torch.float32,
    )
    normalized_ref = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)

    adjusted = _coerce_normalized_ref_point(normalized_ref, normalized_y)

    assert torch.all(adjusted < normalized_y.min(dim=0).values)


class DummyRunner(object):
    def __init__(self):
        self.configs = {"design-name": "demo"}
        self.env_config = {
            "normalization_ranges": {
                "design_area": {"min": 0.0, "max": 5000.0},
                "hold_total_negative_slack": {"min": -25000.0, "max": 0.0},
                "total_power": {"min": 0.0, "max": 0.02},
                "setup_total_negative_slack": {"min": -25000.0, "max": 0.0},
            }
        }


class DummyProblem(object):
    def __init__(self, report_path: str):
        self.rl_runner = DummyRunner()
        self.configs = {
            "rl-bo": {
                "candidate-pool": 1,
                "initial-samples": 0,
                "objective-weights": {},
            },
            "bo": {"max-bo-steps": 1},
            "report": {"path": report_path},
        }
        self._ref_point = torch.tensor([5000.0, -25000.0, 0.02, -25000.0], dtype=torch.float32)


def test_extract_ppa_metrics_reads_dataclass_wns(tmp_path):
    solver = BOOMExplorerSolver(DummyProblem(str(tmp_path)))
    detail = {
        "qor_metrics": QoRMetrics(
            setup_critical_path_slack=-0.11,
            hold_critical_path_slack=-0.22,
            total_cell_area=123.0,
        ),
        "power_metrics": PowerMetrics(
            internal_power=1.1,
            switching_power=2.2,
            leakage_power=3.3,
        ),
    }

    metrics = solver._extract_ppa_metrics(
        detail=detail,
        objective_values=np.array([456.0, -12.0, 0.015, -34.0]),
    )
    solver.run_logger.close()

    assert metrics["timing"]["setup_wns"] == -0.11
    assert metrics["timing"]["hold_wns"] == -0.22
    assert metrics["area"]["total_cell_area"] == 123.0
    assert metrics["power"]["internal"] == 1.1
    assert metrics["power"]["switching"] == 2.2
    assert metrics["power"]["leakage"] == 3.3
