"""
RL-Based ECO Optimization System

A reinforcement learning approach to Engineering Change Order (ECO) optimization
for integrated circuit design, implemented as a parallel alternative to the
LLM-based system in Agent/.

Key Components:
- ECOEnvironment: Gymnasium environment for ECO optimization
- ECOEnv/ECOAgent/eco_a3c: A3C stack mirroring the reference `dse` framework
- StateEncoder: Converts reports to observation vectors
- ActionDecoder: Converts RL actions to TCL commands
- RewardCalculator: Computes rewards from state transitions
"""

from .rl_environment import ECOEnvironment
from .state_encoder import StateEncoder
from .action_decoder import ActionDecoder
from .reward_calculator import RewardCalculator
from .eco_env import ECOEnv
from .eco_agent import ECOAgent
from .eco_a3c import a3c as eco_a3c
from .eco_a3c_config import default_eco_a3c_config

__version__ = "1.0.0"
__all__ = [
    "ECOEnvironment",
    "ECOEnv",
    "ECOAgent",
    "eco_a3c",
    "default_eco_a3c_config",
    "StateEncoder",
    "ActionDecoder",
    "RewardCalculator",
]
