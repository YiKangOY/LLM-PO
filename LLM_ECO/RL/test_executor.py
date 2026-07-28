#!/usr/bin/env python3
"""
Test script for RL command executor
"""

import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../Agent'))

from rl_environment import ECOEnvironment
from design_configs import get_design_overrides
from rl_config import ENV_CONFIG, load_config

def test_simulation_mode():
    """Test with simulation (fast, no pt_shell needed)"""
    print("=== Test 1: Simulation Mode ===")
    load_config(config_overrides=get_design_overrides("ECO_Vex"))
    env = ECOEnvironment(
        design_name="ECO_Vex",
        use_simulation=True,
        render_mode="human"
    )

    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")

    # Reset
    obs, info = env.reset()
    print(f"\n✓ Reset successful")
    print(f"  Violations: {info['violations']}")
    print(f"  Budget: {info['remaining_budget']:.1f}s")

    # Take one step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"\n✓ Step successful")
    print(f"  Reward: {reward:.3f}")
    print(f"  Command: {info['tcl_command'][:80]}...")
    print(f"  Valid: {info['command_valid']}")

    env.close()
    print("\n✓ Simulation mode test passed!\n")


def test_real_mode():
    """Test with real execution (requires pt_shell)"""
    print("=== Test 2: Real Execution Mode ===")
    print("Note: This requires pt_shell and design files to be available")

    # Check if we should skip this test
    load_config(config_overrides=get_design_overrides("ECO_Vex"))
    template_path = os.path.join(ENV_CONFIG["base_path"], 'EDA_scripts/execute_command.tcl')
    if not os.path.exists(template_path):
        print(f"⚠ Skipping real mode test - template not found: {template_path}")
        return

    env = ECOEnvironment(
        design_name="ECO_Vex",
        use_simulation=False,  # Use real execution
        render_mode="human"
    )

    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")

    # Reset
    obs, info = env.reset()
    print(f"\n✓ Reset successful")
    print(f"  Violations: {info['violations']}")

    # Note: We don't take a step here unless pt_shell is actually available
    print("\n✓ Real mode environment created successfully")
    print("  To test actual execution, run with pt_shell available")

    env.close()


if __name__ == "__main__":
    print("Testing RL Command Executor\n")

    # Test 1: Simulation mode (always works)
    test_simulation_mode()

    # Test 2: Real mode (only if template exists)
    test_real_mode()

    print("✓ All tests completed!")
