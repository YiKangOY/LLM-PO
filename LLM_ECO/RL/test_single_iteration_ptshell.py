#!/usr/bin/env python3
"""
Test script for running RL with 1 iteration using real pt_shell execution

This script tests the complete flow:
1. Initialize RL environment with use_simulation=False
2. Reset environment
3. Take one action
4. Execute command via pt_shell
5. Parse results and calculate reward
"""

import os
import sys
import argparse

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../Agent'))

from rl_environment import ECOEnvironment
from Agent.eco_ppa_agent import load_reports_for_iteration
from design_configs import get_design_overrides
from rl_config import ENV_CONFIG, load_config


def test_single_iteration_with_ptshell(
    design_name: str = "ECO_Vex",
    render: bool = True,
    use_random_action: bool = True
):
    """
    Test running 1 RL iteration with real pt_shell execution

    Args:
        design_name: Design name
        render: Show environment state
        use_random_action: Use random action (vs specific action)
    """
    print("="*60)
    print("Test: Single RL Iteration with pt_shell")
    print("="*60)
    print(f"Design: {design_name}")
    print(f"Mode: REAL pt_shell execution")
    print()

    load_config(config_overrides=get_design_overrides(design_name))
    base_path = ENV_CONFIG["base_path"]

    # Check prerequisites
    template_path = os.path.join(base_path, 'EDA_scripts/execute_command.tcl')
    if not os.path.exists(template_path):
        print(f"✗ Error: Template not found: {template_path}")
        print("  This test requires the ECO template file.")
        return False

    session_path = os.path.join(base_path, 'eco_session_0')
    if not os.path.exists(session_path):
        print(f"⚠ Warning: Session file not found: {session_path}")
        print("  pt_shell execution may fail without initial session.")
        print()

    # Create environment with REAL execution
    print("Creating RL environment (use_simulation=False)...")
    env = ECOEnvironment(
        design_name=design_name,
        reports_loader=load_reports_for_iteration,
        use_simulation=False,  # ← KEY: Use real pt_shell
        render_mode="human" if render else None
    )

    print(f"✓ Environment created")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    print()

    try:
        # Reset environment
        print("Resetting environment...")
        obs, info = env.reset()
        print(f"✓ Reset complete")
        print(f"  Initial violations: {info['violations']}")
        print(f"  Budget: {info['remaining_budget']:.1f}s")
        print()

        # Generate action
        if use_random_action:
            print("Generating random action...")
            action = env.action_space.sample()
        else:
            # Create a specific action (timing optimization)
            import numpy as np
            action = np.zeros(env.action_space.shape[0])
            action[0] = 0.0  # Timing target
            action[1] = 0.0  # Setup type
            action[2:7] = [1, 1, 0, 0, 0]  # Methods: fix_fanout, gate_sizing
            action[7:11] = [1, 0, 0, 0]  # Cell types: combinational
            action[11] = 0.1  # Area limit: 100
            print("Using predefined timing action...")

        # Execute step
        print("\n" + "="*60)
        print("EXECUTING STEP WITH pt_shell")
        print("="*60)
        print("This will:")
        print("  1. Decode action to TCL command")
        print("  2. Fill execute_command.tcl template")
        print("  3. Run: pt_shell -f Run_scripts_rl_0.tcl")
        print("  4. Parse execution results")
        print()
        print("⏳ Executing... (this may take 30-180 seconds)")
        print()

        obs, reward, terminated, truncated, info = env.step(action)

        # Display results
        print("\n" + "="*60)
        print("STEP COMPLETED")
        print("="*60)
        print(f"✓ Iteration: {info['iteration']}")
        print(f"✓ Command valid: {info['command_valid']}")
        print(f"✓ TCL command:")
        print(f"  {info['tcl_command']}")
        print()
        print(f"Reward: {reward:.4f}")
        print(f"  Components:")
        for key, value in info['reward_components'].items():
            if key != 'total_reward':
                print(f"    {key}: {value:.4f}")
        print()
        print(f"Violations: {info['violations']}")
        print(f"Budget remaining: {info['remaining_budget']:.1f}s")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print()

        # Check generated files
        print("Generated files:")
        script_file = os.path.join(base_path, 'Run_scripts_rl_0.tcl')
        log_file = os.path.join(base_path, 'logs/pt_rl_0.log')

        if os.path.exists(script_file):
            print(f"  ✓ Script: {script_file}")
        else:
            print(f"  ✗ Script not found: {script_file}")

        if os.path.exists(log_file):
            print(f"  ✓ Log: {log_file}")
            # Show last few lines of log
            with open(log_file, 'r') as f:
                lines = f.readlines()
                print("\n  Last 10 lines of log:")
                for line in lines[-10:]:
                    print(f"    {line.rstrip()}")
        else:
            print(f"  ✗ Log not found: {log_file}")

        print()
        print("="*60)
        print("✓ TEST PASSED - Single iteration with pt_shell completed!")
        print("="*60)
        return True

    except Exception as e:
        print("\n" + "="*60)
        print("✗ TEST FAILED")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Test RL single iteration with pt_shell execution"
    )
    parser.add_argument(
        "--design", type=str, default="ECO_Vex",
        help="Design name (default: ECO_Vex)"
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Don't render environment state"
    )
    parser.add_argument(
        "--specific-action", action="store_true",
        help="Use specific action instead of random"
    )

    args = parser.parse_args()

    success = test_single_iteration_with_ptshell(
        design_name=args.design,
        render=not args.no_render,
        use_random_action=not args.specific_action
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
