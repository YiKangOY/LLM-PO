#!/usr/bin/env python3
"""
Action Decoder for RL ECO System
Converts RL agent actions into executable TCL commands
"""

import numpy as np
import sys
import os
import re
from typing import Dict, List, Any, Tuple, Optional

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Agent.configs import BUFFER_LIST
from rl_config import (
    RL_CONFIG,
    OPTIMIZATION_TARGETS,
    TIMING_ACTION_SPACE,
    POWER_ACTION_SPACE,
    AREA_ACTION_SPACE,
    get_action_space_config
)


class ActionDecoder:
    """
    Decodes RL actions into executable TCL commands

    Supports:
    - Hierarchical action space (high-level target + specialized actions)
    - Flat action space (unified actions)
    - Mixed discrete/continuous parameters
    - Multi-binary selections (actions, cell types)
    """

    def __init__(self, action_space_type: str = None):
        """
        Initialize action decoder

        Args:
            action_space_type: "hierarchical" or "flat"
        """
        self.action_space_type = action_space_type or RL_CONFIG["action_space_type"]

        if self.action_space_type not in ["hierarchical", "flat"]:
            raise ValueError(f"Invalid action space type: {self.action_space_type}")

    def decode_action(self, action: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Decode RL action into TCL command

        Args:
            action: Dictionary containing action parameters from RL agent

        Returns:
            Tuple of (tcl_command, action_info)
            - tcl_command: Formatted TCL command string
            - action_info: Dictionary with decoded action details
        """
        if action.get("noop"):
            return "", {"optimization_target": "noop", "noop": True}

        if self.action_space_type == "hierarchical":
            return self._decode_hierarchical_action(action)
        else:
            return self._decode_flat_action(action)

    def _decode_hierarchical_action(self, action: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Decode hierarchical action (high-level + specialized)"""
        # Extract high-level optimization target
        target_idx = action.get("optimization_target", 0)
        optimization_target = OPTIMIZATION_TARGETS[target_idx]

        # Route to specialized decoder
        if optimization_target == "timing":
            return self._decode_timing_action(action)
        elif optimization_target == "power":
            return self._decode_power_action(action)
        elif optimization_target == "area":
            return self._decode_area_action(action)
        else:
            raise ValueError(f"Unknown optimization target: {optimization_target}")

    def _decode_timing_action(self, action: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Decode timing-specific action"""
        action_space = TIMING_ACTION_SPACE

        # Extract discrete parameters
        violation_type_idx = action.get("violation_type", 0)
        violation_type = action_space["violation_type"][violation_type_idx]

        # Extract multi-binary method selections
        actions_binary = action.get("actions", np.zeros(len(action_space["actions"])))
        selected_actions = [
            method for i, method in enumerate(action_space["actions"])
            if actions_binary[i] > 0.5  # Threshold for binary selection
        ]

        # Extract multi-binary cell type selections
        cell_classes_binary = action.get("cell_classes", np.zeros(len(action_space["cell_classes"])))
        selected_cell_classes = [
            cell_class for i, cell_class in enumerate(action_space["cell_classes"])
            if cell_classes_binary[i] > 0.5
        ]

        # Extract continuous parameters
        area_cap = action["area_cap"]
        if area_cap is None:
            area_cap = 2.0

        # Extract physical mode (discrete: 0=open_slot, 1=occupied_slot)
        site_mode_idx = action.get("site_mode", 0)
        site_mode = "open_slot" if site_mode_idx == 0 else "occupied_slot"

        slack_above = action["slack_above"]
        if slack_above is None:
            slack_above = 0.0
        slack_below = action["slack_below"]
        if slack_below is None:
            slack_below = 0.0

        # Ensure at least one method and cell type selected
        if not selected_actions:
            selected_actions = ["gate_sizing"]  # Default method
        if not selected_cell_classes:
            selected_cell_classes = ["combinational"]  # Default cell type

        # Build TCL command
        tcl_command, applied_method, applied_cell_class, applied_area_cap, applied_slack_lesser, applied_slack_greater = self._build_timing_command(
            violation_type=violation_type,
            actions=selected_actions,
            cell_classes=selected_cell_classes,
            area_cap=area_cap,
            site_mode=site_mode,
            slack_above=slack_above,
            slack_below=slack_below,
        )

        action_info = {
            "optimization_target": "timing",
            "violation_type": violation_type,
            "actions": [applied_method],
            "cell_classes": [applied_cell_class] if applied_cell_class else [],
            "area_cap": applied_area_cap,
            "site_mode": site_mode,
            "slack_above": applied_slack_lesser,
            "slack_below": applied_slack_greater,
        }

        return tcl_command, action_info

    def _decode_power_action(self, action: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Decode power-specific action"""
        action_space = POWER_ACTION_SPACE

        # Extract parameters
        actions_binary = action.get("actions", np.zeros(len(action_space["actions"])))
        selected_actions = [
            method for i, method in enumerate(action_space["actions"])
            if actions_binary[i] > 0.5
        ]

        cell_classes_binary = action.get("cell_classes", np.zeros(len(action_space["cell_classes"])))
        selected_cell_classes = [
            cell_class for i, cell_class in enumerate(action_space["cell_classes"])
            if cell_classes_binary[i] > 0.5
        ]

        power_scope_idx = action.get("power_scope", 0)  # Default: total
        power_scope = action_space["power_scope"][power_scope_idx]
        setup_guard = action["setup_guard"]
        if setup_guard is None:
            setup_guard = 0.0

        # Defaults
        if not selected_actions:
            selected_actions = ["gate_sizing"]
        if not selected_cell_classes:
            selected_cell_classes = ["combinational"]

        # Build TCL command
        tcl_command, applied_method, applied_cell_class, applied_setup_guard = self._build_power_command(
            actions=selected_actions,
            cell_classes=selected_cell_classes,
            power_scope=power_scope,
            setup_guard=setup_guard
        )

        action_info = {
            "optimization_target": "power",
            "actions": [applied_method],
            "cell_classes": [applied_cell_class] if applied_cell_class else [],
            "power_scope": power_scope,
            "setup_guard": applied_setup_guard,
        }

        return tcl_command, action_info

    def _decode_area_action(self, action: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Decode area-specific action"""
        action_space = AREA_ACTION_SPACE

        # Extract parameters
        actions_binary = action.get("actions", np.zeros(len(action_space["actions"])))
        selected_actions = [
            method for i, method in enumerate(action_space["actions"])
            if actions_binary[i] > 0.5
        ]

        cell_classes_binary = action.get("cell_classes", np.zeros(len(action_space["cell_classes"])))
        selected_cell_classes = [
            cell_class for i, cell_class in enumerate(action_space["cell_classes"])
            if cell_classes_binary[i] > 0.5
        ]

        setup_guard = action["setup_guard"]
        if setup_guard is None:
            setup_guard = 0.0

        # Defaults
        if not selected_actions:
            selected_actions = ["gate_sizing"]
        if not selected_cell_classes:
            selected_cell_classes = ["combinational"]

        # Build TCL command
        tcl_command, applied_method, applied_cell_class, applied_setup_guard = self._build_area_command(
            actions=selected_actions,
            cell_classes=selected_cell_classes,
            setup_guard=setup_guard
        )

        action_info = {
            "optimization_target": "area",
            "actions": [applied_method],
            "cell_classes": [applied_cell_class] if applied_cell_class else [],
            "setup_guard": applied_setup_guard,
        }

        return tcl_command, action_info

    def _decode_flat_action(self, action: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Decode flat action space (all parameters in one action)"""
        # Similar to hierarchical but optimization target is also part of action
        target_idx = action.get("optimization_target", 0)
        optimization_target = OPTIMIZATION_TARGETS[target_idx]

        # Reuse hierarchical decoders based on selected target
        if optimization_target == "timing":
            return self._decode_timing_action(action)
        elif optimization_target == "power":
            return self._decode_power_action(action)
        else:
            return self._decode_area_action(action)

    def _build_timing_command(
        self,
        violation_type: str,
        actions: List[str],
        cell_classes: List[str],
        area_cap: float,
        site_mode: str = "open_slot",
        slack_above: Optional[float] = None,
        slack_below: Optional[float] = None
    ) -> Tuple[str, str, str, Optional[float], Optional[float], Optional[float]]:
        """Build TCL command for timing optimization

        Correct format: opt_timing -violation_type [setup|hold] -cell_class [cell_class] -actions [method] -site_mode [mode] (-area_cap [x]) (-slack_above [>0]) (-slack_below [<0])
        - violation_type: setup or hold (only one)
        - actions: gate_sizing, gate_sizing_side_load, buffer_insertion (only one for setup); gate_sizing, buffer_insertion (only one for hold)
        - cell_class: combinational, sequential, clock_tree (only one)
        - site_mode: open_slot, occupied_slot
        - area_cap: optional, only with gate_sizing, rounded to an integer within config range (default 2)
        - slack_above: optional, positive slack threshold
        - slack_below: optional, negative slack threshold
        """
        # Select single method (prefer gate_sizing, then buffer_insertion)
        method_priority = ["gate_sizing", "buffer_insertion", "gate_sizing_side_load"]
        selected_method = None
        for method in method_priority:
            if method in actions:
                # Check if method is valid for timing type
                if violation_type == "hold" and method == "gate_sizing_side_load":
                    continue  # Not valid for hold
                selected_method = method
                break
        if not selected_method:
            selected_method = "gate_sizing"  # Default

        # Select single cell type (prefer combinational)
        cell_class_priority = ["combinational", "sequential", "clock_tree"]
        selected_cell_class = None
        # Prioritize method over cell_class: if method is not gate_sizing, fall back to combinational
        if selected_method != "gate_sizing":
            selected_cell_class = "combinational"
        else:
            for ct in cell_class_priority:
                if ct in cell_classes:
                    selected_cell_class = ct
                    break
            if not selected_cell_class:
                selected_cell_class = "combinational"  # Default

        # Build command
        tcl_command = (
            f"opt_timing "
            f"-violation_type {violation_type} "
            f"-cell_class {selected_cell_class} "
            f"-actions {selected_method} "
            f"-site_mode {site_mode}"
        )

        # Add area_cap only for gate_sizing actions
        # Note: The area_cap will be processed by _formulate_tcl_command in eco_ppa_agent.py
        # which converts it to set_app_var eco_alternative_area_ratio_threshold commands
        applied_area_cap = None
        if selected_method == "gate_sizing":
            area_cfg = TIMING_ACTION_SPACE["area_cap"]
            if isinstance(area_cfg, (list, tuple, np.ndarray)):
                area_vals = np.asarray(area_cfg, dtype=np.float32)
                area_min = float(area_vals.min())
                area_max = float(area_vals.max())
            else:
                area_min = float(area_cfg["low"])
                area_max = float(area_cfg["high"])

            bounded_area = max(min(float(area_cap), area_max), area_min)
            min_int = int(round(area_min))
            max_int = int(round(area_max))
            rounded_area = int(round(bounded_area))
            if rounded_area < min_int:
                rounded_area = min_int
            elif rounded_area > max_int:
                rounded_area = max_int
            applied_area_cap = rounded_area
            # Include -area_cap in the command string for processing by eco_ppa_agent
            tcl_command += f" -area_cap {applied_area_cap}"

        applied_slack_lesser = None
        if slack_above is not None:
            applied_slack_lesser = max(float(slack_above), 0.001)
            tcl_command += f" -slack_above {applied_slack_lesser}"

        applied_slack_greater = None
        if slack_below is not None:
            applied_slack_greater = -abs(float(slack_below))
            if applied_slack_greater == 0.0:
                applied_slack_greater = -0.001
            tcl_command += f" -slack_below {applied_slack_greater}"

        return tcl_command, selected_method, selected_cell_class, applied_area_cap, applied_slack_lesser, applied_slack_greater

    def _build_power_command(
        self,
        actions: List[str],
        cell_classes: List[str],
        power_scope: str,
        setup_guard: Optional[float] = None
    ) -> Tuple[str, str, Optional[str], Optional[float]]:
        """Build TCL command for power optimization

        Correct format: opt_power -actions [method] -cell_class [cell_class] -power_scope [mode] (-setup_guard [<=0])
        - actions: gate_sizing, buffer_removal (only one)
        - power_scope: total, dynamic, leakage
        - cell_class: combinational, sequential (only one, not used with buffer_removal)
        - setup_guard: optional, max 0 (negative allows timing sacrifice)
        """
        # Select single method (prefer gate_sizing)
        method_priority = ["gate_sizing", "buffer_removal"]
        selected_method = None
        for method in method_priority:
            if method in actions:
                selected_method = method
                break
        if not selected_method:
            selected_method = "gate_sizing"  # Default

        # Build base command
        tcl_command = f"opt_power -actions {selected_method} -power_scope {power_scope}"

        selected_cell_class = None
        # Add cell_class only if NOT buffer_removal
        if selected_method != "buffer_removal":
            # Select single cell type (prefer combinational)
            cell_class_priority = ["combinational", "sequential"]
            for ct in cell_class_priority:
                if ct in cell_classes:
                    selected_cell_class = ct
                    break
            if not selected_cell_class:
                selected_cell_class = "combinational"  # Default

            tcl_command += f" -cell_class {selected_cell_class}"

        applied_setup_guard = None
        if setup_guard is not None:
            applied_setup_guard = min(float(setup_guard), 0.0)
            tcl_command += f" -setup_guard {applied_setup_guard}"

        return tcl_command, selected_method, selected_cell_class, applied_setup_guard

    def _build_area_command(
        self,
        actions: List[str],
        cell_classes: List[str],
        setup_guard: Optional[float] = None
    ) -> Tuple[str, str, Optional[str], Optional[float]]:
        """Build TCL command for area optimization

        Correct format: opt_area -actions [method] -cell_class [cell_class] (-setup_guard [<=0])
        - actions: gate_sizing, buffer_removal (only one)
        - cell_class: combinational, sequential, clock_tree (only one, not used with buffer_removal)
        - setup_guard: optional, max 0 (negative allows timing sacrifice)
        """
        # Select single method (prefer gate_sizing)
        method_priority = ["gate_sizing", "buffer_removal"]
        selected_method = None
        for method in method_priority:
            if method in actions:
                selected_method = method
                break
        if not selected_method:
            selected_method = "gate_sizing"  # Default

        # Build base command
        tcl_command = f"opt_area -actions {selected_method}"

        selected_cell_class = None
        # Add cell_class only if NOT buffer_removal
        if selected_method != "buffer_removal":
            # Select single cell type (prefer combinational)
            cell_class_priority = ["combinational", "sequential", "clock_tree"]
            for ct in cell_class_priority:
                if ct in cell_classes:
                    selected_cell_class = ct
                    break
            if not selected_cell_class:
                selected_cell_class = "combinational"  # Default

            tcl_command += f" -cell_class {selected_cell_class}"

        applied_setup_guard = None
        if setup_guard is not None:
            applied_setup_guard = min(float(setup_guard), 0.0)
            tcl_command += f" -setup_guard {applied_setup_guard}"

        return tcl_command, selected_method, selected_cell_class, applied_setup_guard

    def validate_command(self, tcl_command: str) -> bool:
        """
        Validate that generated TCL command has correct syntax

        Args:
            tcl_command: Generated TCL command

        Returns:
            True if valid, False otherwise
        """
        # Basic validation checks
        if tcl_command == "":
            return True

        # Check for required command prefix
        valid_prefixes = ["opt_timing", "opt_power", "opt_area"]
        if not any(tcl_command.startswith(prefix) for prefix in valid_prefixes):
            return False

        # Timing-specific validation
        if tcl_command.startswith("opt_timing"):
            required_params = ["-violation_type", "-cell_class", "-actions", "-site_mode"]
            if not all(param in tcl_command for param in required_params):
                return False
            # Enforce legal combinations: hold cannot use gate_sizing_side_load; non-gate_sizing must use combinational
            method_match = re.search(r"-actions\s+(\S+)", tcl_command)
            cell_class_match = re.search(r"-cell_class\s+(\S+)", tcl_command)
            type_match = re.search(r"-violation_type\s+(\S+)", tcl_command)
            if type_match and method_match:
                violation_type_val = type_match.group(1)
                method_val = method_match.group(1)
                if violation_type_val == "hold" and method_val == "gate_sizing_side_load":
                    return False
            if method_match and cell_class_match:
                method_val = method_match.group(1)
                cell_class_val = cell_class_match.group(1)
                if method_val != "gate_sizing" and cell_class_val != "combinational":
                    return False
            slack_lesser_match = re.search(r"-slack_above\s+([-+]?[\d\.Ee+-]+)", tcl_command)
            if slack_lesser_match and float(slack_lesser_match.group(1)) <= 0:
                return False
            slack_greater_match = re.search(r"-slack_below\s+([-+]?[\d\.Ee+-]+)", tcl_command)
            if slack_greater_match and float(slack_greater_match.group(1)) >= 0:
                return False

        # Power-specific validation
        elif tcl_command.startswith("opt_power"):
            required_params = ["-actions", "-power_scope"]
            if not all(param in tcl_command for param in required_params):
                return False
            # cell_class is optional (not present for buffer_removal)
            if "-actions buffer_removal" in tcl_command and "-cell_class" in tcl_command:
                return False
            setup_guard_match = re.search(r"-setup_guard\s+([-+]?[\d\.Ee+-]+)", tcl_command)
            if setup_guard_match and float(setup_guard_match.group(1)) > 0:
                return False

        # Area-specific validation
        elif tcl_command.startswith("opt_area"):
            if "-actions" not in tcl_command:
                return False
            # cell_class is optional (not present for buffer_removal)
            if "-actions buffer_removal" in tcl_command and "-cell_class" in tcl_command:
                return False
            setup_guard_match = re.search(r"-setup_guard\s+([-+]?[\d\.Ee+-]+)", tcl_command)
            if setup_guard_match and float(setup_guard_match.group(1)) > 0:
                return False

        return True


if __name__ == "__main__":
    # Test action decoder
    print("=== Testing Action Decoder ===")

    decoder = ActionDecoder(action_space_type="hierarchical")

    # Test timing action
    print("\n--- Timing Action ---")
    timing_action = {
        "optimization_target": 0,  # timing
        "violation_type": 0,  # setup
        "actions": np.array([1, 1, 0, 0, 0]),  # gate_sizing, swap_pin
        "cell_classes": np.array([1, 0, 0, 0]),  # combinational
        "area_cap": 150.0,
        "vth_weights": np.array([0.3, 0.4, 0.3]),
        "slack_above": 0.05,
        "slack_below": -0.05,
    }

    tcl_cmd, info = decoder.decode_action(timing_action)
    print(f"TCL Command:\n{tcl_cmd}")
    print(f"Valid: {decoder.validate_command(tcl_cmd)}")
    print(f"Action Info: {info}")

    # Test power action
    print("\n--- Power Action ---")
    power_action = {
        "optimization_target": 1,  # power
        "actions": np.array([0, 0, 1, 0, 0]),  # swap_vth
        "cell_classes": np.array([1, 1, 0, 0]),  # combinational, sequential
        "power_scope": 2,  # both
        "setup_guard": 0.0,
        "vth_weights": np.array([0.1, 0.3, 0.6]),  # Prefer HVT
    }

    tcl_cmd, info = decoder.decode_action(power_action)
    print(f"TCL Command:\n{tcl_cmd}")
    print(f"Valid: {decoder.validate_command(tcl_cmd)}")
    print(f"Action Info: {info}")

    # Test area action (gate_sizing)
    print("\n--- Area Action (gate_sizing) ---")
    area_action = {
        "optimization_target": 2,  # area
        "actions": np.array([1, 0]),  # gate_sizing only
        "cell_classes": np.array([1, 0, 1]),  # combinational, clock_tree
        "setup_guard": 0.0,
    }

    tcl_cmd, info = decoder.decode_action(area_action)
    print(f"TCL Command:\n{tcl_cmd}")
    print(f"Valid: {decoder.validate_command(tcl_cmd)}")
    print(f"Action Info: {info}")

    # Test area action (buffer_removal)
    print("\n--- Area Action (buffer_removal) ---")
    area_action_remove = {
        "optimization_target": 2,  # area
        "actions": np.array([0, 1]),  # buffer_removal only
        "cell_classes": np.array([1, 0, 0]),
        "setup_guard": 0.0,
    }

    tcl_cmd, info = decoder.decode_action(area_action_remove)
    print(f"TCL Command:\n{tcl_cmd}")
    print(f"Valid: {decoder.validate_command(tcl_cmd)}")
    print(f"Action Info: {info}")

    # Test power action (buffer_removal)
    print("\n--- Power Action (buffer_removal) ---")
    power_action_remove = {
        "optimization_target": 1,  # power
        "actions": np.array([0, 1]),  # buffer_removal
        "cell_classes": np.array([1, 0]),
        "power_scope": 0,  # total
        "setup_guard": 0.0,
    }

    tcl_cmd, info = decoder.decode_action(power_action_remove)
    print(f"TCL Command:\n{tcl_cmd}")
    print(f"Valid: {decoder.validate_command(tcl_cmd)}")
    print(f"Action Info: {info}")
