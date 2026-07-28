#!/usr/bin/env python3
"""
Run path helper for Agent PT server execution.
"""

import os
import socket

from config_noBoth import PT_SERVER_CONFIG, base_path, get_agent_dir, get_agent_logs_dir


_RESERVED_PORTS = set()


class AgentRunPaths:
    """
    Holds per-round paths for pt_shell server execution.
    """

    def __init__(self, round_index, run_root=None, server_port=None):
        self.round_index = round_index
        self.workspace = base_path
        self.run_root = run_root or get_agent_dir()
        self.run_dir = os.path.join(self.run_root, f"round_{round_index}")
        self.scripts_dir = self.run_dir
        self.logs_dir = get_agent_logs_dir()
        self.use_pt_server = PT_SERVER_CONFIG["use_pt_server"]
        self.server_host = PT_SERVER_CONFIG["host"]
        if server_port is None:
            self.server_port = PT_SERVER_CONFIG["base_port"] + round_index * PT_SERVER_CONFIG["port_stride"]
        else:
            self.server_port = server_port

    def prepare(self):
        os.makedirs(self.run_root, exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def session_file(self, iteration):
        return os.path.join(self.workspace, f"eco_session_{iteration}")

    def ensure_server_port(self):
        stride = PT_SERVER_CONFIG["port_stride"]
        port = self.server_port
        while True:
            if self._is_port_available(port):
                self.server_port = port
                _RESERVED_PORTS.add(port)
                return port
            port += stride

    def release_port(self):
        if self.server_port in _RESERVED_PORTS:
            _RESERVED_PORTS.remove(self.server_port)

    def _is_port_available(self, port):
        if port in _RESERVED_PORTS:
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            result = sock.connect_ex((self.server_host, port))
            return result != 0
