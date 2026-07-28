#!/usr/bin/env python3
"""
Lightweight manager for the persistent pt_shell server/client flow shown in
example_env. It prepares init/server assets under each env run_dir, stages
end.tcl in run_scripts, starts pt_shell in server mode, and sends RUN commands
for iteration scripts.
"""

import os
import socket
import subprocess
import textwrap
import time
from typing import Optional, Tuple

from rl_command_executor import RunPaths
from rl_config import ENV_CONFIG


class PTServerManager:
    """
    Controls a pt_shell server process for a single environment.
    """

    def __init__(self, run_paths: RunPaths, enable: bool = True):
        self.run_paths = run_paths
        self.enable = enable and run_paths is not None
        self.process: Optional[subprocess.Popen] = None
        self.server_py_path = os.path.join(self.run_paths.run_dir, "server.py")
        self.client_py_path = os.path.join(self.run_paths.run_dir, "client.py")
        self.init_tcl_path = os.path.join(self.run_paths.run_dir, "init.tcl")
        self.end_tcl_path = os.path.join(self.run_paths.scripts_dir, "end.tcl")
        self.start_timeout = ENV_CONFIG.get("pt_server_start_timeout_s", 60.0)
        self.command_timeout = ENV_CONFIG.get(
            "pt_server_command_timeout_s", self.start_timeout
        )

    def ensure_running(self) -> bool:
        """
        Start the server if needed and return True when it is ready.
        """
        if not self.enable:
            return False
        if self.is_running():
            return True
        return self.start()

    def start(self) -> bool:
        """
        Spawn pt_shell in server mode using init.tcl.
        """
        if not self.enable:
            return False

        if hasattr(self.run_paths, "ensure_server_port"):
            self.run_paths.ensure_server_port()
        self._prepare_assets()
        try:
            self.process = subprocess.Popen(
                ["bash", "-lc", f"pt_shell -f {self.init_tcl_path} | tee pt.log"],
                cwd=self.run_paths.run_dir,
            )
        except FileNotFoundError:
            # pt_shell not available; caller will fall back to legacy path
            self.process = None
            if hasattr(self.run_paths, "release_port"):
                self.run_paths.release_port()
            return False

        if self._wait_for_port():
            return True
        if hasattr(self.run_paths, "release_port"):
            self.run_paths.release_port()
        return False

    def shutdown(self) -> None:
        """
        Gracefully stop the server by sending the end script and waiting for
        the process to exit.
        """
        if not self.enable:
            return

        if self.is_running():
            # Try to exit cleanly inside pt_shell
            self.run_script(self.end_tcl_path)
            try:
                self.process.wait(timeout=10)
            except Exception:
                pass

        if self.process and self.process.poll() is None:
            # Force termination if still alive
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
        self.process = None
        if hasattr(self.run_paths, "release_port"):
            self.run_paths.release_port()

    def is_running(self) -> bool:
        """
        True when the pt_shell process is alive and the server socket accepts
        connections.
        """
        if self.process is None or self.process.poll() is not None:
            return False
        return self._is_port_open()

    def run_script(self, script_path: str) -> Tuple[bool, str]:
        """
        Send a RUN command for the given script. Returns (success, response).
        """
        if not self.ensure_running():
            return False, "server_not_available"

        cmd = f"RUN {script_path}"
        try:
            resp = self._send_command(cmd)
            success = resp.startswith("OK")
            return success, resp
        except Exception as exc:  # noqa: BLE001
            return False, f"ERR exception during RUN: {exc}"

    def _wait_for_port(self) -> bool:
        deadline = time.time() + self.start_timeout
        while time.time() < deadline:
            if self._is_port_open():
                return True
            if self.process and self.process.poll() is not None:
                return False
            time.sleep(0.5)
        return False

    def _is_port_open(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            try:
                s.connect((self.run_paths.server_host, self.run_paths.server_port))
                return True
            except OSError:
                return False

    def _send_command(self, command: str) -> str:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(self.command_timeout)
            s.connect((self.run_paths.server_host, self.run_paths.server_port))
            if not command.endswith("\n"):
                command = command + "\n"
            s.sendall(command.encode("utf-8"))
            return self._read_line(s)

    @staticmethod
    def _read_line(sock: socket.socket) -> str:
        buf = b""
        while True:
            chunk = sock.recv(4096)
            if chunk == b"":
                return ""
            buf += chunk
            idx = buf.find(b"\n")
            if idx != -1:
                line = buf[:idx]
                return line.decode("utf-8", "replace")

    def _prepare_assets(self) -> None:
        """
        Write init.tcl, server.py, client.py into run_dir and end.tcl into
        run_scripts using the example_env layout with per-env paths/ports.
        """
        os.makedirs(self.run_paths.run_dir, exist_ok=True)
        os.makedirs(self.run_paths.scripts_dir, exist_ok=True)
        server_source = self._render_server_py()
        client_source = self._render_client_py()
        init_tcl = self._render_init_tcl()
        end_tcl = self._render_end_tcl()

        with open(self.server_py_path, "w") as f:
            f.write(server_source)
        with open(self.client_py_path, "w") as f:
            f.write(client_source)
        with open(self.init_tcl_path, "w") as f:
            f.write(init_tcl)
        with open(self.end_tcl_path, "w") as f:
            f.write(end_tcl)

    def _render_init_tcl(self) -> str:
        session0 = self.run_paths.session_file(0)
        return textwrap.dedent(
            f"""\
            restore_session {{{session0}}}
            set_host_options -max_cores 16;
            py_eval -file {{{self.server_py_path}}}
            """
        )

    def _render_end_tcl(self) -> str:
        template_path = os.path.join(
            os.path.dirname(__file__),
            "example_env",
            "end.tcl",
        )
        with open(template_path, "r") as f:
            template = f.read()
        finish_session = os.path.join(self.run_paths.run_dir, "finish_eco_session")
        return template.replace(
            "data/ECO_Vex/RL/run_dir/env_0/finish_eco_session",
            finish_session,
        )

    def _render_server_py(self) -> str:
        host = self.run_paths.server_host
        port = self.run_paths.server_port
        return textwrap.dedent(
            f"""\
            # -*- coding: utf-8 -*-
            import socket
            import snps


            def read_line(sock):
                buf = b""
                while True:
                    chunk = sock.recv(4096)
                    if chunk == b"":
                        return ""  # client closed
                    buf += chunk
                    idx = buf.find(b"\\n")
                    if idx != -1:
                        line = buf[:idx]
                        return line.decode("utf-8", "replace")


            def handle_command(line):
                # remove trailing \\r if present (for Windows-style \\r\\n)
                if len(line) > 0 and line[-1] == "\\r":
                    line = line[:-1]

                parts = line.split(" ")

                if len(parts) == 0 or parts[0] == "":
                    return "ERR empty command\\n"

                cmd = parts[0]

                if cmd == "PING":
                    return "OK PONG\\n"

                if cmd == "ECHO":
                    # ECHO <text...>
                    if len(line) < 6:
                        return "ERR ECHO requires text\\n"
                    return "OK " + line[5:] + "\\n"

                if cmd == "RUN":
                    # RUN <command string>
                    if len(parts) < 2:
                        return "ERR RUN requires a command\\n"
                    command_to_run = " ".join(parts[1:])

                    try:
                        snps.cmd.source(command_to_run)
                        return "OK command executed successfully\\n"
                    except Exception as e:  # noqa: BLE001
                        return f"ERR execution failed: {{str(e)}}\\n"

                return "ERR unknown command: " + cmd + "\\n"


            def main():
                host = "{host}"
                port = {port}

                srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind((host, port))
                srv.listen(1)

                print("server listening on %s:%d" % (host, port))

                while True:
                    conn, addr = srv.accept()
                    print("client connected:", addr)

                    while True:
                        line = read_line(conn)
                        if line == "":
                            break
                        resp = handle_command(line)
                        conn.sendall(resp.encode("utf-8"))

                    conn.close()
                    print("client disconnected:", addr)


            if __name__ == "__main__":
                main()
            """
        )

    def _render_client_py(self) -> str:
        template_path = os.path.join(
            os.path.dirname(__file__),
            "example_env",
            "client.py",
        )
        with open(template_path, "r") as f:
            template = f.read()
        return template.replace(
            'def send_one(cmd, host="127.0.0.1", port=9009):',
            f'def send_one(cmd, host="{self.run_paths.server_host}", port={self.run_paths.server_port}):',
        )
