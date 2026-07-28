#!/usr/bin/env python3
"""
PT shell server/client manager for Agent.
"""

import os
import socket
import subprocess
import textwrap
import time

from configs import PT_SERVER_CONFIG


class PTServerManager:
    """
    Controls a pt_shell server process for a single Agent run.
    """

    def __init__(self, run_paths, enable=True):
        self.run_paths = run_paths
        self.enable = enable and run_paths is not None and run_paths.use_pt_server
        self.process = None
        self.server_py_path = os.path.join(self.run_paths.run_dir, "server.py")
        self.client_py_path = os.path.join(self.run_paths.run_dir, "client.py")
        self.init_tcl_path = os.path.join(self.run_paths.run_dir, "init.tcl")
        self.end_tcl_path = os.path.join(self.run_paths.scripts_dir, "end.tcl")
        self.start_timeout = PT_SERVER_CONFIG["start_timeout_s"]
        self.command_timeout = PT_SERVER_CONFIG["command_timeout_s"]

    def ensure_running(self):
        if not self.enable:
            return False
        if self.is_running():
            return True
        return self.start()

    def start(self):
        if not self.enable:
            return False
        self.run_paths.ensure_server_port()
        self._prepare_assets()
        self.process = subprocess.Popen(
            ["bash", "-lc", f"pt_shell -f {self.init_tcl_path} | tee pt.log"],
            cwd=self.run_paths.workspace,
        )
        if self._wait_for_port():
            return True
        self.run_paths.release_port()
        return False

    def shutdown(self):
        if not self.enable:
            return

        if self.is_running():
            self.run_script(self.end_tcl_path)
            self.process.terminate()
            self.process.wait()

        self.process = None
        self.run_paths.release_port()

    def is_running(self):
        if self.process is None:
            return False
        if self.process.poll() is not None:
            return False
        return self._is_port_open()

    def run_script(self, script_path):
        if not self.ensure_running():
            return False, "server_not_available"

        command = f"RUN {script_path}"
        response = self._send_command(command)
        success = response.startswith("OK")
        return success, response

    def _wait_for_port(self):
        deadline = time.time() + self.start_timeout
        while time.time() < deadline:
            if self._is_port_open():
                return True
            if self.process.poll() is not None:
                return False
            time.sleep(0.5)
        return False

    def _is_port_open(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            result = s.connect_ex((self.run_paths.server_host, self.run_paths.server_port))
            return result == 0

    def _send_command(self, command):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(self.command_timeout)
            connect_result = s.connect_ex((self.run_paths.server_host, self.run_paths.server_port))
            if connect_result != 0:
                return f"ERR connect_failed {connect_result}"
            if not command.endswith("\n"):
                command = command + "\n"
            s.sendall(command.encode("utf-8"))
            return self._read_line(s)

    def _read_line(self, sock):
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

    def _prepare_assets(self):
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

    def _render_init_tcl(self):
        session0 = self.run_paths.session_file(0)
        return textwrap.dedent(
            f"""\
            restore_session {{{session0}}}
            # set_host_options -max_cores 2;
            py_eval -file {{{self.server_py_path}}}
            """
        )

    def _render_end_tcl(self):
        finish_session = os.path.join(self.run_paths.run_dir, "finish_eco_session")
        return textwrap.dedent(
            f"""\
            save_session {finish_session}
            exit
            """
        )

    def _render_server_py(self):
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
                        return ""
                    buf += chunk
                    idx = buf.find(b"\\n")
                    if idx != -1:
                        line = buf[:idx]
                        return line.decode("utf-8", "replace")


            def handle_command(line):
                if len(line) > 0 and line[-1] == "\\r":
                    line = line[:-1]

                parts = line.split(" ")

                if len(parts) == 0 or parts[0] == "":
                    return "ERR empty command\\n"

                cmd = parts[0]

                if cmd == "PING":
                    return "OK PONG\\n"

                if cmd == "ECHO":
                    if len(line) < 6:
                        return "ERR ECHO requires text\\n"
                    return "OK " + line[5:] + "\\n"

                if cmd == "RUN":
                    if len(parts) < 2:
                        return "ERR RUN requires a command\\n"
                    command_to_run = " ".join(parts[1:])

                    snps.cmd.source(command_to_run)
                    return "OK command executed successfully\\n"

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

    def _render_client_py(self):
        host = self.run_paths.server_host
        port = self.run_paths.server_port
        return textwrap.dedent(
            f"""\
            # -*- coding: utf-8 -*-
            import socket


            def send_one(cmd, host="{host}", port={port}):
                if not cmd.endswith("\\n"):
                    cmd = cmd + "\\n"
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((host, port))
                    s.sendall(cmd.encode("utf-8"))
                    return s.recv(4096).decode("utf-8", "replace")


            if __name__ == "__main__":
                import sys

                if len(sys.argv) < 2:
                    print("usage: client.py <CMD>")
                    sys.exit(1)
                cmd = " ".join(sys.argv[1:])
                print(send_one(cmd).strip())
            """
        )
