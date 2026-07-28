# -*- coding: utf-8 -*-
import socket


def send_one(cmd, host="127.0.0.1", port=9009):
    if not cmd.endswith("\n"):
        cmd = cmd + "\n"
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
