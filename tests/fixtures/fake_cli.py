"""Offline subprocess fixture. It never contacts a model or executes prompts."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

mode = sys.argv[1]
prompt = sys.stdin.buffer.read().decode("utf-8")
if mode == "echo":
    sys.stdout.buffer.write(prompt.encode("utf-8"))
elif mode == "args":
    sys.stdout.write(json.dumps(sys.argv[2:]))
elif mode == "utf8":
    for byte in "Hello 🌏 café".encode():
        sys.stdout.buffer.write(bytes([byte]))
        sys.stdout.buffer.flush()
elif mode == "fail":
    print("PRIVATE-CREDENTIAL", file=sys.stderr)
    sys.exit(7)
elif mode == "empty":
    pass
elif mode == "malformed":
    print("not json")
elif mode == "overflow":
    print("x" * 10000)
elif mode == "sleep":
    time.sleep(60)
elif mode == "tree":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    Path(os.environ["PID_FILE"]).write_text(f"{os.getpid()},{child.pid}")
    print("started", flush=True)
    time.sleep(60)
elif mode == "jsonl":
    print(json.dumps({"type": "text", "text": "hello"}), flush=True)
    print(json.dumps({"type": "done"}), flush=True)
elif mode == "partial_error":
    print(json.dumps({"type": "text", "text": "partial"}), flush=True)
    time.sleep(0.05)
    print(json.dumps({"type": "error", "message": "PRIVATE-CREDENTIAL"}), flush=True)
elif mode == "env":
    print(json.dumps({"key_present": "GLOBAL_API_KEY" in os.environ, "cwd": os.getcwd(), "pwd": os.environ.get("PWD")}))
