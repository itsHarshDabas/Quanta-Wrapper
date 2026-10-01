"""Offline stand-in for the OpenCode CLI. FAKE_OC=v1|v2 picks which flags `run` accepts.

Like the real thing it rejects unknown flags (exit 1) and reads the prompt from stdin.
"""
import json
import os
import sys

V1 = ["--agent", "--attach", "--command", "--continue", "--dir", "--file", "--format", "--help", "--model", "--pure", "--session", "--thinking"]
V2 = ["--agent", "--auto", "--continue", "--file", "--fork", "--format", "--help", "--model", "--session", "--standalone", "--server", "--thinking", "--title"]
allowed = V2 if os.environ.get("FAKE_OC") == "v2" else V1
args = sys.argv[1:]
if args[:1] != ["run"]:
    print("usage: opencode run [message..]", file=sys.stderr)
    sys.exit(2)
if "--help" in args:
    print("opencode run [message..]\n\nOptions:\n" + "\n".join(f"      {flag}  option" for flag in allowed))
    sys.exit(0)
unknown = [a for a in args[1:] if a.startswith("--") and a not in allowed]
if unknown:
    print("ERRORS " + " ".join(f"Unrecognized flag: {a} in command opencode run" for a in unknown), file=sys.stderr)
    sys.exit(1)
prompt = sys.stdin.read()
print(json.dumps({"args": args[1:], "cwd": os.getcwd(), "prompt_len": len(prompt)}))
