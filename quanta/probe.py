"""Detect which command-line flags an installed CLI supports.

OpenCode 1.x and 2.x differ (`run --pure` / `--dir` exist only in 1.x), and other CLIs change
too. Rather than pin one version, ask the installed binary once (`<cli> run --help`) and cache
the answer, so an unknown or older/newer CLI never receives a flag it would reject.
"""

from __future__ import annotations

import os
import re
import subprocess

_FLAG = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]*")
_cache: dict[tuple, frozenset[str]] = {}


def parse_flags(help_text: str) -> frozenset[str]:
    return frozenset(_FLAG.findall(help_text or ""))


def supported_flags(executable: str, prefix: tuple[str, ...] = (), subcommand: tuple[str, ...] = ("run", "--help"),
                    cwd: str | None = None, timeout: float = 20.0,
                    env: dict[str, str] | None = None) -> frozenset[str]:
    """Flags listed by ``<cli> <subcommand>``; empty when they cannot be determined (callers then stay conservative)."""
    try:
        stamp = os.stat(executable).st_mtime_ns
    except OSError:
        stamp = 0
    key = (executable, tuple(prefix), tuple(subcommand), stamp)
    if key in _cache:
        return _cache[key]
    flags: frozenset[str] = frozenset()
    try:
        done = subprocess.run([executable, *prefix, *subcommand], capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL, cwd=cwd, env=env, errors="replace",
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        parsed = parse_flags((done.stdout or "") + "\n" + (done.stderr or ""))
        # A real help screen lists several options; anything less is an error message, not a capability list.
        flags = parsed if len(parsed) >= 3 else frozenset()
    except (OSError, subprocess.SubprocessError):
        flags = frozenset()
    if flags:  # do not cache failures: a transient timeout should be retried next request
        _cache[key] = flags
    return flags
