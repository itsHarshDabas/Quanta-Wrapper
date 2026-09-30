"""Reliable interactive-console detection.

On Windows ``os.isatty`` reports True for the NUL device (``< NUL``,
backgrounded shells), which made ``serve`` start an interactive picker that
could never be answered. A real console must also pass ``GetConsoleMode``.
"""

from __future__ import annotations

import os
import sys


def stdin_is_interactive() -> bool:
    try:
        if not os.isatty(0):
            return False
    except OSError:
        return False
    if os.name == "nt":
        try:
            import ctypes
            mode = ctypes.c_uint32()
            handle = ctypes.windll.kernel32.GetStdHandle(-10)
            return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)))
        except (AttributeError, OSError):
            return False
    return sys.stdin is not None and sys.stdin.isatty()
