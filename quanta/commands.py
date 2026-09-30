from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResolvedCommand:
    executable: str
    prefix: tuple[str, ...] = ()
    entrypoint: str | None = None


def unwrap_windows_shim(filename: Path, node: str | None) -> ResolvedCommand:
    source = filename.read_text(encoding="utf-8-sig")
    native = re.search(r'^"%dp0%[\\/]([^"\r\n]+\.exe)"\s+%\*', source, re.I | re.M)
    if native:
        target = (filename.parent / native[1]).resolve()
        if not target.is_file():
            raise ValueError("CLI shim target is missing")
        return ResolvedCommand(str(target), entrypoint=str(target))
    match = re.search(r'"%_prog%"\s+"%dp0%[\\/]([^"\r\n]+)"\s+%\*', source, re.I)
    if match and re.search(r'SET\s+"_prog=node"', source, re.I) and node:
        target = (filename.parent / match[1]).resolve()
        if not target.is_file():
            raise ValueError("CLI shim entrypoint is missing")
        return ResolvedCommand(node, (str(target),), str(target))
    raise ValueError("Unsupported Windows shell shim; configure a native executable, or node/python with an absolute entrypoint in args")


def resolve_command(command: str, env: dict[str, str] | None = None) -> ResolvedCommand:
    env = os.environ if env is None else env
    search_path = next((v for k, v in env.items() if k.lower() == "path"), "")
    executable = shutil.which(command, path=search_path)
    if not executable:
        raise ValueError(f"Executable not found: {command}")
    filename = Path(executable).resolve()
    if os.name == "nt":
        suffix = filename.suffix.lower()
        node = shutil.which("node", path=search_path)
        if suffix in (".cmd", ".bat"):
            return unwrap_windows_shim(filename, node)
        if suffix in (".js", ".mjs", ".cjs") and node:
            return ResolvedCommand(node, (str(filename),), str(filename))
        if suffix not in (".exe", ".com"):
            raise ValueError("Windows command must resolve to a native executable or standard npm shim")
    return ResolvedCommand(str(filename), entrypoint=str(filename))
