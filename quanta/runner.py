from __future__ import annotations

import asyncio
import codecs
import os
import re
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from .adapters import Decoder, build_invocation
from .commands import resolve_command
from .config import ModelAlias, Provider
from .errors import ApiError


@dataclass
class RunResult:
    text: str
    usage: dict | None = None


_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_SECRET = re.compile(r"(?i)(bearer\s+\S+|(?:sk|gaw|quanta|key|token)[-_][A-Za-z0-9_\-]{8,}|[A-Za-z0-9_\-]{32,})")


def diagnostic_tail(stderr: bytes, stdout: bytes = b"", limit: int = 500) -> str:
    """Short, redacted tail of a failed CLI's output for the operator's console.

    Never sent to API clients: stderr can contain prompts and credentials. Long tokens and
    key-looking strings are masked, colour codes removed and the text truncated.
    """
    raw = stderr if stderr.strip() else stdout
    text = _ANSI.sub("", raw.decode("utf-8", "replace"))
    text = _SECRET.sub("[redacted]", " ".join(text.split()))
    return text[-limit:]


async def terminate_tree(process: asyncio.subprocess.Process):
    if process.pid is None:
        return
    if os.name == "nt":
        # The npm launcher may have a Bun/native grandchild: killing only the
        # immediate Python child would leave generation running in the background.
        taskkill = str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe")
        try:
            killer = await asyncio.create_subprocess_exec(taskkill, "/PID", str(process.pid), "/T", "/F",
                                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                                          creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                await asyncio.wait_for(killer.wait(), 5)
            except TimeoutError:
                killer.kill()
                await killer.wait()
        except (OSError, ProcessLookupError):
            pass
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(process.wait(), 5)
    except TimeoutError:
        pass


async def run_cli(provider: Provider, model: ModelAlias, prompt: str, *,
                  on_text: Callable[[str], Awaitable[None]] | None = None,
                  secret_env: str = "QUANTA_API_KEY") -> RunResult:
    env = {**os.environ, **provider.env, "PWD": provider.workspace, "NO_COLOR": "1", "FORCE_COLOR": "0", "TERM": "dumb"}
    env.pop(secret_env, None)
    try:
        resolved = resolve_command(provider.command, env)
    except (ValueError, OSError):
        raise ApiError(503, "cli_unavailable", "CLI executable is unavailable. Run quanta doctor on the server.") from None
    Path(provider.workspace).mkdir(parents=True, exist_ok=True)
    state_dir = tempfile.mkdtemp(prefix="quanta-")
    process = None
    tasks: list[asyncio.Task] = []
    try:
        invocation = build_invocation(provider, model, prompt, state_dir)
        entrypoint = Path(resolved.entrypoint) if resolved.entrypoint else None
        # Explicitly load only OmniRush's trusted provider integration for device
        # token refresh; leave general extension discovery and local tools disabled.
        if provider.adapter == "omnirush" and entrypoint and entrypoint.as_posix().endswith("/omnirush/src/bin.js"):
            extension = entrypoint.parent.parent / "assets" / "extensions" / "omnirush" / "sota.ts"
            if extension.is_file():
                invocation.args.extend(["--extension", str(extension)])
        args = [resolved.executable, *resolved.prefix, *invocation.args]
        if os.name == "nt" and len(subprocess.list2cmdline(args)) > 30000:
            raise ApiError(413, "prompt_too_large", "Windows command-line length exceeded; use a stdin-based custom bridge.")
        decoder_mode = provider.output_mode if provider.adapter == "custom" else provider.adapter
        if decoder_mode == "opencode":
            decoder_mode = "text"
        decoder = Decoder(decoder_mode)
        byte_count = 0

        def count(chunk):
            nonlocal byte_count
            byte_count += len(chunk)
            if byte_count > provider.max_output_bytes:
                raise ApiError(502, "output_limit", "CLI output exceeded the configured limit.")

        async def output():
            utf8 = codecs.getincrementaldecoder("utf-8")("strict")
            while chunk := await process.stdout.read(16384):
                count(chunk)
                stdout_tail.extend(chunk)
                del stdout_tail[:-4096]
                for text in decoder.feed(utf8.decode(chunk)):
                    if on_text:
                        await on_text(text)
            for text in decoder.feed(utf8.decode(b"", final=True)):
                if on_text:
                    await on_text(text)

        stderr_tail = bytearray()
        stdout_tail = bytearray()

        async def diagnostics():
            while chunk := await process.stderr.read(16384):
                count(chunk)
                stderr_tail.extend(chunk)
                del stderr_tail[:-4096]
            # The tail is only used for the operator-facing `detail` below, never in API responses.

        async def input_data():
            try:
                data = invocation.stdin.encode("utf-8")
                for index in range(0, len(data), 65536):
                    process.stdin.write(data[index:index + 65536])
                    await process.stdin.drain()
                process.stdin.close()
                await process.stdin.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass

        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        async with asyncio.timeout(provider.timeout_ms / 1000):
            try:
                process = await asyncio.create_subprocess_exec(*args, cwd=provider.workspace,
                    env={**env, **invocation.env}, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options)
            except (OSError, ValueError):
                raise ApiError(503, "cli_unavailable", "Unable to start the CLI. Check installation and server permissions.") from None
            tasks = [asyncio.create_task(coro) for coro in (input_data(), output(), diagnostics(), process.wait())]
            await asyncio.gather(*tasks)
            if process.returncode != 0:
                raise ApiError(502, "cli_failed", "CLI exited unsuccessfully. Check login, model configuration and permissions locally.",
                               detail=f"exit code {process.returncode}: {diagnostic_tail(bytes(stderr_tail), bytes(stdout_tail))}")
            for text in decoder.finish():
                if on_text:
                    await on_text(text)
            return RunResult(decoder.text, decoder.usage)
    except TimeoutError:
        raise ApiError(504, "cli_timeout", "The CLI exceeded its request time limit.") from None
    except UnicodeError:
        raise ApiError(502, "cli_protocol_error", "CLI output was not valid UTF-8.") from None
    finally:
        # Complete tree cleanup even when cancellation came from a disconnected
        # HTTP client. Drain tasks are cancelled only after killing descendants.
        if process and (process.returncode is None or any(not task.done() for task in tasks)):
            cleanup = asyncio.create_task(terminate_tree(process))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.to_thread(shutil.rmtree, state_dir, True)
