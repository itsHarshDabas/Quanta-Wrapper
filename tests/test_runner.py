import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from global_api_wrapper.adapters import Decoder, build_invocation
from global_api_wrapper.commands import resolve_command, unwrap_windows_shim
from global_api_wrapper.config import ModelAlias, Provider, example_config, validate_config
from global_api_wrapper.errors import ApiError
from global_api_wrapper.runner import run_cli

FIXTURE = str(Path(__file__).parent / "fixtures" / "fake_cli.py")


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="wrapper test ")
        self.model = ModelAlias(id="test", provider="test")

    async def asyncTearDown(self):
        self.temp.cleanup()

    def provider(self, mode, **kwargs):
        return Provider(adapter="custom", command=sys.executable, args=[FIXTURE, mode], enabled=True,
                        workspace=self.temp.name, prompt_mode="stdin", output_mode="text", **kwargs)

    async def test_stdin_preserves_shell_metacharacters_and_large_input(self):
        prompt = '& echo ATTACK && $(whoami) %PATH% "quoted"\n' + "long text " * 10000
        result = await run_cli(self.provider("echo"), self.model, prompt)
        self.assertEqual(result.text, prompt)

    async def test_split_utf8(self):
        result = await run_cli(self.provider("utf8"), self.model, "hi")
        self.assertEqual(result.text, "Hello 🌏 café")

    async def test_argument_prompt_is_not_shell_interpreted(self):
        p = self.provider("args")
        p.prompt_mode = "arg"
        p.args.append("{prompt}")
        text = 'hi " & touch injected ; $(id)'
        self.assertEqual(json.loads((await run_cli(p, self.model, text)).text), [text])

    async def test_structured_custom_output(self):
        p = self.provider("jsonl")
        p.output_mode = "jsonl"
        parts = []
        async def collect(text):
            parts.append(text)
        result = await run_cli(p, self.model, "hi", on_text=collect)
        self.assertEqual(result.text, "hello")
        self.assertEqual(parts, ["hello"])

    async def test_failures_are_sanitized(self):
        for mode, code in [("fail", "cli_failed"), ("empty", "empty_response"), ("overflow", "output_limit")]:
            with self.subTest(mode=mode):
                with self.assertRaises(ApiError) as error:
                    await run_cli(self.provider(mode, max_output_bytes=1024), self.model, "hi")
                self.assertEqual(error.exception.code, code)
                self.assertNotIn("PRIVATE-CREDENTIAL", str(error.exception))

    async def test_timeout_and_cancellation(self):
        started = time.monotonic()
        with self.assertRaises(ApiError) as error:
            await run_cli(self.provider("sleep", timeout_ms=200), self.model, "hi")
        self.assertEqual(error.exception.status, 504)
        self.assertLess(time.monotonic() - started, 8)
        task = asyncio.create_task(run_cli(self.provider("sleep"), self.model, "hi"))
        await asyncio.sleep(0.2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_cancellation_kills_descendants(self):
        p = self.provider("tree")
        pid_file = Path(self.temp.name) / "pids"
        p.env = {"PID_FILE": str(pid_file)}
        ready = asyncio.Event()
        async def output(_):
            ready.set()
        task = asyncio.create_task(run_cli(p, self.model, "hi", on_text=output))
        await asyncio.wait_for(ready.wait(), 5)
        pids = list(map(int, pid_file.read_text().split(",")))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            for pid in pids:
                handle = kernel.OpenProcess(0x00100000, False, pid)
                if handle:
                    self.assertEqual(kernel.WaitForSingleObject(handle, 0), 0, f"Process {pid} is still alive")
                    kernel.CloseHandle(handle)
        else:
            # Some init systems reap descendants asynchronously; no process
            # should be executing even if a short-lived zombie remains.
            for pid in pids:
                status = Path(f"/proc/{pid}/stat")
                if status.exists():
                    self.assertEqual(status.read_text().split()[2], "Z")

    async def test_missing_executable(self):
        p = self.provider("echo")
        p.command = "nonexistent-global-api-wrapper-test-command-987"
        with self.assertRaises(ApiError) as error:
            await run_cli(p, self.model, "hi")
        self.assertEqual(error.exception.status, 503)

    async def test_wrapper_secret_not_in_child_environment(self):
        with patch.dict(os.environ, {"GLOBAL_API_KEY": "private-key"}):
            result = json.loads((await run_cli(self.provider("env"), self.model, "hi")).text)
        self.assertFalse(result["key_present"])
        self.assertEqual(Path(result["cwd"]), Path(self.temp.name))
        self.assertEqual(result["pwd"], self.temp.name)


class DecoderTests(unittest.TestCase):
    def decode(self, mode, events):
        decoder = Decoder(mode)
        wire = "\r\n".join(json.dumps(e) for e in events)
        for character in wire:
            decoder.feed(character)
        decoder.finish()
        return decoder

    def test_omnirush_dedup_and_usage(self):
        d = self.decode("omnirush", [
            {"type": "message_start", "message": {"role": "assistant"}},
            {"type": "message_update", "assistantMessageEvent": {"type": "thinking_delta", "delta": "PRIVATE"}},
            {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "hello"}},
            {"type": "message_end", "message": {"role": "assistant", "content": [{"type": "text", "text": "hello"}], "usage": {"input": 2, "output": 1, "cacheRead": 3}}},
            {"type": "agent_end"},
        ])
        self.assertEqual(d.text, "hello")
        self.assertEqual(d.usage["total_tokens"], 6)

    def test_omnirush_success_exit_with_error_message_is_failure(self):
        with self.assertRaises(ApiError):
            self.decode("omnirush", [{"type": "message_end", "message": {"role": "assistant", "stopReason": "error"}}])

    def test_opencode_filters_tool_events(self):
        d = self.decode("opencode", [{"type": "tool_use", "part": {"text": "SECRET"}}, {"type": "text", "part": {"text": "hello"}}, {"type": "step_finish"}])
        self.assertEqual(d.text, "hello")

    def test_cline_authoritative_result_and_legacy_snapshots(self):
        d = self.decode("cline", [{"type": "agent_event", "event": {"text": "narration"}}, {"type": "run_result", "finishReason": "completed", "text": "answer"}])
        self.assertEqual(d.text, "answer")
        d = self.decode("cline", [{"type": "say", "say": "text", "ts": 1, "text": "ans"}, {"type": "say", "say": "text", "ts": 1, "text": "answer"}])
        self.assertEqual(d.text, "answer")

    def test_antigravity_result_dedup_and_status(self):
        events = [{"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": "hello"}},
                  {"event": "result", "result": {"status": "SUCCESS", "response": "hello"}}]
        self.assertEqual(self.decode("antigravity", events).text, "hello")
        events[-1]["result"]["status"] = "WAITING"
        with self.assertRaises(ApiError):
            self.decode("antigravity", events)

    def test_missing_completion_and_malformed_output(self):
        with self.assertRaises(ApiError):
            self.decode("jsonl", [{"type": "text", "text": "partial"}])
        with self.assertRaises(ApiError):
            Decoder("opencode").feed("not JSON\n")


class ConfigurationTests(unittest.TestCase):
    def config(self):
        raw = example_config()
        raw["server"]["apiKey"] = "test-" + "a" * 32
        return raw

    def test_config_defaults_and_secret_override(self):
        config = validate_config(self.config(), env={"GLOBAL_API_KEY": "b" * 32})
        self.assertEqual(config.server.api_key, "b" * 32)
        self.assertEqual(config.providers["omnirush"].max_concurrent, 1)

    def test_invalid_keys_and_unsupported_native_freebuff(self):
        for field, value in [("apiKey", "short"), ("port", -1), ("corsOrigins", ["*"]), ("maxConcurrent", 0)]:
            raw = self.config()
            raw["server"][field] = value
            with self.assertRaises(ValueError):
                validate_config(raw, env={})
        raw = self.config()
        raw["providers"]["freebuff"]["enabled"] = True
        with self.assertRaisesRegex(ValueError, "headless"):
            validate_config(raw, env={})

    def test_antigravity_requires_explicit_risk_acknowledgement(self):
        raw = self.config()
        raw["providers"]["antigravity"]["enabled"] = True
        with self.assertRaisesRegex(ValueError, "deny permissions"):
            validate_config(raw, env={})

    def test_python_executable_resolution(self):
        self.assertTrue(Path(resolve_command(sys.executable).executable).is_file())

    def test_native_presets_use_stdin_and_disable_local_tools(self):
        model = ModelAlias(id="test", provider="test", upstream_model="upstream")
        for adapter in ("omnirush", "opencode", "cline", "antigravity"):
            p = Provider(adapter=adapter, command=adapter)
            invocation = build_invocation(p, model, "PROMPT", "state")
            self.assertNotIn("PROMPT", invocation.args)
            self.assertIn("PROMPT", invocation.stdin)
            self.assertIn("--model", invocation.args)
            self.assertNotIn("--dangerously-skip-permissions", invocation.args)

    @unittest.skipUnless(os.name == "nt", "Windows shim parser")
    def test_windows_unknown_batch_script_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            shim = Path(directory) / "unknown.cmd"
            shim.write_text("@echo off\necho %*\n")
            with self.assertRaisesRegex(ValueError, "Unsupported Windows shell shim"):
                unwrap_windows_shim(shim, sys.executable)
