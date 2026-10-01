"""OpenCode 1.x vs 2.x: pass only the flags the installed CLI lists in `run --help`."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quanta import probe
from quanta.adapters import build_invocation
from quanta.commands import ResolvedCommand
from quanta.config import ModelAlias, Provider
from quanta.runner import run_cli

FAKE = str(Path(__file__).parent / "fixtures" / "fake_opencode.py")
MODEL = ModelAlias(id="oc", provider="oc", upstream_model="opencode/big-pickle")


class ParseTests(unittest.TestCase):
    def test_parse_flags_ignores_prose_and_single_dashes(self):
        text = "opencode run [message..]\n  -m, --model  model to use (provider/model)\n  --pure   skip plugins\n  non--flag  x"
        self.assertEqual(probe.parse_flags(text), {"--model", "--pure"})

    def test_unparseable_help_gives_no_capabilities_and_is_not_cached(self):
        probe._cache.clear()
        with patch("quanta.probe.subprocess.run") as run:
            run.return_value.stdout, run.return_value.stderr = "", "Error: something broke"
            self.assertEqual(probe.supported_flags(sys.executable, (FAKE,)), frozenset())
            self.assertEqual(probe.supported_flags(sys.executable, (FAKE,)), frozenset())
            self.assertEqual(run.call_count, 2)  # failures are retried, never cached

    def test_success_is_cached(self):
        probe._cache.clear()
        with patch("quanta.probe.subprocess.run") as run:
            run.return_value.stdout, run.return_value.stderr = "--model\n--pure\n--dir\n", ""
            first = probe.supported_flags(sys.executable, (FAKE,))
            second = probe.supported_flags(sys.executable, (FAKE,))
        self.assertEqual(first, second)
        self.assertEqual(run.call_count, 1)


class InvocationTests(unittest.TestCase):
    provider = Provider(adapter="opencode", command="opencode")

    def args(self, flags):
        return build_invocation(self.provider, MODEL, "p", "state", flags).args

    def test_pure_only_when_listed_and_dir_never(self):
        self.assertIn("--pure", self.args(frozenset({"--pure", "--dir", "--model"})))
        self.assertNotIn("--pure", self.args(frozenset({"--model", "--auto"})))   # 2.x
        self.assertNotIn("--pure", self.args(None))                              # unknown: be conservative
        for flags in (frozenset({"--pure", "--dir"}), frozenset(), None):
            self.assertNotIn("--dir", self.args(flags))
        self.assertEqual(self.args(frozenset({"--pure"}))[-2:], ["--model", "opencode/big-pickle"])


class EndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def run_as(self, version):
        probe._cache.clear()
        with tempfile.TemporaryDirectory(prefix="oc compat ") as workspace:
            provider = Provider(adapter="opencode", command="opencode", enabled=True, workspace=workspace,
                                env={"FAKE_OC": version})
            with patch("quanta.runner.resolve_command", return_value=ResolvedCommand(sys.executable, (FAKE,))):
                result = await run_cli(provider, MODEL, "hello")
            return json.loads(result.text), workspace

    async def test_v1_gets_pure_v2_does_not_and_both_succeed(self):
        v1, workspace = await self.run_as("v1")
        self.assertIn("--pure", v1["args"])
        v2, _ = await self.run_as("v2")  # a strict 2.x would exit 1 on --pure; success proves it was not sent
        self.assertNotIn("--pure", v2["args"])
        for seen in (v1, v2):
            self.assertNotIn("--dir", seen["args"])
            self.assertEqual(seen["args"][-2:], ["--model", "opencode/big-pickle"])
            self.assertNotEqual(Path(seen["cwd"]), Path(workspace))  # isolated working directory

    async def test_unavailable_cli_does_not_leak_a_temp_dir(self):
        provider = Provider(adapter="opencode", command="opencode", enabled=True)
        with patch("quanta.runner.resolve_command", side_effect=ValueError("nope")), \
                patch("quanta.runner.tempfile.mkdtemp") as mkdtemp:
            with self.assertRaises(Exception):
                await run_cli(provider, MODEL, "hi")
        mkdtemp.assert_not_called()


if __name__ == "__main__":
    unittest.main()
