"""`quanta serve` startup menu: CLI vs GUI."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from quanta import cli
from quanta.config import example_config


class StartupMenuTests(unittest.TestCase):
    def run_serve(self, argv, inputs=(), interactive=True):
        raw = example_config()
        raw["server"]["apiKey"] = "test-" + "k" * 24
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "q.json")
            path.write_text(json.dumps(raw), encoding="utf-8-sig")  # also proves BOM tolerance
            calls = {"ui": [], "open": [], "run": 0, "picked": 0}
            timer = MagicMock(side_effect=lambda delay, fn: MagicMock(start=lambda: fn()))
            with patch.object(sys, "argv", ["quanta", "serve", "-c", str(path), *argv]), \
                    patch("quanta.cli.stdin_is_interactive", return_value=interactive), \
                    patch("builtins.input", side_effect=list(inputs)), \
                    patch("uvicorn.run", side_effect=lambda *a, **k: calls.__setitem__("run", calls["run"] + 1)), \
                    patch("quanta.uiserver.start_ui_thread", side_effect=lambda *a, **k: calls["ui"].append(k)), \
                    patch("quanta.menu.SessionMenu.start"), \
                    patch("webbrowser.open", side_effect=lambda url: calls["open"].append(url)), \
                    patch("threading.Timer", timer), \
                    patch("quanta.picker.interactive_serve_selection",
                          side_effect=lambda cfg, **k: (calls.__setitem__("picked", calls["picked"] + 1), cfg)[1]):
                cli.main()
        return calls

    def test_menu_gui_starts_ui_and_opens_browser_with_key_fragment(self):
        calls = self.run_serve([], inputs=["2"])
        self.assertEqual(calls["run"], 1)
        self.assertEqual(calls["picked"], 0)
        self.assertEqual(len(calls["ui"]), 1)
        self.assertEqual(len(calls["open"]), 1)
        url = calls["open"][0]
        self.assertRegex(url, r"^http://127\.0\.0\.1:8788/#h=[A-Za-z0-9_-]{20,}$")
        self.assertNotIn("k" * 24, url)  # the API key is never in the URL
        self.assertEqual(calls["ui"][0]["handoff"].redeem(url.split("#h=")[1]), "test-" + "k" * 24)

    def test_menu_cli_uses_terminal_picker_and_no_browser(self):
        calls = self.run_serve([], inputs=["1"])
        self.assertEqual(calls["picked"], 1)
        self.assertEqual(calls["open"], [])
        self.assertEqual(calls["ui"], [])

    def test_no_browser_flag_prints_instead_of_opening(self):
        calls = self.run_serve(["--mode", "gui", "--no-browser"])
        self.assertEqual(calls["open"], [])
        self.assertEqual(len(calls["ui"]), 1)

    def test_explicit_provider_skips_menu(self):
        calls = self.run_serve(["--provider", "opencode", "--model", "x"], inputs=[])
        self.assertEqual(calls["picked"], 1)

    def test_non_interactive_skips_menu(self):
        calls = self.run_serve([], interactive=False)
        self.assertEqual((calls["picked"], calls["open"], calls["run"]), (0, [], 1))


class HandoffTests(unittest.TestCase):
    def test_single_use_expiry_and_wrong_guess(self):
        from quanta.uiserver import Handoff

        h = Handoff("secret-key", ttl=60)
        nonce = h.issue()
        self.assertIsNone(h.redeem("wrong"))
        self.assertIsNone(h.redeem(nonce))  # a wrong guess burns the nonce
        nonce = h.issue()
        self.assertEqual(h.redeem(nonce), "secret-key")
        self.assertIsNone(h.redeem(nonce))  # single use
        expired = Handoff("secret-key", ttl=-1)
        self.assertIsNone(expired.redeem(expired.issue()))

    def test_http_endpoint(self):
        import threading
        import urllib.error
        import urllib.request

        from quanta.uiserver import Handoff, make_ui_server

        handoff = Handoff("secret-key")
        server = make_ui_server("http://127.0.0.1:1", port=0, handoff=handoff)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            nonce = handoff.issue()
            self.assertEqual(json.load(urllib.request.urlopen(f"{base}/handoff?n={nonce}")), {"key": "secret-key"})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(f"{base}/handoff?n={nonce}")
            self.assertEqual(caught.exception.code, 404)
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
