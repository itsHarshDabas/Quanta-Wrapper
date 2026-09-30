"""Tests for the dynamic provider/model picker, mid-session menu and admin API."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from quanta.app import create_app
from quanta.config import example_config, validate_config
from quanta.menu import SessionMenu
from quanta.picker import (
    _suggest_model_id,
    compose_dynamic_model,
    pick_from_list,
)
from quanta.runner import RunResult
from tests.test_app import make_app, ok_runner


def config():
    raw = example_config()
    raw["server"]["apiKey"] = "test-" + "k" * 24
    return validate_config(raw, env={})


class PickerTests(unittest.TestCase):
    def test_compose_dynamic_model_enables_one_provider(self):
        selected, alias = compose_dynamic_model(config(), provider_name="opencode",
                                                upstream="opencode/some-model")
        self.assertEqual(alias, "opencode-opencode-some-model")
        self.assertTrue(selected.providers["opencode"].enabled)
        self.assertFalse(selected.providers["omnirush"].enabled)
        self.assertEqual(selected.models[0].upstream_model, "opencode/some-model")
        self.assertEqual(len(selected.exposed_models()), 1)

    def test_compose_rejects_unknown_provider_and_empty_model(self):
        with self.assertRaises(ValueError):
            compose_dynamic_model(config(), provider_name="nope", upstream="x")
        with self.assertRaises(ValueError):
            compose_dynamic_model(config(), provider_name="opencode", upstream="  ")
        with self.assertRaises(ValueError):
            compose_dynamic_model(config(), provider_name="freebuff", upstream="x")

    def test_custom_alias_and_slug(self):
        _, alias = compose_dynamic_model(config(), provider_name="cline",
                                         upstream="my model", alias="mine")
        self.assertEqual(alias, "mine")
        self.assertEqual(_suggest_model_id("cline", "a/b:c d")[:12], "cline-a-b-c-")

    def test_pick_from_list_repeats_until_valid(self):
        with patch("builtins.input", side_effect=["x", "0", "2"]):
            self.assertEqual(pick_from_list("Title", ["a", "b"]), 1)


    def test_persist_selection_keeps_secrets_and_server(self):
        import json
        import tempfile
        from pathlib import Path
        from quanta.picker import persist_selection

        raw = example_config()
        raw["server"]["apiKey"] = "quanta_" + "a" * 60
        raw["providers"]["opencode"]["args"] = ["--keep-me"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quanta.config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            selected, _ = compose_dynamic_model(config(), provider_name="opencode",
                                                upstream="opencode/m")
            persist_selection(path, selected)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["server"]["apiKey"], "quanta_" + "a" * 60)
        self.assertEqual(saved["providers"]["opencode"]["args"], ["--keep-me"])
        self.assertTrue(saved["providers"]["opencode"]["enabled"])
        self.assertFalse(saved["providers"]["omnirush"]["enabled"])
        self.assertEqual(saved["models"], [{"id": "opencode-opencode-m", "provider": "opencode",
                                            "upstreamModel": "opencode/m"}])


class SessionSwitchTests(unittest.TestCase):
    def setUp(self):
        self.config, app = make_app(runner=ok_runner())
        self.app = app
        self.client = TestClient(app, raise_server_exceptions=False)
        self.auth = {"Authorization": "Bearer " + self.config.api_key
                     if hasattr(self.config, "api_key") else "Bearer x",
                     "Content-Type": "application/json"}

    def test_session_info_and_switch_round_trip(self):
        key = self.client.app.state.config.server.api_key
        auth = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        info = self.client.get("/v1/session", headers=auth)
        self.assertEqual(info.status_code, 200)
        self.assertEqual(info.json()["serving"], "config file")
        switch = self.client.post("/v1/session/switch", headers=auth,
                                  json={"provider": "opencode", "model": "opencode/m"})
        self.assertEqual(switch.status_code, 200, switch.text)
        self.assertEqual(switch.json()["serving"], "opencode-opencode-m")
        models = self.client.get("/v1/models", headers=auth).json()["data"]
        self.assertEqual([m["id"] for m in models], ["opencode-opencode-m"])
        # Old alias is gone; new one completes.
        old = self.client.post("/v1/chat/completions", headers=auth,
                               json={"model": "omnirush",
                                     "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(old.status_code, 404)
        new = self.client.post("/v1/chat/completions", headers=auth,
                               json={"model": "opencode-opencode-m",
                                     "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(new.status_code, 200, new.text)

    def test_switch_rejects_bad_input(self):
        key = self.client.app.state.config.server.api_key
        auth = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        bad = self.client.post("/v1/session/switch", headers=auth,
                               json={"provider": "nope", "model": "x"})
        self.assertEqual(bad.status_code, 400)
        unauthed = self.client.post("/v1/session/switch",
                                    json={"provider": "opencode", "model": "x"})
        self.assertEqual(unauthed.status_code, 401)

    def test_menu_switch_applies_in_memory(self):
        _, app = make_app(runner=ok_runner())
        menu = SessionMenu(app, config())
        alias = menu.switch("cline", "upstream-x")
        self.assertEqual(alias, "cline-upstream-x")
        self.assertEqual(app.state.dynamic_label, alias)
        ids = [m.id for m in app.state.config.exposed_models()]
        self.assertEqual(ids, [alias])


if __name__ == "__main__":
    unittest.main()
