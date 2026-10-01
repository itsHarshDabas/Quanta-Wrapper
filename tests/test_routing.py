"""Per-request provider:model routing, provider capabilities and adapter fixes."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from quanta.adapters import Decoder, build_invocation, effective_model
from quanta.app import create_app
from quanta.config import ModelAlias, example_config, validate_config
from quanta.errors import ApiError
from quanta.picker import pick_from_list
from quanta.runner import RunResult


def build(enabled=("omnirush", "opencode", "cline")):
    raw = example_config()
    raw["server"]["apiKey"] = "test-" + "k" * 24
    for name in enabled:
        raw["providers"][name]["enabled"] = True
    config = validate_config(raw, env={})
    seen = []

    async def runner(provider, model, prompt, **kwargs):
        seen.append((model.provider, model.upstream_model, model.id))
        return RunResult("ok")

    client = TestClient(create_app(config, runner=runner), raise_server_exceptions=False)
    headers = {"Authorization": "Bearer " + config.server.api_key, "Content-Type": "application/json"}
    return config, client, headers, seen


def ask(client, headers, model):
    return client.post("/v1/chat/completions", headers=headers,
                       json={"model": model, "messages": [{"role": "user", "content": "hi"}]})


class AdHocRoutingTests(unittest.TestCase):
    def test_provider_model_syntax_routes_to_enabled_provider(self):
        _, client, headers, seen = build()
        for model in ("opencode:opencode/big-pickle", "cline:default", "omnirush:openrouter/~x/y:free"):
            self.assertEqual(ask(client, headers, model).status_code, 200, model)
        self.assertEqual(seen[0][:2], ("opencode", "opencode/big-pickle"))
        self.assertEqual(seen[1][:2], ("cline", "default"))
        # only the first colon separates provider from upstream id
        self.assertEqual(seen[2][:2], ("omnirush", "openrouter/~x/y:free"))

    def test_response_echoes_requested_model(self):
        _, client, headers, _ = build()
        self.assertEqual(ask(client, headers, "opencode:a/b").json()["model"], "opencode:a/b")

    def test_switching_between_providers_needs_no_server_state(self):
        _, client, headers, seen = build()
        for model in ("opencode:a/b", "cline:default", "opencode:c/d", "omnirush:m"):
            self.assertEqual(ask(client, headers, model).status_code, 200)
        self.assertEqual([s[0] for s in seen], ["opencode", "cline", "opencode", "omnirush"])

    def test_disabled_unknown_and_bridge_providers_are_404(self):
        _, client, headers, seen = build(enabled=("omnirush",))
        for model in ("opencode:a/b", "nope:x", "freebuff:x", "omnirush:", "omnirush:  ", "omnirush"[:0] + "plain"):
            response = ask(client, headers, model)
            self.assertEqual(response.status_code, 404, model)
            self.assertEqual(response.json()["error"]["code"], "model_not_found")
        self.assertEqual(seen, [])

    def test_adhoc_routing_still_requires_auth(self):
        _, client, _, seen = build()
        response = ask(client, {"Authorization": "Bearer wrong", "Content-Type": "application/json"}, "opencode:a/b")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(seen, [])

    def test_base_config_keeps_routing_after_dynamic_narrowing(self):
        from quanta.picker import compose_dynamic_model

        config, _, _, _ = build()
        narrowed, _ = compose_dynamic_model(config, provider_name="opencode", upstream="opencode/x")

        async def runner(provider, model, prompt, **kwargs):
            return RunResult("ok")

        client = TestClient(create_app(narrowed, runner=runner, base_config=config), raise_server_exceptions=False)
        headers = {"Authorization": "Bearer " + config.server.api_key, "Content-Type": "application/json"}
        self.assertEqual(ask(client, headers, "cline:default").status_code, 200)
        self.assertEqual(ask(client, headers, "opencode-opencode-x").status_code, 200)


class PreflightTests(unittest.TestCase):
    def test_cors_preflight_has_no_body(self):
        """Regression: a 204 carrying a body crashed older Starlette/h11 ("Too much data for declared Content-Length")."""
        raw = example_config()
        raw["server"]["apiKey"] = "test-" + "k" * 24
        raw["server"]["corsOrigins"] = ["http://127.0.0.1:8788"]
        client = TestClient(create_app(validate_config(raw, env={}), runner=lambda *a, **k: None), raise_server_exceptions=False)
        response = client.options("/v1/models", headers={"Origin": "http://127.0.0.1:8788", "Access-Control-Request-Method": "GET"})
        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.content, b"")
        self.assertEqual(response.headers["access-control-allow-origin"], "http://127.0.0.1:8788")
        self.assertIn("Authorization", response.headers["access-control-allow-headers"])
        blocked = client.options("/v1/models", headers={"Origin": "http://evil.example"})
        self.assertEqual(blocked.status_code, 403)


class ProvidersEndpointTests(unittest.TestCase):
    def test_lists_capabilities_honestly(self):
        _, client, headers, _ = build(enabled=("omnirush",))
        assert client.get("/v1/providers").status_code == 401
        data = {p["id"]: p for p in client.get("/v1/providers", headers=headers).json()["data"]}
        self.assertTrue(data["omnirush"]["enabled"])
        self.assertFalse(data["opencode"]["enabled"])
        caps = data["omnirush"]["capabilities"]
        self.assertEqual(caps["subagents"], False)
        self.assertEqual(caps["tool_calling"], "client-side prompt bridge")
        self.assertTrue(caps["provider_switching"] and caps["model_switching"])
        self.assertFalse(data["freebuff"]["available"])
        self.assertFalse(any(data["freebuff"]["capabilities"].values()))
        self.assertIn("interactive", data["freebuff"]["note"])


class AdapterTests(unittest.TestCase):
    def provider(self, name):
        return validate_config({**example_config(), "server": {"apiKey": "k" * 30}}, env={}).providers[name]

    def test_default_model_uses_cli_default(self):
        self.assertIsNone(effective_model(ModelAlias(id="a", provider="cline", upstream_model="Default")))
        self.assertIsNone(effective_model(ModelAlias(id="a", provider="cline")))
        self.assertEqual(effective_model(ModelAlias(id="a", provider="cline", upstream_model="m")), "m")
        invocation = build_invocation(self.provider("cline"), ModelAlias(id="a", provider="cline", upstream_model="default"), "hi", "/tmp")
        self.assertNotIn("--model", invocation.args)

    def test_cline_uses_its_own_credentials_dir(self):
        invocation = build_invocation(self.provider("cline"), ModelAlias(id="a", provider="cline", upstream_model="m"), "hi", "/tmp/x")
        self.assertNotIn("--data-dir", invocation.args)
        self.assertEqual(invocation.args[-1], "hi")

    def test_antigravity_runs_read_only(self):
        invocation = build_invocation(self.provider("antigravity"), ModelAlias(id="a", provider="antigravity", upstream_model="m"), "hi", "/tmp")
        self.assertIn("plan", invocation.args[invocation.args.index("--mode") + 1:][:1])

    def test_upstream_error_detail_is_surfaced_and_429_mapped(self):
        decoder = Decoder("omnirush")
        line = '{"type":"message_end","message":{"role":"assistant","content":[],"stopReason":"error","errorMessage":"omnirush API error (429): 429 Omnirush grant exhausted"}}\n'
        with self.assertRaises(ApiError) as caught:
            decoder.feed(line)
        self.assertEqual(caught.exception.status, 429)
        self.assertIn("grant exhausted", caught.exception.message)

    def test_cline_error_detail_is_surfaced(self):
        decoder = Decoder("cline")
        with self.assertRaises(ApiError) as caught:
            decoder.feed('{"type":"error","message":"Unauthorized: re-authenticate"}\n')
        self.assertEqual(caught.exception.status, 502)
        self.assertIn("Unauthorized", caught.exception.message)


class PickerInputTests(unittest.TestCase):
    def test_closed_stdin_does_not_loop_forever(self):
        with patch("builtins.input", side_effect=EOFError):
            with self.assertRaises(ValueError):
                pick_from_list("Title", ["a", "b"])

    def test_omnirush_models_parsed_from_list_models(self):
        import subprocess
        from quanta import picker

        table = "provider    model   context\nomnirush    muse-spark-1.3    400K\nopenrouter  ~x/y-latest   1M\n"
        done = subprocess.CompletedProcess([], 0, stdout=table, stderr="")
        with patch("shutil.which", return_value="omnirush"), patch("subprocess.run", return_value=done):
            self.assertEqual(picker._provider_models("omnirush"), ["muse-spark-1.3", "openrouter/~x/y-latest"])


if __name__ == "__main__":
    unittest.main()
