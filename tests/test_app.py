"""HTTP-level tests for the FastAPI app using an offline fake runner."""

from __future__ import annotations

import json
import unittest

from fastapi.testclient import TestClient

from quanta.app import create_app
from quanta.config import example_config, validate_config
from quanta.errors import ApiError
from quanta.runner import RunResult


def make_app(runner=None, **overrides):
    raw = example_config()
    raw["server"]["apiKey"] = "test-" + "k" * 24
    raw["server"].update(overrides.pop("server", {}))
    for key, value in overrides.items():
        raw[key] = value
    config = validate_config(raw, env={})
    return config, create_app(config, runner=runner or (lambda *a, **k: None))


def ok_runner(text="hello", usage=None):
    async def run(provider, model, prompt, **kwargs):
        on_text = kwargs.get("on_text")
        if on_text:
            await on_text(text)
        return RunResult(text, usage)
    return run


class AppTests(unittest.TestCase):
    def setUp(self):
        self.config, app = make_app(runner=ok_runner())
        self.client = TestClient(app, raise_server_exceptions=False)
        self.auth = {"Authorization": "Bearer " + self.config.server.api_key,
                     "Content-Type": "application/json"}

    def post(self, body, headers=None):
        return self.client.post("/v1/chat/completions", json=body,
                                headers=headers or self.auth)

    def test_health_needs_no_auth(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_models_requires_bearer(self):
        self.assertEqual(self.client.get("/v1/models").status_code, 401)
        response = self.client.get("/v1/models", headers=self.auth)
        self.assertEqual(response.status_code, 200)
        ids = [m["id"] for m in response.json()["data"]]
        self.assertIn("omnirush", ids)
        self.assertNotIn("opencode", ids)

    def test_wrong_key_rejected_and_request_id_present(self):
        bad = {"Authorization": "Bearer wrong", "Content-Type": "application/json"}
        body = {"model": "omnirush", "messages": [{"role": "user", "content": "hi"}]}
        response = self.post(body, bad)
        self.assertEqual(response.status_code, 401)
        self.assertTrue(response.headers.get("X-Request-Id"))

    def test_plain_completion(self):
        body = {"model": "omnirush", "messages": [{"role": "user", "content": "Hello"}]}
        response = self.post(body)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["choices"][0]["message"]["content"], "hello")
        self.assertEqual(data["choices"][0]["finish_reason"], "stop")

    def test_tool_bridge_round_trip(self):
        tool = {"type": "function", "function": {"name": "lookup", "parameters": {
            "type": "object", "properties": {"query": {"type": "string"}}}}}
        envelope = '{"type":"tool_calls","tool_calls":[{"name":"lookup",'
        envelope += '"arguments":{"query":"x"}}]}'
        _, app = make_app(runner=ok_runner(envelope))
        client = TestClient(app, raise_server_exceptions=False)
        body = {"model": "omnirush", "messages": [{"role": "user", "content": "go"}],
                "tools": [tool], "tool_choice": "auto"}
        response = client.post("/v1/chat/completions", json=body, headers=self.auth)
        self.assertEqual(response.status_code, 200, response.text)
        message = response.json()["choices"][0]["message"]
        self.assertEqual(response.json()["choices"][0]["finish_reason"], "tool_calls")
        self.assertTrue(message["tool_calls"][0]["id"].startswith("call_"))
        args = json.loads(message["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(args, {"query": "x"})

    def test_streaming_plain_text(self):
        body = {"model": "omnirush", "messages": [{"role": "user", "content": "hi"}]}
        response = self.post({**body, "stream": True})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertTrue(response.text.startswith("data: "))
        self.assertIn("data: [DONE]", response.text)
        self.assertIn("hello", response.text)

    def test_unknown_model_and_bad_request(self):
        bad_model = {"model": "nope", "messages": [{"role": "user", "content": "hi"}]}
        self.assertEqual(self.post(bad_model).status_code, 404)
        empty = {"model": "omnirush", "messages": []}
        self.assertEqual(self.post(empty).status_code, 400)

    def test_cli_errors_are_sanitized(self):
        async def failing(provider, model, prompt, **kwargs):
            raise RuntimeError("boom SECRET-XYZ")
        _, app = make_app(runner=failing)
        client = TestClient(app, raise_server_exceptions=False)
        body = {"model": "omnirush", "messages": [{"role": "user", "content": "hi"}]}
        response = client.post("/v1/chat/completions", json=body, headers=self.auth)
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("SECRET-XYZ", response.text)
        self.assertEqual(response.json()["error"]["code"], "internal_error")

    def test_oversized_body_rejected(self):
        big = "x" * (self.config.server.max_body_bytes + 10)
        body = {"model": "omnirush", "messages": [{"role": "user", "content": big}]}
        response = self.client.post("/v1/chat/completions", content=json.dumps(body),
                                    headers=self.auth)
        self.assertEqual(response.status_code, 413)


if __name__ == "__main__":
    unittest.main()

