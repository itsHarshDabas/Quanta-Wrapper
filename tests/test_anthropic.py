"""Anthropic Messages API (Claude Code) compatibility, using an offline fake runner."""

from __future__ import annotations

import json
import unittest

from fastapi.testclient import TestClient

from quanta.anthropic import AnthropicRequestError, to_openai
from quanta.app import create_app
from quanta.config import example_config, validate_config
from quanta.errors import ApiError
from quanta.runner import RunResult

KEY = "test-" + "k" * 24
BASH = {"name": "Bash", "description": "Run a command", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}


def make(reply="hello", usage=None, fail=None):
    raw = example_config()
    raw["server"]["apiKey"] = KEY
    config = validate_config(raw, env={})
    seen = []

    async def runner(provider, model, prompt, **kwargs):
        seen.append(prompt)
        if fail:
            raise fail
        text = reply(prompt) if callable(reply) else reply
        if kwargs.get("on_text"):
            await kwargs["on_text"](text)
        return RunResult(text, usage)

    return TestClient(create_app(config, runner=runner), raise_server_exceptions=False), seen


def post(client, body, headers=None, path="/v1/messages"):
    return client.post(path, json=body, headers=headers or {"x-api-key": KEY})


def sse_events(response):
    events = []
    for frame in response.text.split("\n\n"):
        if frame.strip():
            lines = dict(line.split(": ", 1) for line in frame.splitlines() if ": " in line)
            events.append((lines["event"], json.loads(lines["data"])))
    return events


class MessagesTests(unittest.TestCase):
    def test_text_reply_shape_and_routing(self):
        client, _ = make("hi there", usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10})
        r = post(client, {"model": "omnirush", "max_tokens": 64, "system": "Be brief.", "messages": [{"role": "user", "content": "Hello"}]})
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual((data["type"], data["role"], data["model"], data["stop_reason"]), ("message", "assistant", "omnirush", "end_turn"))
        self.assertEqual(data["content"], [{"type": "text", "text": "hi there"}])
        self.assertEqual(data["usage"], {"input_tokens": 7, "output_tokens": 3})

    def test_both_auth_styles_and_rejection_in_anthropic_format(self):
        client, _ = make()
        body = {"model": "omnirush", "messages": [{"role": "user", "content": "x"}]}
        self.assertEqual(post(client, body, {"Authorization": "Bearer " + KEY}).status_code, 200)
        bad = post(client, body, {"x-api-key": "nope"})
        self.assertEqual(bad.status_code, 401)
        self.assertEqual(bad.json(), {"type": "error", "error": {"type": "authentication_error", "message": "A valid Bearer API key is required."}})

    def test_claude_model_names_route_to_first_served_alias(self):
        client, seen = make()
        r = post(client, {"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["model"], "claude-sonnet-4-5")  # echoes what the client asked for
        self.assertEqual(r.headers["X-Quanta-Routed-Model"], "omnirush")
        unknown = post(client, {"model": "gpt-9", "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(unknown.status_code, 404)
        self.assertEqual(unknown.json()["error"]["type"], "not_found_error")

    def test_tool_use_round_trip(self):
        envelope = json.dumps({"type": "tool_calls", "tool_calls": [{"name": "Bash", "arguments": {"command": "echo hi"}}]})
        client, _ = make(envelope)
        r = post(client, {"model": "omnirush", "tools": [BASH], "messages": [{"role": "user", "content": "run echo"}]})
        data = r.json()
        self.assertEqual(data["stop_reason"], "tool_use")
        block = data["content"][-1]
        self.assertEqual((block["type"], block["name"], block["input"]), ("tool_use", "Bash", {"command": "echo hi"}))
        # Claude Code now answers with a tool_result; the model then gives final text.
        client2, seen = make(json.dumps({"type": "message", "content": "It printed hi."}))
        messages = [{"role": "user", "content": "run echo"},
                    {"role": "assistant", "content": [{"type": "text", "text": "Running."}, {"type": "tool_use", "id": block["id"], "name": "Bash", "input": {"command": "echo hi"}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": block["id"], "content": [{"type": "text", "text": "hi"}]}]}]
        final = post(client2, {"model": "omnirush", "tools": [BASH], "messages": messages}).json()
        self.assertEqual((final["stop_reason"], final["content"][0]["text"]), ("end_turn", "It printed hi."))
        self.assertIn(block["id"], seen[0])

    def test_streaming_text_events(self):
        client, _ = make("abc")
        events = sse_events(post(client, {"model": "omnirush", "stream": True, "messages": [{"role": "user", "content": "x"}]}))
        names = [e for e, _ in events]
        self.assertEqual(names[0], "message_start")
        self.assertEqual(names[-2:], ["message_delta", "message_stop"])
        self.assertIn("content_block_start", names)
        text = "".join(d["delta"]["text"] for e, d in events if e == "content_block_delta")
        self.assertEqual(text, "abc")
        self.assertEqual(events[-2][1]["delta"]["stop_reason"], "end_turn")

    def test_streaming_tool_use_events(self):
        envelope = json.dumps({"type": "tool_calls", "tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]})
        client, _ = make(envelope)
        events = sse_events(post(client, {"model": "omnirush", "stream": True, "tools": [BASH], "messages": [{"role": "user", "content": "x"}]}))
        start = next(d for e, d in events if e == "content_block_start")
        self.assertEqual((start["content_block"]["type"], start["content_block"]["name"]), ("tool_use", "Bash"))
        partial = "".join(d["delta"]["partial_json"] for e, d in events if e == "content_block_delta")
        self.assertEqual(json.loads(partial), {"command": "ls"})
        self.assertEqual(events[-2][1]["delta"]["stop_reason"], "tool_use")
        # the raw envelope is never leaked as text
        self.assertFalse(any(d["delta"].get("type") == "text_delta" for e, d in events if e == "content_block_delta"))

    def test_upstream_errors_use_anthropic_error_shape(self):
        client, _ = make(fail=ApiError(429, "cli_reported_error", "grant exhausted"))
        r = post(client, {"model": "omnirush", "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(r.status_code, 429)
        self.assertEqual(r.json(), {"type": "error", "error": {"type": "rate_limit_error", "message": "grant exhausted"}})
        self.assertEqual(r.headers["retry-after"], "2")

    def test_count_tokens_is_an_estimate(self):
        client, _ = make()
        r = post(client, {"model": "omnirush", "messages": [{"role": "user", "content": "a" * 400}]}, path="/v1/messages/count_tokens")
        self.assertEqual(r.status_code, 200)
        self.assertGreaterEqual(r.json()["input_tokens"], 100)
        self.assertEqual(r.headers["X-Quanta-Token-Count"], "estimate")

    def test_models_endpoint_accepts_x_api_key(self):
        client, _ = make()
        r = client.get("/v1/models", headers={"x-api-key": KEY})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["data"][0]["type"], "model")


class ConversionTests(unittest.TestCase):
    def test_orphan_tool_result_and_unanswered_tool_use_are_tolerated(self):
        body = to_openai({"model": "m", "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "gone", "content": "old"}, {"type": "text", "text": "continue"}]}]})
        roles = [m["role"] for m in body["messages"]]
        self.assertEqual(roles, ["user", "assistant", "user"])
        self.assertNotIn("tool_calls", body["messages"][1])
        self.assertIn("no result was returned", body["messages"][1]["content"])
        self.assertIn("old", body["messages"][2]["content"])

    def test_tool_results_precede_user_text_and_error_flag(self):
        body = to_openai({"model": "m", "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "Bash", "input": {"command": "x"}}]},
            {"role": "user", "content": [{"type": "text", "text": "and then?"}, {"type": "tool_result", "tool_use_id": "a", "is_error": True, "content": "boom"}]}]})
        self.assertEqual([m["role"] for m in body["messages"]], ["user", "assistant", "tool", "user"])
        self.assertEqual(body["messages"][2]["content"], "Error: boom")

    def test_tool_choice_and_server_tools(self):
        base = {"model": "m", "messages": [{"role": "user", "content": "x"}], "tools": [BASH, {"type": "web_search_20250305", "name": "web_search"}]}
        self.assertEqual(len(to_openai(base)["tools"]), 1)  # server tools are skipped
        self.assertEqual(to_openai({**base, "tool_choice": {"type": "any"}})["tool_choice"], "required")
        forced = to_openai({**base, "tool_choice": {"type": "tool", "name": "Bash", "disable_parallel_tool_use": True}})
        self.assertEqual(forced["tool_choice"], {"type": "function", "function": {"name": "Bash"}})
        self.assertFalse(forced["parallel_tool_calls"])

    def test_system_role_inside_messages_is_folded_into_system_prompt(self):
        body = to_openai({"model": "m", "system": [{"type": "text", "text": "Top."}], "messages": [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": [{"type": "text", "text": "Reminder."}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "Bash", "input": {}}]},
            {"role": "system", "content": "Another."},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "ok"}]}]})
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user", "assistant", "tool"])
        self.assertIn("Top.", body["messages"][0]["content"])
        self.assertIn("Reminder.", body["messages"][0]["content"])
        self.assertIn("tool_calls", body["messages"][2])  # the answer is still found past the system entry

    def test_rejects_bad_shapes(self):
        for bad in ({}, {"model": "m"}, {"model": "m", "messages": [{"role": "tool", "content": "x"}]}):
            with self.assertRaises(AnthropicRequestError):
                to_openai(bad)


if __name__ == "__main__":
    unittest.main()
