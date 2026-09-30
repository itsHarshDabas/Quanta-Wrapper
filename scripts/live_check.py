"""Live end-to-end checks against a running Quanta server using the official
OpenAI client (the same client shape Hermes uses).

    python scripts/live_check.py [--base http://127.0.0.1:8787/v1] [--models a,b,...]

The API key is read from $QUANTA_API_KEY or quanta.config.json; it is never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
from openai import APIStatusError, OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--base", default="http://127.0.0.1:8787/v1")
parser.add_argument("--models", default="opencode,antigravity,cline,omnirush")
args = parser.parse_args()

key = os.environ.get("QUANTA_API_KEY")
if not key:
    key = json.loads(Path(__file__).resolve().parents[1].joinpath("quanta.config.json").read_text(encoding="utf-8"))["server"]["apiKey"]
client = OpenAI(base_url=args.base, api_key=key, timeout=240, max_retries=0)
results: list[tuple[str, str, str]] = []


def record(name, ok, detail=""):
    results.append((name, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def attempt(name, fn):
    started = time.time()
    try:
        detail = fn()
        record(name, True, f"({time.time() - started:.1f}s) {detail or ''}")
    except Exception as error:  # noqa: BLE001 - report every failure mode
        message = getattr(error, "message", str(error))
        record(name, False, f"({time.time() - started:.1f}s) {type(error).__name__}: {message[:260]}")


root = args.base.rsplit("/v1", 1)[0]
attempt("health", lambda: str(httpx.get(root + "/health").json()))
attempt("models list", lambda: ",".join(m.id for m in client.models.list().data))
attempt("providers", lambda: ",".join(f"{p['id']}={'on' if p['enabled'] else 'off'}" for p in httpx.get(args.base + "/providers", headers={"Authorization": f"Bearer {key}"}).json()["data"]))


def bad_key():
    try:
        OpenAI(base_url=args.base, api_key="wrong-key-wrong-key-wrong-key", max_retries=0).models.list()
    except APIStatusError as error:
        assert error.status_code == 401, error.status_code
        return "401 as expected"
    raise AssertionError("bad key accepted")


attempt("auth rejects bad key", bad_key)


def bad_model():
    try:
        client.chat.completions.create(model="nope", messages=[{"role": "user", "content": "hi"}])
    except APIStatusError as error:
        assert error.status_code == 404, error.status_code
        return "404 as expected"
    raise AssertionError("unknown model accepted")


attempt("unknown model -> 404", bad_model)

for model in [m for m in args.models.split(",") if m]:
    def basic(model=model):
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Reply with exactly the single word: pong"}])
        text = r.choices[0].message.content
        assert "pong" in text.lower(), text
        return f"text={text.strip()[:40]!r} usage={r.usage.total_tokens if r.usage else None}"

    def stream(model=model):
        chunks = list(client.chat.completions.create(model=model, stream=True, stream_options={"include_usage": True},
                      messages=[{"role": "user", "content": "Count from 1 to 5 separated by spaces."}]))
        text = "".join((c.choices[0].delta.content or "") for c in chunks if c.choices)
        assert "5" in text, text
        return f"chunks={len(chunks)} text={text.strip()[:40]!r}"

    attempt(f"{model}: chat completion", basic)
    attempt(f"{model}: streaming", stream)



def _expect404(model):
    try:
        client.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}])
    except APIStatusError as error:
        assert error.status_code == 404, error.status_code
        return "404 as expected"
    raise AssertionError("accepted")


WEATHER = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def tool_roundtrip(model):
    messages = [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}]
    first = client.chat.completions.create(model=model, messages=messages, tools=WEATHER, tool_choice="auto")
    choice = first.choices[0]
    assert choice.finish_reason == "tool_calls" and choice.message.tool_calls, f"no tool call: {choice.message.content!r}"
    call = choice.message.tool_calls[0]
    assert call.function.name == "get_weather", call.function.name
    arguments = json.loads(call.function.arguments)
    assert "paris" in arguments.get("city", "").lower(), arguments
    messages += [choice.message.model_dump(exclude_none=True),
                 {"role": "tool", "tool_call_id": call.id, "content": json.dumps({"temp_c": 23, "sky": "sunny"})}]
    final = client.chat.completions.create(model=model, messages=messages, tools=WEATHER)
    text = final.choices[0].message.content or ""
    assert final.choices[0].finish_reason == "stop" and "23" in text, text
    return f"call={call.function.name}({arguments}) -> final={text.strip()[:60]!r}"


def tool_stream(model):
    chunks = list(client.chat.completions.create(model=model, stream=True, tools=WEATHER,
                  messages=[{"role": "user", "content": "What is the weather in Tokyo? Use the tool."}]))
    names = "".join((tc.function.name or "") for c in chunks if c.choices for tc in (c.choices[0].delta.tool_calls or []))
    arguments = "".join((tc.function.arguments or "") for c in chunks if c.choices for tc in (c.choices[0].delta.tool_calls or []))
    assert names == "get_weather" and "tokyo" in json.loads(arguments)["city"].lower(), (names, arguments)
    return f"streamed tool_call {names}({arguments})"


def tool_choice_none(model):
    r = client.chat.completions.create(model=model, tools=WEATHER, tool_choice="none",
                                       messages=[{"role": "user", "content": "Say hello in one word."}])
    assert not r.choices[0].message.tool_calls and r.choices[0].message.content
    return "no tool call when tool_choice=none"


for model in [m for m in args.models.split(",") if m][:2]:
    attempt(f"{model}: tool-call round trip", lambda model=model: tool_roundtrip(model))
    attempt(f"{model}: streamed tool call", lambda model=model: tool_stream(model))
    attempt(f"{model}: tool_choice=none", lambda model=model: tool_choice_none(model))


def switch_chain(chain, label):
    history = [{"role": "system", "content": "You are terse. Answer in as few words as possible."},
               {"role": "user", "content": "My secret codeword is PINEAPPLE-42. Acknowledge with OK."}]
    used = []
    for step, model in enumerate(chain):
        if step:
            history.append({"role": "user", "content": "What is my secret codeword? Answer with the codeword only."})
        r = client.chat.completions.create(model=model, messages=history)
        reply = r.choices[0].message.content or ""
        history.append({"role": "assistant", "content": reply})
        used.append(f"{model}->{reply.strip()[:24]!r}")
        if step:
            assert "pineapple-42" in reply.lower(), f"lost context at {model}: {reply!r}"
    return " | ".join(used)


attempt("provider switch A->B->A (same conversation)",
        lambda: switch_chain(["opencode", "antigravity:gemini-3.8-flash-low", "cline:default"], "provider"))
attempt("model switch within a provider (same conversation)",
        lambda: switch_chain(["antigravity:gemini-3.8-flash-low", "antigravity:gemini-3.7-flash-low"], "model"))
attempt("ad-hoc model on unknown provider -> 404", lambda: _expect404("freebuff:whatever"))

failed = [r for r in results if r[1] == "FAIL"]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
