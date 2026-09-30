"""Anthropic Messages API <-> the OpenAI-shaped core (for Claude Code and other Anthropic clients).

Requests are translated to the OpenAI chat format the rest of Quanta validates, and
results are translated back, so both APIs share one implementation of routing, limits,
cancellation and the client-side tool bridge. Tools are still proposed by the model
through that bridge and executed by the client (Claude Code), never by Quanta.
"""

from __future__ import annotations

import json
import uuid
from typing import AsyncIterator

ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
               404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error"}
STOP_REASONS = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}


class AnthropicRequestError(ValueError):
    """A request the bridge cannot represent (reported as invalid_request_error)."""


def error_body(status: int, message: str) -> dict:
    return {"type": "error", "error": {"type": ERROR_TYPES.get(status, "api_error"), "message": message}}


def estimate_tokens(value) -> int:
    """Rough token estimate (~4 chars/token); used only when a CLI reports no usage."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return max(1, len(text) // 4)


def _blocks(content, where: str) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list) and all(isinstance(b, dict) for b in content):
        return content
    raise AnthropicRequestError(f"{where} must be a string or an array of content blocks.")


def _text_of(content, where: str) -> str:
    """Flatten tool_result/system content to text; non-text media is replaced by a marker."""
    parts = []
    for block in _blocks(content, where):
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text", "")))
        elif kind in ("image", "document"):
            parts.append(f"[{kind} omitted: not supported by this gateway]")
        else:
            parts.append(json.dumps(block, ensure_ascii=False))
    return "\n".join(parts)


def to_openai(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise AnthropicRequestError("Request body must be a JSON object.")
    model = raw.get("model")
    messages = raw.get("messages")
    if not isinstance(model, str) or not model.strip():
        raise AnthropicRequestError("model is required.")
    if not isinstance(messages, list) or not messages:
        raise AnthropicRequestError("messages must be a non-empty array.")

    out: list[dict] = []
    system_parts = [_text_of(raw["system"], "system")] if raw.get("system") else []

    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant", "system", "developer"):
            raise AnthropicRequestError(f"messages[{index}].role must be 'user' or 'assistant'.")
        blocks = _blocks(message.get("content", ""), f"messages[{index}].content")
        if message["role"] in ("system", "developer"):
            # Newer Claude Code puts system reminders inside messages; fold them into the system prompt.
            system_parts.append(_text_of(blocks, f"messages[{index}].content"))
            continue
        following = next((m for m in messages[index + 1:] if isinstance(m, dict) and m.get("role") not in ("system", "developer")), None)
        answered = set()
        if isinstance(following, dict) and isinstance(following.get("content"), list):
            answered = {b.get("tool_use_id") for b in following["content"] if isinstance(b, dict) and b.get("type") == "tool_result"}

        if message["role"] == "assistant":
            texts, calls = [], []
            for block in blocks:
                kind = block.get("type")
                if kind == "text":
                    texts.append(str(block.get("text", "")))
                elif kind == "tool_use":
                    call_id, name = block.get("id"), block.get("name")
                    arguments = json.dumps(block.get("input") or {}, ensure_ascii=False, separators=(",", ":"))
                    if call_id in answered:
                        calls.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}})
                    else:  # interrupted call with no result: keep the fact, not a dangling tool call
                        texts.append(f"[called {name}({arguments}); no result was returned]")
                # thinking / redacted_thinking blocks are not replayed
            entry = {"role": "assistant", "content": "\n".join(t for t in texts if t) if (texts or not calls) else None}
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
            continue

        known = {c["id"] for prior in out if prior.get("role") == "assistant" for c in prior.get("tool_calls", [])}
        results, texts = [], []
        for block in blocks:
            kind = block.get("type")
            if kind == "tool_result":
                body = _text_of(block.get("content", ""), "tool_result.content")
                if block.get("is_error"):
                    body = "Error: " + body
                call_id = block.get("tool_use_id")
                if call_id in known:
                    results.append({"role": "tool", "tool_call_id": call_id, "content": body})
                else:  # orphan result (e.g. history was compacted): keep it as plain text
                    texts.append(f"[tool result]\n{body}")
            elif kind == "text":
                texts.append(str(block.get("text", "")))
            elif kind in ("image", "document"):
                texts.append(f"[{kind} omitted: not supported by this gateway]")
        out.extend(results)  # tool results must directly follow the assistant tool calls
        if texts or not results:
            out.append({"role": "user", "content": "\n".join(texts)})

    system_text = "\n\n".join(part for part in system_parts if part.strip())
    if system_text:
        out.insert(0, {"role": "system", "content": system_text})
    body: dict = {"model": model, "messages": out, "stream": bool(raw.get("stream"))}
    tools = raw.get("tools")
    if tools:
        functions = []
        for tool in tools:
            if not isinstance(tool, dict) or not tool.get("name"):
                continue
            if tool.get("type") not in (None, "custom"):  # server tools (web_search etc.) cannot run client-side
                continue
            function = {"name": tool["name"], "parameters": tool.get("input_schema") or {"type": "object", "properties": {}}}
            if tool.get("description"):
                function["description"] = str(tool["description"])
            functions.append({"type": "function", "function": function})
        if functions:
            body["tools"] = functions
            choice = raw.get("tool_choice") or {"type": "auto"}
            kind = choice.get("type") if isinstance(choice, dict) else choice
            if kind == "any":
                body["tool_choice"] = "required"
            elif kind == "none":
                body["tool_choice"] = "none"
            elif kind == "tool" and isinstance(choice, dict) and choice.get("name"):
                body["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
            else:
                body["tool_choice"] = "auto"
            if isinstance(choice, dict) and choice.get("disable_parallel_tool_use"):
                body["parallel_tool_calls"] = False
    for source, target in (("max_tokens", "max_tokens"), ("temperature", "temperature"), ("top_p", "top_p")):
        if isinstance(raw.get(source), (int, float)) and not isinstance(raw.get(source), bool):
            body[target] = raw[source]
    stops = raw.get("stop_sequences")
    if isinstance(stops, list) and stops and all(isinstance(s, str) and s for s in stops):
        body["stop"] = stops[:4]
    return body


def new_message_id() -> str:
    return "msg_" + uuid.uuid4().hex[:24]


def from_openai(response: dict, model: str, input_estimate: int) -> dict:
    choice = response["choices"][0]
    message = choice["message"]
    content: list[dict] = []
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for call in message.get("tool_calls") or []:
        try:
            arguments = json.loads(call["function"]["arguments"] or "{}")
        except ValueError:
            arguments = {}
        content.append({"type": "tool_use", "id": call["id"], "name": call["function"]["name"],
                        "input": arguments if isinstance(arguments, dict) else {}})
    if not content:
        content.append({"type": "text", "text": ""})
    usage = response.get("usage") or {}
    return {"id": new_message_id(), "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": STOP_REASONS.get(choice.get("finish_reason"), "end_turn"), "stop_sequence": None,
            "usage": {"input_tokens": usage.get("prompt_tokens", input_estimate),
                      "output_tokens": usage.get("completion_tokens", estimate_tokens(content))}}


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


async def stream_from_openai(frames: AsyncIterator[str], model: str, input_estimate: int) -> AsyncIterator[str]:
    """Convert OpenAI chat.completion.chunk SSE frames into Anthropic streaming events."""
    yield sse("message_start", {"type": "message_start", "message": {
        "id": new_message_id(), "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": input_estimate, "output_tokens": 1}}})
    index = -1
    open_kind = None            # "text" | "tool" | None
    tool_blocks: dict[int, int] = {}
    tool_started_args: set[int] = set()
    stop_reason, produced, usage = "end_turn", 0, None

    def close_block():
        nonlocal open_kind
        events = []
        if open_kind == "tool" and index not in tool_started_args:
            events.append(sse("content_block_delta", {"type": "content_block_delta", "index": index,
                                                      "delta": {"type": "input_json_delta", "partial_json": "{}"}}))
        if open_kind is not None:
            events.append(sse("content_block_stop", {"type": "content_block_stop", "index": index}))
        open_kind = None
        return events

    async for frame in frames:
        if frame.startswith(":"):
            yield sse("ping", {"type": "ping"})
            continue
        data = "".join(line[5:].strip() for line in frame.splitlines() if line.startswith("data:"))
        if not data or data == "[DONE]":
            continue
        event = json.loads(data)
        if "error" in event and "choices" not in event:
            for chunk in close_block():
                yield chunk
            yield sse("error", error_body(502, event["error"].get("message", "Upstream error.")))
            return
        if not event.get("choices"):
            usage = event.get("usage") or usage
            continue
        choice = event["choices"][0]
        delta = choice.get("delta") or {}
        if delta.get("content"):
            if open_kind != "text":
                for chunk in close_block():
                    yield chunk
                index += 1
                open_kind = "text"
                yield sse("content_block_start", {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}})
            produced += len(delta["content"])
            yield sse("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": delta["content"]}})
        for call in delta.get("tool_calls") or []:
            slot = call.get("index", 0)
            if call.get("id"):
                for chunk in close_block():
                    yield chunk
                index += 1
                tool_blocks[slot] = index
                open_kind = "tool"
                yield sse("content_block_start", {"type": "content_block_start", "index": index, "content_block": {
                    "type": "tool_use", "id": call["id"], "name": (call.get("function") or {}).get("name", ""), "input": {}}})
            arguments = (call.get("function") or {}).get("arguments")
            if arguments and slot in tool_blocks:
                tool_started_args.add(tool_blocks[slot])
                produced += len(arguments)
                yield sse("content_block_delta", {"type": "content_block_delta", "index": tool_blocks[slot],
                                                  "delta": {"type": "input_json_delta", "partial_json": arguments}})
        if choice.get("finish_reason"):
            stop_reason = STOP_REASONS.get(choice["finish_reason"], "end_turn")
    if index < 0:  # nothing was produced: Anthropic clients expect at least one block
        index = 0
        open_kind = "text"
        yield sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
    for chunk in close_block():
        yield chunk
    output = (usage or {}).get("completion_tokens") or max(1, produced // 4)
    yield sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                                "usage": {"output_tokens": output}})
    yield sse("message_stop", {"type": "message_stop"})
