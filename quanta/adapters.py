from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .config import ModelAlias, Provider
from .errors import ApiError


@dataclass
class Invocation:
    args: list[str]
    stdin: str
    env: dict[str, str] = field(default_factory=dict)


OPENCODE_DEFAULT_MODEL = "opencode/big-pickle"


def effective_model(model: ModelAlias) -> str | None:
    """Upstream model id, or None to use the CLI's own configured default."""
    upstream = (model.upstream_model or "").strip()
    return None if upstream.lower() in ("", "default") else upstream


def build_invocation(provider: Provider, model: ModelAlias, prompt: str, state_dir: str) -> Invocation:
    upstream = effective_model(model)
    if provider.adapter == "opencode" and not upstream:
        # OpenCode's own default is whatever the machine last used; on a clean install it picks an image
        # model (google/gemini-3-pro-image-preview) and fails "User not found". Use its free tier instead.
        upstream = OPENCODE_DEFAULT_MODEL
    model_args = ["--model", upstream] if upstream else []
    provider_args = ["--provider", model.upstream_provider] if model.upstream_provider else []
    extra = provider.args
    if provider.adapter == "omnirush":
        return Invocation([*extra, "--print", "--mode", "json", "--no-session", "--no-tools", "--no-extensions",
                           "--no-skills", "--no-prompt-templates", "--no-themes", "--no-context-files", "--no-approve", "--no-yolo",
                           "--system-prompt", "You are a helpful text-only assistant.", "--append-system-prompt", "",
                           *provider_args, *model_args], prompt, {"OMNIRUSH_OFFLINE": "1"})
    if provider.adapter == "opencode":
        # opencode v2 `run` has no --pure/--dir flags: `opencode run --model
        # <id>` reads the prompt from stdin and uses cwd as the project
        # directory (the runner isolates cwd per request; see runner.py).
        return Invocation(["run", *extra, *model_args], prompt)
    if provider.adapter == "cline":
        # Cline CLI rejects stdin prompts in JSON mode ("JSON output mode
        # requires a prompt argument or piped stdin"): the prompt must be a
        # positional argument. The prompt is passed as a single argv item, so
        # shell metacharacters are data, not interpreted. Cline's own data dir
        # is used (an isolated --data-dir has no credentials and every run
        # fails "Unauthorized"); per-provider concurrency is 1 by default.
        return Invocation([*extra, "--json", "--plan", "--auto-approve", "false",
                           "--timeout", str(max(1, (provider.timeout_ms + 999) // 1000)),
                           *provider_args, *model_args, prompt], "",
                          {"CLINE_SESSION_BACKEND_MODE": "local", "CLINE_TOOL_APPROVAL_MODE": "terminal",
                           "CLINE_COMMAND_PERMISSIONS": '{"deny":["*"]}'})
    if provider.adapter == "antigravity":
        # --mode plan keeps the agent read-only; tool permission requests cannot
        # be approved in print mode, so they are not auto-granted.
        return Invocation([*extra, "--mode", "plan", "--input-format", "stream-json", "--output-format", "stream-json", "--print-timeout",
                           f"{max(1, (provider.timeout_ms + 999) // 1000)}s", *model_args],
                          json.dumps({"event": "user", "message": {"content": prompt}}, ensure_ascii=False) + "\n")
    if provider.adapter == "custom":
        args = [prompt if arg == "{prompt}" else arg.replace("{model}", upstream or model.id) for arg in extra]
        return Invocation(args, prompt if provider.prompt_mode == "stdin" else "")
    raise ApiError(503, "unsupported_adapter", "This CLI requires a non-interactive bridge.")


def _protocol_error():
    return ApiError(502, "cli_protocol_error", "CLI output does not match its expected protocol. Check CLI version and configuration locally.")


def _upstream_error(detail: str | None = None):
    message = "The CLI reported an unsuccessful model run. Check login and model configuration locally."
    status = 502
    if isinstance(detail, str) and detail.strip():
        # API error text from the upstream provider (never stderr/prompt text).
        clean = " ".join(detail.split())[:240]
        message += f" Upstream: {clean}"
        if re.search(r"429|grant exhausted|rate.?limit", clean, re.I):
            status = 429
    return ApiError(status, "cli_reported_error", message)


class Decoder:
    """Decode only assistant text; never expose reasoning, tool results or stderr."""

    def __init__(self, mode: str):
        self.mode = mode
        self.buffer = ""
        self.text = ""
        self.message_text = ""
        self.completed = False
        self.usage: dict | None = None
        self.cline_result = None
        self.legacy: dict = {}
        self.pending: list[str] = []

    def _add(self, text):
        if not isinstance(text, str):
            raise _protocol_error()
        self.text += text
        self.message_text += text
        if text:
            self.pending.append(text)

    def _authoritative(self, text, previous=None):
        previous = self.text if previous is None else previous
        if not isinstance(text, str) or not text.startswith(previous):
            raise _protocol_error()
        self._add(text[len(previous):])

    def _usage(self, input_tokens, output_tokens):
        if all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (input_tokens, output_tokens)):
            self.usage = {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}

    def _line(self, raw):
        if not raw.strip():
            return
        try:
            event = json.loads(raw)
            if not isinstance(event, dict):
                raise _protocol_error()
            self._event(event)
        except (ValueError, KeyError, AttributeError, TypeError):
            raise _protocol_error() from None

    def _event(self, event):
        kind = event.get("type")
        if self.mode == "omnirush":
            message = event.get("message", {})
            if kind == "message_start" and message.get("role") == "assistant":
                self.message_text = ""
            if kind == "message_update" and event.get("assistantMessageEvent", {}).get("type") == "text_delta":
                self._add(event["assistantMessageEvent"]["delta"])
            if kind == "message_end" and message.get("role") == "assistant":
                if message.get("stopReason") in ("error", "aborted") or message.get("errorMessage"):
                    raise _upstream_error(message.get("errorMessage"))
                text = "".join(p.get("text", "") for p in message.get("content", []) if p.get("type") == "text")
                self._authoritative(text, self.message_text)
                u = message.get("usage", {})
                self._usage(u.get("input", 0) + u.get("cacheRead", 0) + u.get("cacheWrite", 0), u.get("output"))
            if kind == "agent_end":
                self.completed = True
            if kind == "error":
                raise _upstream_error(event.get("message") if isinstance(event.get("message"), str) else None)
        elif self.mode == "cline":
            # Current Cline's nested events mix narration and tool activity.
            # Use the authoritative run_result rather than guessing deltas.
            if kind == "run_result":
                if event.get("finishReason") != "completed":
                    raise _upstream_error()
                self.cline_result = event["text"]
                self.completed = True
                u = event.get("aggregateUsage", event.get("usage", {}))
                self._usage(u.get("inputTokens", 0) + u.get("cacheReadTokens", 0) + u.get("cacheWriteTokens", 0), u.get("outputTokens"))
            if kind == "say" and event.get("say") in ("text", "completion_result"):
                self.legacy[event.get("ts", "last")] = event["text"]
            if kind in ("run_aborted", "error") or (kind == "say" and event.get("say") == "error"):
                raise _upstream_error(event.get("message") if isinstance(event.get("message"), str) else None)
        elif self.mode == "antigravity":
            update = event.get("step_update", {})
            if event.get("event") == "step_update" and update.get("step_type") == "agent_response" and "text_delta" in update:
                self._add(update["text_delta"])
            if event.get("event") == "result":
                result = event["result"]
                if result.get("status") != "SUCCESS":
                    raise _upstream_error()
                self._authoritative(result["response"])
                self.completed = True
                u = result.get("usage", {})
                self._usage(u.get("input_tokens"), u.get("output_tokens"))
        elif self.mode == "jsonl":
            if kind == "text":
                self._add(event["text"])
            elif kind == "done":
                self.completed = True
            elif kind == "error":
                raise _upstream_error()
            else:
                raise _protocol_error()
        else:
            raise _protocol_error()

    def feed(self, data: str) -> list[str]:
        self.pending = []
        if self.mode == "text":
            self._add(data)
        else:
            self.buffer += data
            while "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                self._line(line)
        return self.pending

    def finish(self) -> list[str]:
        self.pending = []
        if self.mode != "text":
            if self.buffer.strip():
                self._line(self.buffer)
            self.buffer = ""
            if self.mode == "cline":
                if self.completed:
                    self._add(self.cline_result)
                elif self.legacy:
                    self._add("\n".join(self.legacy.values()))
                    self.completed = True
            if not self.completed:
                raise _protocol_error()
        if not self.text.strip():
            raise ApiError(502, "empty_response", "The CLI returned no assistant text.")
        return self.pending
