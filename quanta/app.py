from __future__ import annotations

import asyncio
import collections
import contextlib
import hashlib
import hmac
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Awaitable, Callable

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import Configuration
from .anthropic import AnthropicRequestError, error_body, estimate_tokens, from_openai, stream_from_openai, to_openai
from .errors import ApiError
from .protocol import ProtocolError, bridge_enabled, format_prompt, parse_response, validate_request
from .runner import RunResult, run_cli

log = logging.getLogger("quanta")
auth_scheme = HTTPBearer(auto_error=False)


def _sse(data):
    return "data: " + (data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, separators=(",", ":"))) + "\n\n"


async def _body(request: Request, limit: int) -> dict:
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise ApiError(415, "unsupported_media_type", "Use Content-Type: application/json.")
    if request.headers.get("content-encoding", "identity") != "identity":
        raise ApiError(415, "unsupported_encoding", "Compressed request bodies are not supported.")
    size = request.headers.get("content-length")
    if size:
        try:
            if int(size) > limit:
                raise ApiError(413, "body_too_large", "Request body exceeds the configured limit.")
        except ValueError:
            raise ApiError(400, "invalid_body", "Invalid Content-Length header.") from None
    cached = getattr(request.state, "cached_body", None)
    if cached is None:
        cached = await request.body()  # pragma: no cover - middleware caches first
    if len(cached) > limit:
        raise ApiError(413, "body_too_large", "Request body exceeds the configured limit.")
    data = bytes(cached)
    try:
        def invalid_constant(value):
            raise ValueError(value)
        return json.loads(data.decode("utf-8"), parse_constant=invalid_constant)
    except (ValueError, UnicodeError):
        raise ApiError(400, "invalid_json", "Request body is not valid JSON.") from None


async def _cancel(task):
    if task is not None:
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


async def _disconnected(request):
    while not await request.is_disconnected():
        await asyncio.sleep(0.2)


CAPABILITIES = {
    "streaming": True,
    # Tools are declared by the client and executed by the client; the CLI only
    # proposes calls through a validated structured-output envelope.
    "tool_calling": "client-side prompt bridge",
    "subagents": False,
    "session_persistence": False,  # stateless: the client resends the conversation each request
    "model_switching": True,
    "provider_switching": True,
}


def create_app(config: Configuration, runner: Callable[..., Awaitable[RunResult]] = run_cli,
               base_config: Configuration | None = None) -> FastAPI:
    """``base_config`` is the full provider set used for per-request ``provider:model``
    routing; it defaults to ``config`` (which may be a single-provider dynamic copy)."""
    base = base_config or config
    running: set[asyncio.Task] = set()
    recent: collections.deque = collections.deque(maxlen=100)
    active: dict[str, int] = {}
    total = 0
    uploads = 0
    stopping = False
    key_hash = hashlib.sha256(config.server.api_key.encode()).digest()
    models = {m.id: m for m in config.exposed_models()}
    state_lock = asyncio.Lock()
    app_state_label = getattr(config, "dynamic_label", None) or "config file"

    @asynccontextmanager
    async def lifespan(app):
        nonlocal stopping
        yield
        stopping = True
        await asyncio.gather(*(_cancel(task) for task in list(running)))

    app = FastAPI(title="Quanta", version="0.3.0", lifespan=lifespan,
                  description="OpenAI Chat Completions over local AI CLIs. Tool calls are validated structured-output requests executed by the client (e.g. Hermes), never by the wrapper. This is a prompt bridge, not native provider function calling.")
    app.state.config = config
    app.state.dynamic_label = app_state_label

    def apply_dynamic_model(validated, alias_id: str) -> None:
        """Swap the served config in place (mid-session menu / admin API).

        In-flight requests keep their already-resolved provider/model; only
        new requests use the new map. Capacity counters are reset so a stuck
        busy count from the old provider cannot wedge the new one.
        """
        nonlocal total
        models.clear()
        models.update({m.id: m for m in validated.exposed_models()})
        app.state.config = validated
        app.state.dynamic_label = alias_id
        total = 0
        active.clear()

    app.state.apply_dynamic_model = apply_dynamic_model

    @app.exception_handler(ApiError)
    async def api_error(request, error):
        headers = {"WWW-Authenticate": "Bearer"} if error.status == 401 else {"Retry-After": "2"} if error.status == 429 else {}
        return JSONResponse(error.body(), status_code=error.status, headers=headers)

    @app.middleware("http")
    async def boundaries(request, call_next):
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        origin = request.headers.get("origin")
        # Consume the body once here. Starlette's BaseHTTPMiddleware wraps
        # receive(); if the endpoint later streams request.stream() and the
        # client already sent everything, downstream would otherwise wait for
        # a disconnect message that never arrives and hang. Caching the body
        # lets _body() validate without re-reading the consumed stream.
        body_bytes = await request.body()
        if origin and origin not in config.server.cors_origins:
            response = JSONResponse(ApiError(403, "origin_not_allowed", "Browser origin is not allowed; configure server.corsOrigins.").body(), status_code=403)
        elif request.method == "OPTIONS":
            response = Response(status_code=204, headers={"Access-Control-Allow-Methods": "GET, POST, OPTIONS", "Access-Control-Allow-Headers": "Authorization, Content-Type", "Access-Control-Max-Age": "600"})
        else:
            try:
                request.state.cached_body = bytes(body_bytes)
                response = await call_next(request)
            except Exception:
                # Do not log tracebacks or exception text from upstream processes.
                log.error("request=%s code=internal_error", request_id)
                response = JSONResponse(ApiError(500, "internal_error", "An internal server error occurred.").body(), status_code=500)
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        if origin in config.server.cors_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
        return response

    def check_key(token: str):
        supplied = hashlib.sha256(token.encode()).digest()
        if not token or not hmac.compare_digest(key_hash, supplied):
            raise ApiError(401, "invalid_api_key", "A valid Bearer API key is required.")
        if stopping:
            raise ApiError(503, "server_stopping", "Server is shutting down.")

    async def authorize(request: Request, credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(auth_scheme)]):
        # Bearer for OpenAI clients; x-api-key for Anthropic clients (Claude Code).
        check_key(credentials.credentials if credentials else request.headers.get("x-api-key", ""))

    @app.get("/health", tags=["Service"])
    async def health():
        """Liveness only; does not verify upstream login or model availability."""
        return {"status": "ok"}

    def resolve_model(name):
        """Return (alias, provider) for a configured alias or an ad-hoc ``provider:model``.

        The ad-hoc form lets a client change CLI and model between turns of the
        same conversation with no server-side switch: the conversation lives in
        the request, so history and system prompts carry over unchanged.
        """
        from .config import BRIDGE_NOTES, ModelAlias

        alias = models.get(name)
        if alias:
            return alias, app.state.config.providers[alias.provider]
        provider_name, sep, upstream = (name or "").partition(":")
        provider = base.providers.get(provider_name)
        if not sep or not provider or not provider.enabled or provider.adapter in BRIDGE_NOTES:
            return None, None
        upstream = upstream.strip()
        if not upstream or "\0" in upstream or len(upstream) > 200:
            return None, None
        # Validated above; model_construct skips the alias-id pattern, which would
        # reject legitimate upstream ids (e.g. "~vendor/model" or "id:tag").
        return ModelAlias.model_construct(id=name[:200], provider=provider_name, upstream_model=upstream, upstream_provider=None), provider

    @app.get("/v1/providers", dependencies=[Depends(authorize)], tags=["Service"])
    async def list_providers():
        """Configured CLI providers, whether they are enabled, and what the gateway supports."""
        from .config import BRIDGE_NOTES

        data = []
        for name, item in base.providers.items():
            blocked = item.adapter in BRIDGE_NOTES
            data.append({"id": name, "adapter": item.adapter, "enabled": item.enabled and not blocked,
                         "available": not blocked, "note": BRIDGE_NOTES.get(item.adapter),
                         "capabilities": CAPABILITIES if not blocked else {k: False for k in CAPABILITIES},
                         "model_syntax": f"{name}:<upstream-model-id> (or 'default' for the CLI's own default)"})
        return {"object": "list", "data": data}

    model_cache: dict[str, tuple[float, list[str]]] = {}

    @app.get("/v1/providers/{name}/models", dependencies=[Depends(authorize)], tags=["Service"])
    async def provider_models(name: str, q: str = "", limit: int = 0):
        """Upstream model ids the provider's CLI reports (cached 60s; empty when the CLI cannot list them)."""
        from .config import BRIDGE_NOTES
        from .picker import _provider_models

        provider = base.providers.get(name)
        if not provider or provider.adapter in BRIDGE_NOTES:
            raise ApiError(404, "provider_not_found", "Unknown or unavailable provider; use GET /v1/providers.", "name")
        now = time.monotonic()
        cached = model_cache.get(name)
        if not cached or now - cached[0] > 60:
            listed = await asyncio.to_thread(_provider_models, provider.adapter)
            model_cache[name] = cached = (now, listed)
        from .picker import filter_models

        matches = filter_models(cached[1], q) if q.strip() else cached[1]
        page = matches[:limit] if limit > 0 else matches
        return {"object": "list", "provider": name, "listable": provider.adapter in ("omnirush", "opencode", "antigravity"),
                "total": len(cached[1]), "matched": len(matches),
                "data": [{"id": m, "object": "model", "owned_by": name} for m in page]}

    @app.get("/v1/requests", dependencies=[Depends(authorize)], tags=["Service"])
    async def recent_requests():
        """Most recent chat completions (metadata only: never prompts, outputs or keys)."""
        return {"object": "list", "data": list(recent)}

    @app.get("/v1/config", dependencies=[Depends(authorize)], tags=["Service"])
    async def public_config():
        """Read-only, secret-free view of the running configuration."""
        current = app.state.config
        return {"object": "config", "serving": app.state.dynamic_label,
                "server": {"host": base.server.host, "port": base.server.port, "max_concurrent": base.server.max_concurrent,
                           "max_body_bytes": base.server.max_body_bytes, "cors_origins": base.server.cors_origins},
                "defaults": {"timeout_ms": base.defaults.timeout_ms, "max_output_bytes": base.defaults.max_output_bytes},
                "providers": {n: {"adapter": p.adapter, "enabled": p.enabled, "timeout_ms": p.timeout_ms,
                                  "max_concurrent": p.max_concurrent} for n, p in base.providers.items()},
                "aliases": [{"id": m.id, "provider": m.provider, "upstream_model": m.upstream_model}
                            for m in current.exposed_models()]}

    @app.get("/v1/models", dependencies=[Depends(authorize)], tags=["OpenAI"])
    async def list_models():
        data = [{"id": m.id, "object": "model", "type": "model", "display_name": m.id, "created": 0,
                 "created_at": "1970-01-01T00:00:00Z", "owned_by": m.provider,
                 **({"upstream_model": m.upstream_model} if m.upstream_model else {})}
                for m in models.values()]
        return {"object": "list", "data": data, "has_more": False,
                "first_id": data[0]["id"] if data else None, "last_id": data[-1]["id"] if data else None}

    @app.get("/v1/session", dependencies=[Depends(authorize)], tags=["Service"])
    async def session_info():
        """Current in-memory selection: alias, provider, upstream model id."""
        current = [{"id": m.id, "provider": m.provider, "upstream_model": m.upstream_model}
                   for m in models.values()]
        return {"object": "session", "serving": app.state.dynamic_label, "models": current}

    @app.post("/v1/session/switch", dependencies=[Depends(authorize)], tags=["Service"])
    async def session_switch(request: Request):
        """Switch provider/model mid-session without touching config files.

        Body: {"provider": "<name>", "model": "<upstream-id>", "alias": "<optional>"}.
        In-flight requests finish on the old mapping; new requests use the new
        one. Same Bearer key required.
        """
        from .picker import compose_dynamic_model

        try:
            body = await _body(request, 65536)
        except ApiError as error:
            raise error
        if not isinstance(body, dict):
            raise ApiError(400, "invalid_body", "Body must be a JSON object.")
        provider_name = body.get("provider")
        upstream = body.get("model")
        alias = body.get("alias")
        if not isinstance(provider_name, str) or not isinstance(upstream, str):
            raise ApiError(400, "invalid_request", "Fields 'provider' (string) and 'model' (string) are required.")
        if alias is not None and not isinstance(alias, str):
            raise ApiError(400, "invalid_request", "Field 'alias' must be a string.")
        base_config = app.state.config
        try:
            async with state_lock:
                validated, alias_id = compose_dynamic_model(base_config, provider_name=provider_name,
                                                            upstream=upstream, alias=alias)
                apply_dynamic_model(validated, alias_id)
        except ValueError as error:
            raise ApiError(400, "invalid_request", str(error)) from None
        log.info("session switch: provider=%s alias=%s upstream=%s", provider_name, alias_id, upstream)
        return {"object": "session", "serving": alias_id, "provider": provider_name,
                "upstream_model": upstream.strip(), "models": [{"id": alias_id}]}

    request_schema = {
        "requestBody": {"required": True, "content": {"application/json": {"schema": {
            "type": "object", "required": ["model", "messages"], "properties": {
                "model": {"type": "string"}, "messages": {"type": "array", "items": {"type": "object"}},
                "tools": {"type": "array", "items": {"type": "object"}}, "tool_choice": {"oneOf": [{"type": "string"}, {"type": "object"}]},
                "parallel_tool_calls": {"type": "boolean"}, "stream": {"type": "boolean", "default": False},
                "temperature": {"type": "number", "description": "Advisory CLI prompt hint, not a native sampler control"},
                "max_tokens": {"type": "integer", "description": "Advisory output-budget hint; not an exact native token limit"}
            }}, "example": {"model": "omnirush", "messages": [{"role": "user", "content": "Hello"}]}}}}
    }

    async def chat(request: Request, prebuilt: dict | None = None):
        """Shared core for the OpenAI and Anthropic endpoints; ``prebuilt`` is an OpenAI-shaped body."""
        nonlocal total, uploads
        if uploads >= config.server.max_concurrent * 4:
            raise ApiError(429, "server_busy", "Too many pending requests. Retry later.")
        uploads += 1
        try:
            try:
                snapshot = app.state.config
                body = validate_request(prebuilt if prebuilt is not None else await _body(request, snapshot.server.max_body_bytes))
            except ProtocolError as error:
                raise ApiError(400, error.code, str(error), error.param) from None
            model, provider = resolve_model(body["model"])
            if not model:
                raise ApiError(404, "model_not_found", "Unknown or disabled model; use GET /v1/models or '<provider>:<model>' (GET /v1/providers).", "model")
            async with state_lock:
                if total >= snapshot.server.max_concurrent or active.get(model.provider, 0) >= provider.max_concurrent:
                    raise ApiError(429, "provider_busy", "CLI capacity is busy. Retry later.")
                total += 1
                active[model.provider] = active.get(model.provider, 0) + 1
        finally:
            uploads -= 1

        started = time.monotonic()
        released = False

        def release():
            nonlocal total, released
            if not released:
                released = True
                total -= 1
                active[model.provider] -= 1

        queue: asyncio.Queue = asyncio.Queue(maxsize=16)
        prompt_bridge = bridge_enabled(body)
        stream = body.get("stream", False)

        async def text_delta(text):
            # Never expose a raw tool envelope as assistant content. Buffer and
            # validate it first, then emit genuine OpenAI tool_calls deltas.
            if stream and not prompt_bridge:
                await queue.put(("text", text))

        async def generate():
            outcome = {"status": "cancelled", "code": "request_cancelled", "tool_calls": 0}
            try:
                result = await runner(provider, model, format_prompt(body), on_text=text_delta,
                                      secret_env=snapshot.server.api_key_env)
                try:
                    parsed = parse_response(result.text, body)
                except ProtocolError as error:
                    raise ApiError(502, "invalid_model_output", str(error)) from None
                outcome.update(status="ok", code=None, tool_calls=len(parsed.get("tool_calls") or []))
                await queue.put(("result", (parsed, result.usage)))
            except asyncio.CancelledError:
                raise
            except ApiError as error:
                outcome.update(status=error.status, code=error.code)
                log.warning("request=%s model=%s code=%s", request.state.request_id, model.id, error.code)
                await queue.put(("error", error))
            except Exception:
                outcome.update(status=500, code="internal_error")
                log.error("request=%s model=%s code=internal_error", request.state.request_id, model.id)
                await queue.put(("error", ApiError(500, "internal_error", "An internal server error occurred.")))
            finally:
                duration = int((time.monotonic() - started) * 1000)
                recent.appendleft({"id": request.state.request_id, "time": int(time.time()), "model": model.id,
                                   "provider": model.provider, "upstream_model": model.upstream_model,
                                   "stream": bool(stream), "tools": prompt_bridge, "duration_ms": duration, **outcome})
                log.info("request=%s model=%s duration_ms=%d", request.state.request_id, model.id, duration)

        job = asyncio.create_task(generate())
        running.add(job)
        job.add_done_callback(running.discard)
        watcher = asyncio.create_task(_disconnected(request))
        first = asyncio.create_task(queue.get())
        try:
            done, _ = await asyncio.wait((first, watcher), return_when=asyncio.FIRST_COMPLETED)
            if watcher in done:
                raise ApiError(499, "request_cancelled", "Request cancelled.")
            item = first.result()
            if item[0] == "error":
                raise item[1]
        except BaseException:
            await _cancel(first)
            await _cancel(job)
            release()
            raise
        finally:
            await _cancel(watcher)

        completion_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        base = {"id": completion_id, "created": created, "model": model.id}
        headers = {"X-Quanta-Tool-Mode": "prompt-bridge", "X-Quanta-Generation-Controls": "advisory"}

        if not stream:
            parsed, usage = item[1]
            message = {"role": "assistant", "content": parsed["content"]}
            if parsed.get("tool_calls"):
                message["tool_calls"] = parsed["tool_calls"]
            response = {**base, "object": "chat.completion", "choices": [{"index": 0, "message": message, "finish_reason": parsed["finish_reason"]}]}
            if usage is not None:
                response["usage"] = usage
            await job
            release()
            return JSONResponse(response, headers=headers)

        include_usage = (body.get("stream_options") or {}).get("include_usage", False)

        def chunk(delta, finish=None):
            return {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **({"usage": None} if include_usage else {})}

        async def events():
            emitted_text = False
            next_item = item
            try:
                yield _sse(chunk({"role": "assistant", "content": ""}))
                while True:
                    kind, value = next_item
                    if kind == "error":
                        yield _sse(value.body())
                        yield _sse("[DONE]")
                        return
                    if kind == "text":
                        emitted_text = True
                        yield _sse(chunk({"content": value}))
                    elif kind == "result":
                        parsed, usage = value
                        if parsed.get("tool_calls"):
                            for index, call in enumerate(parsed["tool_calls"]):
                                function = call["function"]
                                yield _sse(chunk({"tool_calls": [{"index": index, "id": call["id"], "type": "function", "function": {"name": function["name"], "arguments": ""}}]}))
                                arguments = function["arguments"]
                                for offset in range(0, len(arguments), 256):
                                    yield _sse(chunk({"tool_calls": [{"index": index, "function": {"arguments": arguments[offset:offset + 256]}}]}))
                        elif not emitted_text and parsed.get("content"):
                            yield _sse(chunk({"content": parsed["content"]}))
                        yield _sse(chunk({}, parsed["finish_reason"]))
                        if include_usage:
                            yield _sse({**base, "object": "chat.completion.chunk", "choices": [], "usage": usage})
                        yield _sse("[DONE]")
                        return
                    try:
                        next_item = await asyncio.wait_for(queue.get(), 15)
                    except TimeoutError:
                        yield ": keep-alive\n\n"
                        # Continue waiting; never process the previous text twice.
                        next_item = ("heartbeat", None)
            finally:
                await _cancel(job)
                release()

        headers.update({"X-Accel-Buffering": "no", "Cache-Control": "no-cache, no-transform"})
        return StreamingResponse(events(), media_type="text/event-stream", headers=headers)

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)], tags=["OpenAI"], openapi_extra=request_schema)
    async def completions(request: Request):
        return await chat(request)

    # ---- Anthropic Messages API (Claude Code) -------------------------------------------
    def anthropic_error(error: ApiError):
        headers = {"retry-after": "2"} if error.status == 429 else {}
        return JSONResponse(error_body(error.status, error.message), status_code=error.status, headers=headers)

    async def anthropic_guard(request: Request):
        token = request.headers.get("x-api-key", "")
        auth = request.headers.get("authorization", "")
        if not token and auth.lower().startswith("bearer "):
            token = auth[7:].strip()
        check_key(token)

    def input_estimate(raw: dict) -> int:
        return estimate_tokens([raw.get("system") or "", raw.get("messages") or [], raw.get("tools") or []])

    @app.post("/v1/messages", tags=["Anthropic"])
    async def anthropic_messages(request: Request):
        """Anthropic Messages API over the same CLI core. Tool use is proposed through the client-side bridge."""
        try:
            await anthropic_guard(request)
            raw = await _body(request, app.state.config.server.max_body_bytes)
            try:
                body = to_openai(raw)
            except AnthropicRequestError as error:
                raise ApiError(400, "invalid_request", str(error)) from None
            requested = body["model"]
            # Claude Code asks for claude-* model names; route those to the first served alias.
            if not resolve_model(requested)[0] and requested.lower().startswith("claude") and models:
                body["model"] = next(iter(models))
            estimate = input_estimate(raw)
            response = await chat(request, body)
        except ApiError as error:
            return anthropic_error(error)
        headers = {"X-Quanta-Tool-Mode": "prompt-bridge", "X-Quanta-Routed-Model": body["model"]}
        if isinstance(response, StreamingResponse):
            inner = response.body_iterator

            async def frames():
                try:
                    async for frame in stream_from_openai(inner, requested, estimate):
                        yield frame
                finally:
                    close = getattr(inner, "aclose", None)
                    if close is not None:
                        await close()

            headers.update({"X-Accel-Buffering": "no", "Cache-Control": "no-cache, no-transform"})
            return StreamingResponse(frames(), media_type="text/event-stream", headers=headers)
        return JSONResponse(from_openai(json.loads(response.body), requested, estimate), headers=headers)

    @app.post("/v1/messages/count_tokens", tags=["Anthropic"])
    async def anthropic_count_tokens(request: Request):
        """Estimate only (~4 chars/token): the CLIs do not expose a tokenizer."""
        try:
            await anthropic_guard(request)
            raw = await _body(request, app.state.config.server.max_body_bytes)
        except ApiError as error:
            return anthropic_error(error)
        if not isinstance(raw, dict):
            return anthropic_error(ApiError(400, "invalid_request", "Request body must be a JSON object."))
        return JSONResponse({"input_tokens": input_estimate(raw)}, headers={"X-Quanta-Token-Count": "estimate"})

    return app
