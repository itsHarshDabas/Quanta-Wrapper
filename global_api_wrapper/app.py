from __future__ import annotations

import asyncio
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
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import Configuration
from .errors import ApiError
from .protocol import ProtocolError, bridge_enabled, format_prompt, parse_response, validate_request
from .runner import RunResult, run_cli

log = logging.getLogger("global_api_wrapper")
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


def create_app(config: Configuration, runner: Callable[..., Awaitable[RunResult]] = run_cli) -> FastAPI:
    running: set[asyncio.Task] = set()
    active: dict[str, int] = {}
    total = 0
    uploads = 0
    stopping = False
    key_hash = hashlib.sha256(config.server.api_key.encode()).digest()
    models = {m.id: m for m in config.exposed_models()}

    @asynccontextmanager
    async def lifespan(app):
        nonlocal stopping
        yield
        stopping = True
        await asyncio.gather(*(_cancel(task) for task in list(running)))

    app = FastAPI(title="Global API Wrapper", version="0.2.0", lifespan=lifespan,
                  description="OpenAI Chat Completions over local AI CLIs. Tool calls are validated structured-output requests executed by the client (e.g. Hermes), never by the wrapper. This is a prompt bridge, not native provider function calling.")
    app.state.config = config

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
            response = JSONResponse(None, status_code=204, headers={"Access-Control-Allow-Methods": "GET, POST, OPTIONS", "Access-Control-Allow-Headers": "Authorization, Content-Type", "Access-Control-Max-Age": "600"})
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

    async def authorize(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(auth_scheme)]):
        supplied = hashlib.sha256((credentials.credentials if credentials else "").encode()).digest()
        if not credentials or not hmac.compare_digest(key_hash, supplied):
            raise ApiError(401, "invalid_api_key", "A valid Bearer API key is required.")
        if stopping:
            raise ApiError(503, "server_stopping", "Server is shutting down.")

    @app.get("/health", tags=["Service"])
    async def health():
        """Liveness only; does not verify upstream login or model availability."""
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[Depends(authorize)], tags=["OpenAI"])
    async def list_models():
        return {"object": "list", "data": [{"id": m.id, "object": "model", "created": 0, "owned_by": m.provider} for m in models.values()]}

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

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)], tags=["OpenAI"], openapi_extra=request_schema)
    async def completions(request: Request):
        nonlocal total, uploads
        if uploads >= config.server.max_concurrent * 4:
            raise ApiError(429, "server_busy", "Too many pending requests. Retry later.")
        uploads += 1
        try:
            try:
                body = validate_request(await _body(request, config.server.max_body_bytes))
            except ProtocolError as error:
                raise ApiError(400, error.code, str(error), error.param) from None
            model = models.get(body["model"])
            if not model:
                raise ApiError(404, "model_not_found", "Unknown or disabled model; use GET /v1/models.", "model")
            provider = config.providers[model.provider]
            if total >= config.server.max_concurrent or active.get(model.provider, 0) >= provider.max_concurrent:
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
            try:
                result = await runner(provider, model, format_prompt(body), on_text=text_delta,
                                      secret_env=config.server.api_key_env)
                try:
                    parsed = parse_response(result.text, body)
                except ProtocolError as error:
                    raise ApiError(502, "invalid_model_output", str(error)) from None
                await queue.put(("result", (parsed, result.usage)))
            except asyncio.CancelledError:
                raise
            except ApiError as error:
                log.warning("request=%s model=%s code=%s", request.state.request_id, model.id, error.code)
                await queue.put(("error", error))
            except Exception:
                log.error("request=%s model=%s code=internal_error", request.state.request_id, model.id)
                await queue.put(("error", ApiError(500, "internal_error", "An internal server error occurred.")))
            finally:
                log.info("request=%s model=%s duration_ms=%d", request.state.request_id, model.id, (time.monotonic() - started) * 1000)

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
        headers = {"X-Wrapper-Tool-Mode": "prompt-bridge", "X-Wrapper-Generation-Controls": "advisory"}

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

    return app
