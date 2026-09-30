# Global API Wrapper

OpenAI-compatible FastAPI bridge for local AI CLIs, including
client-side tool calling via a structured-output prompt bridge.

The wrapper spawns a configured CLI per request, sends the conversation
as a text prompt, validates the model's structured tool-call envelope,
and returns an OpenAI-style completion. Tool calls are **returned to
the client for execution** (e.g. Hermes) — the wrapper never executes
them and never runs shell commands from transcripts.


## Quickstart

```bash
pip install -e ".[test]"
global-api init                    # creates wrapper.config.json + random apiKey
global-api doctor                  # resolve CLI executables, no model calls
global-api serve                   # http://127.0.0.1:8787/v1
```

Authenticate every request with the configured key:

```bash
curl http://127.0.0.1:8787/v1/models \
  -H "Authorization: Bearer $GLOBAL_API_KEY"
curl http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer $GLOBAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"omnirush","messages":[{"role":"user","content":"Hello"}]}'
```

`GLOBAL_API_KEY` overrides `server.apiKey` when set. Keep
`wrapper.config.json` private — it holds the API key (0600 on init).

## Endpoints

- `GET /health` — liveness, no auth.
- `GET /v1/models` — enabled model aliases, Bearer auth.
- `POST /v1/chat/completions` — OpenAI Chat Completions subset, Bearer auth.
  Supports `stream: true` (SSE), `tools` + `tool_choice`, and
  `stream_options: {"include_usage": true}`. Only text content is
  supported; images/audio and legacy `functions`/`function_call` are
  rejected. The tool bridge is a structured-output prompt protocol —
  disable native CLI tools/permissions separately.

## Configuration

See `wrapper.config.example.json`. Key fields:

- `server`: `host`, `port`, `apiKeyEnv`, `apiKey`, `maxBodyBytes`,
  `maxConcurrent`, `corsOrigins` (exact `http(s)` origins, no wildcards).
- `defaults`: `timeoutMs`, `maxOutputBytes`, `maxConcurrent`, `workspace`.
- `providers.<name>`: `adapter` (`omnirush|opencode|cline|antigravity|freebuff|custom`),
  `enabled`, `command`, plus `args`/`env` passthrough. `custom` also needs
  `promptMode` (`stdin|arg`), `outputMode` (`text|jsonl`), and `{prompt}` /
  `{model}` placeholders.
- `models`: alias list mapping `id` to `provider`, with optional
  `upstreamModel` / `upstreamProvider`.

Native presets run with tools disabled (`--no-tools`-style flags or deny
policies) and pass the prompt over stdin. Antigravity requires Google
`agy >= 1.1.15` with deny permissions plus `acknowledgeAgentRisk: true`.
FreeBuff has no verified headless interface — configure a `custom` bridge.

## Development

```bash
python -m pytest -q
global-api doctor
```
