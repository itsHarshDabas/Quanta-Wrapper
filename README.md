# Quanta

OpenAI-compatible FastAPI bridge for local AI CLIs, including
client-side tool calling via a structured-output prompt bridge.

Quanta spawns a configured CLI per request, sends the conversation
as a text prompt, validates the model's structured tool-call envelope,
and returns an OpenAI-style completion. Tool calls are **returned to
the client for execution** (e.g. Hermes) — Quanta never executes
them and never runs shell commands from transcripts.


## Quickstart

```bash
pip install -e ".[test]"
quanta init                        # creates quanta.config.json + random apiKey
quanta doctor                      # resolve CLI executables, no model calls
quanta serve                       # picker: list a client, type a model id
```

`global-api` remains installed as a deprecated alias for `quanta`, and
`quanta.config.json` / a legacy `wrapper.config.json` are both accepted
(the legacy file is used automatically when the new one is absent).

`serve` opens an interactive picker: choose the client from a numbered
list, then enter the upstream model id (a numbered known-model list is
shown for omnirush/opencode, or type any id — e.g. `provider/model` for
opencode). The selection is in-memory only; config files are never
modified. Non-interactive equivalent:

```bash
quanta serve --provider opencode --model opencode/mymodel --alias myalias
quanta chat --provider opencode --model opencode/mymodel "Say hi"
```

Mid-session switching (no restart, no file edits):

- Server console: type `menu` + Enter, pick a new client, enter a new
  model id (`--no-menu` disables this; piping stdin disables it too).
- HTTP admin API (same Bearer key): `GET /v1/session`,
  `POST /v1/session/switch {"provider": "...", "model": "..."}`.
- Persist a choice to the config file (keeps your API key and provider
  args): `quanta switch` (picker), or `quanta switch -p opencode -m opencode/mymodel`.
- In-flight requests finish on the old mapping; new requests use the new one.

Authenticate every request with the configured key:

```bash
curl http://127.0.0.1:8787/v1/models \
  -H "Authorization: Bearer $QUANTA_API_KEY"
curl http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer $QUANTA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"omnirush","messages":[{"role":"user","content":"Hello"}]}'
```

`QUANTA_API_KEY` overrides `server.apiKey` when set (`GLOBAL_API_KEY` is
still read as a fallback). Keep `quanta.config.json` private — it holds
the API key (0600 on init).

## Endpoints

- `GET /health` — liveness, no auth.
- `GET /v1/models` — enabled model aliases, Bearer auth.
  Dynamic mode exposes exactly one alias (`<provider>-<model slug>` unless
  `--alias` overrides); the response includes `upstream_model`.
- `GET /v1/session` — current in-memory selection, Bearer auth.
- `POST /v1/session/switch` — `{"provider","model","alias?"}` switches the
  served client/model mid-session without touching files, Bearer auth.
- `POST /v1/chat/completions` — OpenAI Chat Completions subset, Bearer auth.
  Supports `stream: true` (SSE), `tools` + `tool_choice`, and
  `stream_options: {"include_usage": true}`. Only text content is
  supported; images/audio and legacy `functions`/`function_call` are
  rejected. The tool bridge is a structured-output prompt protocol —
  disable native CLI tools/permissions separately.

## Configuration

See `quanta.config.example.json`. Key fields:

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

## Client status (verified locally)

| Client | Adapter | Invocation | Status |
| --- | --- | --- | --- |
| OmniRush | `omnirush` | `omnirush --print --mode json ...` (stdin prompt, tools off) | wiring OK; needs CLI login |
| Opencode | `opencode` | `opencode run --pure --dir <ws> --model <id>` (stdin prompt, plain text) | working (tested `opencode/muse-spark-1.3-contributor-free`) |
| Cline | `cline` | `cline <prompt> --json --plan --auto-approve false ...` | wiring OK; needs Cline re-auth |
| Antigravity | `antigravity` | `agy --input-format stream-json --output-format stream-json ...` | handshake OK; needs deny permissions + `acknowledgeAgentRisk` |
| FreeBuff | `freebuff` | none — interactive TUI only (no prompt/JSON mode) | blocked by design; use `custom` bridge |

`quanta doctor` verifies paths only, never login/models.

## Development

```bash
python -m pytest -q
quanta doctor
```
