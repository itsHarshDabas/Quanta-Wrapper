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

`quanta serve` first asks how to set up:

- **CLI** — pick the client and model in the terminal (below).
- **GUI** — starts the API and the web UI and opens your browser, already signed in
  (a one-time, 60-second nonce is exchanged for the key over loopback; the key is never in a URL or process argument). Configure, test and switch there.

Skip the question with `--mode cli|gui` (`--no-browser` prints the UI address instead of
opening it). Piped/non-interactive runs skip it and serve the config file.

The CLI path opens an interactive picker: choose the client from a numbered
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
  args): `quanta switch` (picker), or `quanta switch --provider opencode --model opencode/mymodel`.
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
- `GET /v1/providers` — each CLI, whether it is enabled, and its capabilities
  (streaming, tool_calling, subagents, session_persistence, model_switching,
  provider_switching). Unsupported capabilities are reported as `false`.
- `GET /v1/providers/{name}/models` — upstream model ids the CLI can list
  (OmniRush, OpenCode, Antigravity; cached 60 s).
- `GET /v1/requests` — last 100 completions (metadata only, no prompts/outputs).
- `GET /v1/config` — secret-free view of the running configuration.
- `GET /v1/session` — current in-memory selection, Bearer auth.
- `POST /v1/session/switch` — `{"provider","model","alias?"}` switches the
  served client/model mid-session without touching files, Bearer auth.
- `POST /v1/chat/completions` — OpenAI Chat Completions subset, Bearer auth.
  Supports `stream: true` (SSE), `tools` + `tool_choice`, and
  `stream_options: {"include_usage": true}`. Only text content is
  supported; images/audio and legacy `functions`/`function_call` are
  rejected. The tool bridge is a structured-output prompt protocol —
  disable native CLI tools/permissions separately.

## Finding a model in a long list

OpenCode alone lists 450+ models. Everywhere a model is chosen you can search instead of scrolling:

- **CLI picker / `menu`:** type part of a name (`gpt 5`, `free`, `nemotron`). Every word must match, case-insensitive.
  Pick by number, or a single match is selected for you. Text that matches nothing can be used as a custom id after a `y`.
- **Web UI:** the model box is a searchable dropdown (arrow keys + Enter, or click) in the Playground and the server-default form.
- **API:** `GET /v1/providers/{name}/models?q=gpt%205&limit=20` returns `total` and `matched` counts.

## Switching CLI and model inside one conversation

The API is stateless: the client resends the conversation each turn, so history
and system prompts always carry over. To change CLI or model between turns, just
change the `model` field to `<provider>:<upstream-model-id>` (first colon splits;
use `default` for a CLI's own default model). Configured aliases still work.

```text
turn 1  model=opencode:opencode/big-pickle
turn 2  model=antigravity:gemini-3.8-flash-low   # same messages array, new CLI
turn 3  model=cline:default
```

Only providers with `enabled: true` are routable. FreeBuff cannot be enabled
(no headless interface). Nothing is preserved across a switch except the
messages you send: native CLI sessions are never reused.

## Capabilities (what is and is not supported)

- **Tool calling** is a client-side prompt bridge: the model proposes calls in a
  validated envelope, the client (Hermes) executes them and sends `tool` messages
  back. Quanta does not execute tools and CLI-native tools are disabled. It is not
  native provider function calling.
- **Subagents** are not exposed through this API for any provider.
- **Usage** is reported where the CLI reports it (Cline, Antigravity, OmniRush);
  OpenCode's plain-text mode reports none, so `usage` is omitted.
- Parameters such as `temperature` are advisory prompt hints, not sampler controls.

## Web UI

```bash
quanta serve --ui          # API on :8787, UI on http://127.0.0.1:8788
quanta ui                  # UI only (API must already be running)
```

Add `http://127.0.0.1:8788` to `server.corsOrigins`. Paste your API key in the
Connect section (kept in that browser's localStorage only).

## Hermes

Use an OpenAI-compatible custom endpoint in Hermes `config.yaml`:

```yaml
model:
  provider: custom
  base_url: http://127.0.0.1:8787/v1
  api_key: <server.apiKey from quanta.config.json>
  default: opencode
  api_mode: chat_completions
```

## Claude Code

Quanta also speaks the Anthropic Messages API (`POST /v1/messages`, `POST /v1/messages/count_tokens`;
auth via `x-api-key` or Bearer), so Claude Code can use any enabled CLI as its model. PowerShell:

```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8787"          # no /v1 suffix
$env:ANTHROPIC_AUTH_TOKEN = "<server.apiKey>"
$env:ANTHROPIC_MODEL = "opencode:opencode/big-pickle"       # or a served alias
$env:ANTHROPIC_SMALL_FAST_MODEL = $env:ANTHROPIC_MODEL
$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"
claude
```

- Any `claude-*` model name (what Claude Code asks for by default) is routed to the first served alias;
  the response echoes the requested name.
- Claude Code executes its own tools (Bash, Edit, ...). The CLI model proposes calls through the
  client-side bridge; Quanta runs nothing. Reliability depends on the model following the envelope.
- `count_tokens` is an estimate (about 4 characters per token); CLIs expose no tokenizer.
- Prefer stdin-based providers (OpenCode, Antigravity, OmniRush): Claude Code prompts are large and
  Cline receives its prompt as a command-line argument (Windows limit ~30 000 characters).
- Raise `server.maxBodyBytes` (up to 16 MB) for long sessions; images/documents are not supported.
- Claude Code may print "unrecognized model" and "auto mode" notices for a non-Anthropic gateway; they are harmless.

## Windows notes

- CLIs are resolved from npm `.cmd` shims to `node <entrypoint>` / native `.exe`; no shell is used.
- Timeouts and cancellation kill the whole process tree with `taskkill /T /F`.
- `serve` detects a real console (`GetConsoleMode`): from a script or `< NUL` it skips the
  interactive picker and uses the config file. Use `--provider/--model` to select explicitly.
- Cline runs against its own `~/.cline` state (login required once: `cline auth`).
- Antigravity runs with `--mode plan` (read-only) and requires `acknowledgeAgentRisk: true`.

## Troubleshooting

- `401 invalid_api_key` — wrong Bearer key. `403 origin_not_allowed` — add the browser origin to `corsOrigins`.
- `404 model_not_found` — provider disabled/unknown; see `GET /v1/providers`.
- `502 cli_reported_error ... Upstream: ...` — the CLI's own error (login, quota, unavailable service).
- `502 cli_failed` — the CLI exited with an error. The server console now logs `detail='exit code N: ...'`
  (redacted, never sent to API clients). To see it directly:
  `quanta chat --provider opencode --model <id> "hi"`. Typical causes: a model id that is not in
  `opencode models` for that machine (free models rotate), or not logged in. OpenCode runs each
  request in a fresh empty directory (never the wrapper checkout), because running it inside a
  large repo can fail with `Instruction initialization blocked by unavailable sources`.
  Flags are detected from `opencode run --help` (OpenCode 2.x has no `--pure`/`--dir`; 1.x does), so both work.
  A message like `Unrecognized flag: --xyz` means a CLI changed again: report it with `opencode --version`.
- `504 cli_timeout` — raise `timeoutMs`; the process tree is already killed.

## Live checks

```bash
python scripts/live_check.py                       # OpenAI-client checks against a running server
node scripts/ui_check.js                           # browser-driven UI check (needs Edge + playwright-core)
```

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
FreeBuff has no headless interface (verified: `--help` lists only `login`, `--continue`, `--cwd`).

## Client status (verified locally)

| Client | Adapter | Invocation | Status |
| --- | --- | --- | --- |
| OmniRush | `omnirush` | `omnirush --print --mode json ...` (stdin prompt, tools off) | wiring OK; needs CLI login |
| Opencode | `opencode` | `opencode run [--pure] --model <id>` (stdin prompt, plain text; isolated cwd; `--pure` only if `run --help` lists it) | working (tested `opencode/muse-spark-1.3-contributor-free`) |
| Cline | `cline` | `cline <prompt> --json --plan --auto-approve false ...` | wiring OK; needs Cline re-auth |
| Antigravity | `antigravity` | `agy --input-format stream-json --output-format stream-json ...` | handshake OK; needs deny permissions + `acknowledgeAgentRisk` |
| FreeBuff | `freebuff` | none — interactive TUI only (no prompt/JSON mode) | blocked by design; use `custom` bridge |

`quanta doctor` verifies paths only, never login/models.

## Development

```bash
python -m pytest -q
quanta doctor
```
