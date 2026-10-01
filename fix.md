# fix.md — why Quanta was not working, and what was fixed

Date: 2026-10-01. Environment: Windows, Python 3.14.2, opencode v2.0.21, cline 3.0.67.

## Symptoms

- `quanta chat --provider opencode --model opencode/muse-spark-1.3-contributor-free "hi"` failed:
  `Error (cli_failed)` with server-side
  `detail='exit code 1: ERRORS Unrecognized flag: --pure in command opencode run
  Unrecognized flag: --dir in command opencode run'`.
- Same failure through the API: `POST /v1/chat/completions` with an opencode model
  returned `502 cli_failed` (verified the API never leaks the detail — it only goes to
  the operator console, as designed).
- `quanta doctor` exited `1` (`omnirush` / `agy` executables not found, both enabled).
- `python -m pytest -q` could not even run at first (package not installed in this shell).

## Root causes

### 1. OpenCode adapter used flags that do not exist (primary bug)

`quanta/adapters.py` built the invocation as

```python
["run", *extra, "--pure", "--dir", provider.workspace, *model_args]
```

but the installed `opencode v2.0.21` has neither `--pure` nor `--dir`
(`opencode run --help` lists `--model, --agent, --format, --file, --title,
--thinking, --auto, --continue, --session, --fork, --standalone, --server` only).
Every OpenCode request therefore died before any model call. The code (and a test
asserting `--pure`, plus two README lines) was written against an older CLI.

### 2. OpenCode fails when cwd is inside this repo

Even with correct flags, `opencode run --model <id>` fails with

```text
Error: Instruction initialization blocked by unavailable sources: core/instructions
```

(server log: `failed to activate instruction source ... Maximum call stack size exceeded`)
whenever cwd is anywhere under `C:\Projects kolij\quanta-wrapper` — including the
configured `./workspace` (empty, git-ignored) and a fresh empty probe dir — while the
identical command succeeds in a clean temp dir, in a path with spaces, and in a fresh
git repo. OpenCode treats cwd as its project directory and walks up to the enclosing
repo, so running inside this checkout (huge `venv/`, caches, docs) breaks its
instruction loading and would also expose wrapper sources to the model.

### 3. Local config enabled CLIs that are not installed (environmental)

`quanta.config.json` (git-ignored local file) had `omnirush` and `antigravity`
`enabled: true`, but neither `omnirush` nor `agy` is on PATH here, so `quanta doctor`
always exited 1 and the server exposed aliases that can only ever return
`503 cli_unavailable`. (`freebuff` was already correctly disabled — TUI-only by design.)

## Fixes (code)

- `quanta/adapters.py` — OpenCode invocation is now `["run", *extra, "--model", <id>]`
  with the prompt over stdin. No `--pure`, no `--dir`.
- `quanta/runner.py` — `run_cli` now runs the **opencode adapter in the fresh,
  empty per-request temp dir** (`state_dir`, outside any repo) instead of the shared
  `provider.workspace`; `PWD` follows `cwd`. All other adapters still use
  `provider.workspace`. This matches the stateless design (native CLI sessions are
  never reused) and keeps wrapper sources out of the model's project context.
- `tests/test_runner.py` — updated `test_opencode_adapter_is_plain_text_over_stdin`
  (asserts `run` first, no `--pure`/`--dir`) and added
  `test_opencode_runs_in_isolated_directory` (cwd/PWD ≠ shared workspace for opencode).
- `README.md` — client-status table now shows `opencode run --model <id>`; the
  `cli_failed` troubleshooting entry no longer blames a missing `--pure` and documents
  the isolated-cwd behavior plus the `core/instructions` symptom.
- `quanta.config.json` (local only, git-ignored) — disabled `omnirush` and
  `antigravity` to match the CLIs actually installed on this machine.

## Verification

- `python -m pytest -q` → **142 passed** (was 141; +1 new isolation test), 221 subtests.
- `quanta doctor` → exit **0**; exposed aliases: `opencode, cline`.
- `quanta chat --provider opencode --model opencode/muse-spark-1.3-contributor-free`
  → works (previously `cli_failed`).
- Against `quanta serve` (full config file): `scripts/live_check.py --models opencode,cline`
  → health, models, providers, auth-401, unknown-model-404, opencode chat + streaming,
  cline chat + streaming, opencode tool-call round trip + streamed tool call +
  `tool_choice=none`, cline tool-call round trip — **all PASS**, plus the remaining
  cline streamed-tool/`tool_choice=none` checks run separately — **PASS**.
- Remaining `live_check.py` items (antigravity/omnirush chats, provider/model switch
  chains) require the `agy`/`omnirush` binaries, which are not installed here —
  environmental, unrelated to these fixes.

## Notes / not changed

- After force-killing a streaming client mid-request, the provider slot can briefly
  report `429 provider_busy` until the orphaned CLI run finishes; it recovers on its
  own (verified with an abandon-then-retry probe). Consider surfacing busier-state
  diagnostics if Hermes hits this with short timeouts.
- `Req.txt` is a `pip freeze` snapshot (fine as-is); fresh installs should keep using
  `pip install -e .[test]` per the README quickstart.
