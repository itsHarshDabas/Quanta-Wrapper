from __future__ import annotations

import argparse
import logging
import os
import socket
from pathlib import Path

from .commands import resolve_command
from .config import BRIDGE_NOTES, initialize_config, load_config


def main():
    parser = argparse.ArgumentParser(description="Expose local AI CLIs as an authenticated OpenAI-compatible API")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [("init", "Create a configuration and random API key"), ("doctor", "Resolve CLI executables without model calls"), ("serve", "Run the FastAPI server (interactive client/model picker)"), ("chat", "One-shot prompt against a chosen client/model (no server)")]:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", "-c", default="wrapper.config.json", help="Configuration path (default: wrapper.config.json)")
        if name in ("serve", "chat"):
            command.add_argument("--provider", help="Skip the client list and use this provider directly")
            command.add_argument("--model", help="Skip the model prompt and use this upstream model id directly")
            command.add_argument("--alias", help="Alias id to expose (default: <provider>-<model slug>)")
        if name == "serve":
            command.add_argument("--host", help="Use 0.0.0.0 to listen on the network")
            command.add_argument("--port", type=int)
            command.add_argument("--no-menu", action="store_true", help="Disable the mid-session 'menu' console switcher")
        if name == "chat":
            command.add_argument("prompt", nargs="?", help="Prompt text (reads stdin when omitted)")
    args = parser.parse_args()
    try:
        if args.command == "init":
            initialize_config(args.config)
            print(f"Created {Path(args.config).resolve()}")
            print("A random API key is stored in server.apiKey; keep this file private.")
            print("Next: global-api doctor, then global-api serve")
            return
        overrides = {key: getattr(args, key) for key in ("host", "port") if getattr(args, key, None) is not None}
        config = load_config(args.config, overrides)
        if args.command == "doctor":
            failures = 0
            for name, provider in config.providers.items():
                if provider.adapter in BRIDGE_NOTES:
                    print(f"{name}: BRIDGE REQUIRED — {BRIDGE_NOTES[provider.adapter]}")
                    continue
                try:
                    resolved = resolve_command(provider.command, {**os.environ, **provider.env})
                    print(f"{name}: {'ENABLED' if provider.enabled else 'disabled'} — {resolved.executable} {' '.join(resolved.prefix)}")
                except (ValueError, OSError) as error:
                    print(f"{name}: {'ERROR' if provider.enabled else 'disabled / unavailable'} — {error}")
                    failures += int(provider.enabled)
                if provider.adapter == "antigravity":
                    print("  Requires Google agy >=1.1.15 (not the antigravity IDE launcher), with deny permissions configured.")
            models = config.exposed_models()
            print("Exposed aliases: " + (", ".join(m.id for m in models) or "(none)"))
            print("Tool calling: structured-output prompt bridge; Hermes executes the returned tool calls.")
            print("Doctor checks paths only, not login, CLI version/protocol, permissions, or model availability.")
            raise SystemExit(1 if failures or not models else 0)
        from .picker import interactive_serve_selection
        if args.command == "chat":
            import asyncio
            import sys
            from .protocol import format_prompt, validate_request
            from .runner import run_cli
            prompt = args.prompt or (sys.stdin.read() if not sys.stdin.isatty() else "")
            if not (prompt or "").strip():
                prompt = input("Enter prompt: ")
            selected = interactive_serve_selection(config, provider_arg=args.provider,
                                                   model_arg=args.model, alias_arg=args.alias)
            model = selected.exposed_models()[0]
            provider = selected.providers[model.provider]
            print(f"Running {model.id!r} via {model.provider} ...")
            result = asyncio.run(run_cli(provider, model, format_prompt(validate_request(
                {"model": model.id, "messages": [{"role": "user", "content": prompt}]}))))
            print(result.text)
            return
        non_interactive = args.provider is not None or args.model is not None or not os.isatty(0)
        if non_interactive and args.provider is None and args.model is None:
            print("Non-interactive stdin: using config-file models (no picker).")
            print("Tip: skip the prompts with `global-api serve --provider <name> --model <upstream-id>`.")
            selected = config
        else:
            selected = interactive_serve_selection(config, provider_arg=args.provider,
                                                   model_arg=args.model, alias_arg=args.alias)
        from .app import create_app
        import uvicorn
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        app = create_app(selected)
        menu = None
        # The menu reads the server console's stdin. `--no-menu` (or a
        # non-interactive console) disables it; the admin HTTP API below
        # always stays available for programmatic mid-session switches.
        if not args.no_menu and os.isatty(0):
            from .menu import SessionMenu
            menu = SessionMenu(app, config)
            menu.start()
        else:
            print("Mid-session menu disabled; use the admin API: POST /v1/session/switch.")
        print(f"API: http://{selected.server.host}:{selected.server.port}/v1")
        print(f"Swagger docs: http://{selected.server.host}:{selected.server.port}/docs")
        print("Bearer authentication is required. API key is not logged.")
        print("Mid-session: type `menu` in this console, or POST /v1/session/switch.")
        if selected.server.host not in ("127.0.0.1", "::1", "localhost"):
            print("WARNING: Network access enabled. Restrict firewall access; use HTTPS or a VPN on untrusted networks.")
            try:
                for addr in sorted({record[4][0] for record in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}):
                    print(f"LAN candidate: http://{addr}:{selected.server.port}/v1")
            except OSError:
                pass
        # One worker owns capacity limits and CLI cancellation. Reload/multiple
        # workers would duplicate capacity and can break Windows subprocess loops.
        try:
            uvicorn.run(app, host=selected.server.host, port=selected.server.port, loop="asyncio", workers=1,
                        access_log=False, timeout_keep_alive=5, timeout_graceful_shutdown=8,
                        limit_concurrency=max(32, selected.server.max_concurrent * 8))
        finally:
            if menu is not None:
                menu.stop()
    except FileExistsError:
        parser.exit(1, "Configuration already exists; it was not overwritten.\n")
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
