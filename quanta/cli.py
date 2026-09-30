from __future__ import annotations

import argparse
import logging
import os
import socket
from pathlib import Path

from .commands import resolve_command
from .config import BRIDGE_NOTES, initialize_config, load_config
from .tty import stdin_is_interactive


QUANTA_CONFIG = "quanta.config.json"
LEGACY_CONFIG = "wrapper.config.json"


def _default_config() -> str:
    """Prefer quanta.config.json, but keep serving an existing legacy file."""
    if not Path(QUANTA_CONFIG).exists() and Path(LEGACY_CONFIG).exists():
        return LEGACY_CONFIG
    return QUANTA_CONFIG


def main():
    parser = argparse.ArgumentParser(prog="quanta", description="Quanta — expose local AI CLIs as an authenticated OpenAI-compatible API")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [("init", "Create a configuration and random API key"), ("doctor", "Resolve CLI executables without model calls"), ("serve", "Run the FastAPI server (interactive client/model picker)"), ("chat", "One-shot prompt against a chosen client/model (no server)"), ("switch", "Rewrite the configuration to one client/model (persisted)"), ("ui", "Serve the web UI on its own local port (talks to a running `quanta serve`)")]:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", "-c", default=None, help=f"Configuration path (default: {QUANTA_CONFIG}, else {LEGACY_CONFIG})")
        if name in ("serve", "chat", "switch"):
            command.add_argument("--provider", help="Skip the client list and use this provider directly")
            command.add_argument("--model", help="Skip the model prompt and use this upstream model id directly")
            command.add_argument("--alias", help="Alias id to expose (default: <provider>-<model slug>)")
        if name == "serve":
            command.add_argument("--host", help="Use 0.0.0.0 to listen on the network")
            command.add_argument("--port", type=int)
            command.add_argument("--no-menu", action="store_true", help="Disable the mid-session 'menu' console switcher")
        if name in ("serve", "ui"):
            command.add_argument("--ui-port", type=int, default=8788, help="Port for the web UI (default 8788)")
        if name == "serve":
            command.add_argument("--ui", action="store_true", help=argparse.SUPPRESS)  # UI is on by default
            command.add_argument("--no-ui", action="store_true", help="Do not serve the web UI")
            command.add_argument("--mode", choices=("cli", "gui"), help="Skip the startup menu: cli = terminal setup, gui = open the browser UI")
            command.add_argument("--no-browser", action="store_true", help="With gui mode, print the UI address instead of opening a browser")
        if name == "chat":
            command.add_argument("prompt", nargs="?", help="Prompt text (reads stdin when omitted)")
    args = parser.parse_args()
    if args.config is None:
        args.config = _default_config()
    try:
        if args.command == "init":
            initialize_config(args.config)
            print(f"Created {Path(args.config).resolve()}")
            print("A random API key is stored in server.apiKey; keep this file private.")
            print("Next: quanta doctor, then quanta serve")
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
        if args.command == "ui":
            from .uiserver import make_ui_server
            server = make_ui_server(f"http://{config.server.host}:{config.server.port}", port=args.ui_port)
            print(f"Quanta UI: http://127.0.0.1:{args.ui_port}  (API: http://{config.server.host}:{config.server.port}/v1)")
            print(f"Allow this origin in server.corsOrigins: http://127.0.0.1:{args.ui_port}")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            return
        from .picker import interactive_serve_selection, persist_selection, pick_from_list
        if args.command == "switch":
            selected = interactive_serve_selection(config, provider_arg=args.provider,
                                                   model_arg=args.model, alias_arg=args.alias)
            persist_selection(args.config, selected)
            alias = selected.exposed_models()[0].id
            print(f"Saved to {Path(args.config).resolve()}: serving {alias!r}. "
                  "Restart `quanta serve` to apply.")
            return
        if args.command == "chat":
            import asyncio
            import sys
            from .protocol import format_prompt, validate_request
            from .runner import run_cli
            prompt = args.prompt or (sys.stdin.read() if not stdin_is_interactive() else "")
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
        interactive = stdin_is_interactive()
        mode = args.mode
        if mode is None and (args.provider is not None or args.model is not None):
            mode = "cli"
        if mode is None and interactive:
            choice = pick_from_list("How do you want to set up Quanta?", [
                "CLI — choose the client and model here in the terminal",
                f"GUI — open the browser UI (http://127.0.0.1:{args.ui_port}) to set up and test"])
            mode = "cli" if choice == 0 else "gui"
        if mode == "gui":
            print("GUI mode: serving the configuration file; choose providers and models in the browser.")
            selected = config
        elif mode is None:
            print("Non-interactive stdin: using config-file models (no picker).")
            print("Tip: `quanta serve --mode gui`, or `--provider <name> --model <upstream-id>`.")
            selected = config
        else:
            selected = interactive_serve_selection(config, provider_arg=args.provider,
                                                   model_arg=args.model, alias_arg=args.alias)
        from .app import create_app
        import uvicorn
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        app = create_app(selected, base_config=config)
        menu = None
        # The menu reads the server console's stdin. `--no-menu` (or a
        # non-interactive console) disables it; the admin HTTP API below
        # always stays available for programmatic mid-session switches.
        if not args.no_menu and interactive and mode != "gui":
            from .menu import SessionMenu
            menu = SessionMenu(app, config)
            menu.start()
        else:
            print("Mid-session menu disabled; use the admin API: POST /v1/session/switch.")
        if not args.no_ui:
            from .uiserver import Handoff, start_ui_thread
            handoff = Handoff(selected.server.api_key) if mode == "gui" and not args.no_browser else None
            ui_origin = None
            api_base = f"http://{selected.server.host}:{selected.server.port}"
            for port in range(args.ui_port, args.ui_port + 10):  # first free port
                try:
                    start_ui_thread(api_base, port=port, handoff=handoff)
                    ui_origin = f"http://127.0.0.1:{port}"
                    break
                except OSError:
                    continue
            if ui_origin is None:
                print(f"Web UI disabled: ports {args.ui_port}-{args.ui_port + 9} are busy.")
            else:
                # In-memory only: let the bundled UI call the API without editing the config file.
                # `selected` is a separate copy in CLI mode, and it is the one the server enforces.
                for origin in (ui_origin, ui_origin.replace("127.0.0.1", "localhost")):
                    for target in (config, selected):
                        if origin not in target.server.cors_origins:
                            target.server.cors_origins.append(origin)
                print(f"Web UI:     {ui_origin}  (paste your API key there once)")
                if handoff is not None:
                    import threading
                    import webbrowser
                    # Only a one-time nonce is put in the URL (fragment: not sent to the server, not logged);
                    # the page exchanges it for the key over loopback. The key is never a URL or argv item.
                    url = f"{ui_origin}/#h={handoff.issue()}"
                    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
                    print("Opening your browser…")
        print(f"Quanta API: http://{selected.server.host}:{selected.server.port}/v1")
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
