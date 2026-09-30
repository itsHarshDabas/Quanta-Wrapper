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
    for name, help_text in [("init", "Create a configuration and random API key"), ("doctor", "Resolve CLI executables without model calls"), ("serve", "Run the FastAPI server")]:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", "-c", default="wrapper.config.json", help="Configuration path (default: wrapper.config.json)")
        if name == "serve":
            command.add_argument("--host", help="Use 0.0.0.0 to listen on the network")
            command.add_argument("--port", type=int)
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
        from .app import create_app
        import uvicorn
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        print(f"API: http://{config.server.host}:{config.server.port}/v1")
        print(f"Swagger docs: http://{config.server.host}:{config.server.port}/docs")
        print("Bearer authentication is required. API key is not logged.")
        if config.server.host not in ("127.0.0.1", "::1", "localhost"):
            print("WARNING: Network access enabled. Restrict firewall access; use HTTPS or a VPN on untrusted networks.")
            try:
                for addr in sorted({record[4][0] for record in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}):
                    print(f"LAN candidate: http://{addr}:{config.server.port}/v1")
            except OSError:
                pass
        # One worker owns capacity limits and CLI cancellation. Reload/multiple
        # workers would duplicate capacity and can break Windows subprocess loops.
        uvicorn.run(create_app(config), host=config.server.host, port=config.server.port, loop="asyncio", workers=1,
                    access_log=False, timeout_keep_alive=5, timeout_graceful_shutdown=8,
                    limit_concurrency=max(32, config.server.max_concurrent * 8))
    except FileExistsError:
        parser.exit(1, "Configuration already exists; it was not overwritten.\n")
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
