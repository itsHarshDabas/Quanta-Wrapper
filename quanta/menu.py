from __future__ import annotations

import asyncio
import copy
import logging
import threading

log = logging.getLogger("quanta.menu")


class SessionMenu:
    """Interactive server-console menu to switch provider/model mid-session.

    Runs an ``input()`` loop on a daemon thread so ``serve`` keeps serving.
    Every accepted switch rebuilds the app's routing map via
    ``app.state.apply_dynamic_model`` (in-memory only; files untouched).
    """

    def __init__(self, app, config, refresh_known=None):
        self._app = app
        self._config = config
        self._refresh_known = refresh_known
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def current(self) -> str:
        return getattr(self._app.state, "dynamic_label", "config file")

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="quanta-menu", daemon=True)
        self._thread.start()
        print('Menu: type "menu" + Enter in this console to change client/model mid-session.')

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                line = input().strip().lower()
            except EOFError:
                return
            if line not in ("menu", "m", "switch", "model"):
                continue
            try:
                self._interactive_switch()
            except (ValueError, EOFError) as error:
                print(f"Switch cancelled: {error}")

    def _interactive_switch(self) -> None:
        from .picker import _provider_models, _resolve_providers, compose_dynamic_model, pick_from_list

        usable = [n for n, p in self._config.providers.items()
                  if p.adapter not in __import__("quanta.config", fromlist=["BRIDGE_NOTES"]).BRIDGE_NOTES]
        resolved = _resolve_providers(self._config)
        labels = [f"{n} ({self._config.providers[n].adapter}) — {resolved.get(n) or 'not found on PATH'}"
                  for n in usable]
        provider_name = usable[pick_from_list("Select a client (provider):", labels)]
        from .picker import choose_model

        known = list(self._refresh_known(provider_name) if self._refresh_known else _provider_models(provider_name))
        upstream = choose_model(provider_name, known, reader=lambda prompt: input(f"{prompt}: "))
        if not upstream:
            print("Cancelled; keeping current model.")
            return
        self.switch(provider_name, upstream)

    def switch(self, provider_name: str, upstream: str, alias: str | None = None) -> str:
        """Apply a provider/model switch now; returns the new alias id."""
        from .picker import compose_dynamic_model

        validated, alias_id = compose_dynamic_model(self._config, provider_name=provider_name,
                                                    upstream=upstream, alias=alias)
        apply = getattr(self._app.state, "apply_dynamic_model", None)
        if apply is None:  # pragma: no cover - wired in create_app
            raise RuntimeError("App does not support mid-session switching.")
        apply(validated, alias_id)
        print(f"Switched to {alias_id!r} on {provider_name!r} (upstream {upstream.strip()!r}).")
        return alias_id


async def stdin_menu_watcher(app, config, refresh_known=None) -> None:
    """Async stdin watcher alternative (used only when explicitly awaited)."""
    menu = SessionMenu(app, config, refresh_known)
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, input)
        if line.strip().lower() in ("menu", "m", "switch", "model"):
            try:
                menu._interactive_switch()
            except (ValueError, EOFError) as error:
                print(f"Switch cancelled: {error}")
