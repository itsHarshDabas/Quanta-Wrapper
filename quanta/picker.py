"""Interactive provider/model selection for `serve` (no file edits)."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

OMNIRUSH_KNOWN_MODELS = [
    "deepseek-v3.2",
    "glm-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gemini-3-flash",
    "gemini-3-pro",
    "kimi-k2.5",
    "minimax-m2.1",
    "qwen3-max",
]


def _read_choice(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        return default
    return answer or default


def pick_from_list(title: str, options: list[str]) -> int:
    """Numbered-list picker returning the chosen index; repeats until valid."""
    print(f"\n{title}")
    for index, option in enumerate(options, 1):
        print(f"  {index}. {option}")
    while True:
        answer = _read_choice(f"Enter number (1-{len(options)})")
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        print(f"Please enter a number between 1 and {len(options)}.")


def _resolve_providers(config) -> dict[str, str]:
    """Best-effort CLI resolution for the picker display; never raises."""
    from .commands import resolve_command

    resolved: dict[str, str] = {}
    for name, provider in config.providers.items():
        try:
            found = resolve_command(provider.command, {**os.environ, **provider.env})
            parts = [found.executable, *found.prefix]
            resolved[name] = " ".join(p for p in parts if p)
        except (ValueError, OSError):
            resolved[name] = ""
    return resolved

def _provider_models(provider_name: str) -> list[str]:
    """Known upstream model ids for a provider (probe first, curated fallback)."""
    import shutil
    import subprocess

    def run(binary: str, args: list[str], timeout: int) -> list[str]:
        try:
            completed = subprocess.run(
                [binary, *args], capture_output=True, text=True, timeout=timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            return []
        if completed.returncode != 0:
            return []
        return [line.strip() for line in completed.stdout.splitlines() if line.strip()]

    if provider_name == "omnirush":
        binary = shutil.which("omnirush")
        if binary:
            for flag in ("models", "--models", "list"):
                ids = [line.split()[0] for line in run(binary, [flag], 20)
                       if line and not line.startswith(("#", "-", "="))]
                if len(ids) >= 2:
                    return ids
        return list(OMNIRUSH_KNOWN_MODELS)
    if provider_name == "opencode":
        binary = shutil.which("opencode")
        if binary:
            return run(binary, ["models"], 30)
        return []
    return []


def _suggest_model_id(provider_name: str, upstream: str) -> str:
    slug = upstream.replace("/", "-").replace(":", "-").replace(" ", "-")
    candidate = f"{provider_name}-{slug}"[:200] or provider_name
    if not candidate[0].isalnum():
        candidate = f"{provider_name}-{candidate}"[:200]
    return candidate



def compose_dynamic_model(config, *, provider_name: str, upstream: str, alias: str | None = None):
    """Build a validated config copy with one dynamic model alias (no file edits).

    Enables exactly ``provider_name`` and exposes a single alias pointing at
    ``upstream``. Raises ``ValueError`` for unknown/unusable providers, empty
    model ids, or invalid composed configs.
    """
    from .config import BRIDGE_NOTES, ModelAlias, validate_config

    usable = [name for name, item in config.providers.items() if item.adapter not in BRIDGE_NOTES]
    if provider_name not in config.providers or provider_name not in usable:
        raise ValueError(f"Unknown or unusable provider: {provider_name!r} (choices: {', '.join(usable)})")
    upstream = (upstream or "").strip()
    if not upstream or "\0" in upstream:
        raise ValueError("A model id is required.")
    provider = config.providers[provider_name]
    if provider.adapter == "custom" and not any("{model}" in arg for arg in provider.args):
        raise ValueError("The custom provider args must include {model} for dynamic model selection.")
    alias_id = (alias or "").strip() or _suggest_model_id(provider_name, upstream)
    selected = copy.deepcopy(config)
    for name, item in selected.providers.items():
        item.enabled = (name == provider_name)
    selected.models = [ModelAlias(id=alias_id, provider=provider_name, upstream_model=upstream)]
    raw = {"server": selected.server.model_dump(by_alias=True),
           "defaults": selected.defaults.model_dump(by_alias=True),
           "providers": {n: p.model_dump(by_alias=True) for n, p in selected.providers.items()},
           "models": [{"id": alias_id, "provider": provider_name, "upstreamModel": upstream}]}
    base = Path(provider.workspace)
    validated = validate_config(raw, base.parent if base.is_absolute() else ".")
    return validated, alias_id


def interactive_serve_selection(config, *, provider_arg: str | None = None,
                                model_arg: str | None = None, alias_arg: str | None = None):
    """List-picker for provider + free-text model id; returns a validated copy.

    ``provider_arg`` / ``model_arg`` / ``alias_arg`` skip the matching prompt
    (non-interactive use). Never modifies config files.
    """
    from .config import BRIDGE_NOTES

    provider_name = provider_arg
    if provider_name is None:
        usable = [n for n, p in config.providers.items() if p.adapter not in BRIDGE_NOTES]
        resolved = _resolve_providers(config)
        labels = []
        for name in usable:
            provider = config.providers[name]
            status = "enabled" if provider.enabled else "disabled"
            exe = resolved.get(name) or "not found on PATH"
            labels.append(f"{name} ({provider.adapter}, currently {status}) — {exe}")
        provider_name = usable[pick_from_list("Select a client (provider):", labels)]
        print(f"Selected client: {provider_name}")

    upstream = (model_arg or "").strip() if model_arg is not None else ""
    if not upstream and model_arg is None:
        known = _provider_models(provider_name)
        if known:
            print(f"\nKnown {provider_name} models:")
            for index, item in enumerate(known, 1):
                print(f"  {index}. {item}")
            print("  Or type any other model id directly.")
            upstream = _read_choice("Enter model id (number or full id)")
            if upstream.isdigit() and 1 <= int(upstream) <= len(known):
                upstream = known[int(upstream) - 1]
        else:
            hint = ' (e.g. "provider/model" for opencode)' if provider_name == "opencode" else ""
            upstream = _read_choice(f"Enter upstream model id for {provider_name}{hint}")
    if provider_name == "opencode" and upstream and "/" not in upstream:
        print('Note: opencode model ids are usually "provider/model" (see `opencode models`). Continuing anyway.')
    validated, alias_id = compose_dynamic_model(config, provider_name=provider_name,
                                                upstream=upstream, alias=alias_arg)
    print(f"Serving model {alias_id!r} on provider {provider_name!r} (upstream {upstream!r}).")
    print("Config files were not modified; this selection lives only for this process.")
    return validated


def persist_selection(config_path: str | Path, validated) -> None:
    """Write a composed selection back to the config file, keeping the API key.

    Only `providers[].enabled` and `models` are rewritten; every other field
    (server settings, apiKey, provider args/env/timeouts) is left untouched.
    """
    path = Path(config_path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    for name, provider in validated.providers.items():
        if name in raw.get("providers", {}):
            raw["providers"][name]["enabled"] = provider.enabled
    raw["models"] = [{"id": m.id, "provider": m.provider,
                      **({"upstreamModel": m.upstream_model} if m.upstream_model else {}),
                      **({"upstreamProvider": m.upstream_provider} if m.upstream_provider else {})}
                     for m in validated.models]
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
