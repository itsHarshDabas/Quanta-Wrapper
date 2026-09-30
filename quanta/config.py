from __future__ import annotations

import json
import os
import re
import secrets
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic.alias_generators import to_camel

BRIDGE_NOTES = {
    "freebuff": "FreeBuff's current CLI exposes an interactive TUI, not a verified headless interface. Configure a custom bridge instead.",
}


class SettingsModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid", strict=True)


class ServerSettings(SettingsModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8787, ge=0, le=65535)
    api_key_env: str = "QUANTA_API_KEY"
    api_key: str = Field(default="", repr=False)
    max_body_bytes: int = Field(default=1048576, ge=1024, le=16777216)
    max_concurrent: int = Field(default=4, ge=1, le=64)
    cors_origins: list[str] = Field(default_factory=list)

    @field_validator("cors_origins")
    @classmethod
    def exact_origins(cls, values):
        from urllib.parse import urlsplit
        for origin in values:
            parsed = urlsplit(origin)
            if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.path or parsed.query or parsed.fragment or parsed.username:
                raise ValueError("CORS origins must be exact http(s) origins without paths or wildcards")
            if "*" in origin:
                raise ValueError("wildcard CORS origins are not allowed")
        return values


class Defaults(SettingsModel):
    timeout_ms: int = Field(default=180000, ge=100, le=3600000)
    max_output_bytes: int = Field(default=8388608, ge=1024, le=67108864)
    max_concurrent: int = Field(default=1, ge=1, le=32)
    workspace: str = "./workspace"


class Provider(Defaults):
    adapter: Literal["omnirush", "opencode", "cline", "antigravity", "freebuff", "custom"]
    enabled: bool = False
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    prompt_mode: Literal["stdin", "arg"] | None = None
    output_mode: Literal["text", "jsonl"] | None = None
    acknowledge_agent_risk: bool = False


class ModelAlias(SettingsModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,199}$")
    provider: str
    upstream_model: str | None = None
    upstream_provider: str | None = None


class Configuration(SettingsModel):
    server: ServerSettings
    defaults: Defaults
    providers: dict[str, Provider]
    models: list[ModelAlias]

    def exposed_models(self) -> list[ModelAlias]:
        return [model for model in self.models if self.providers[model.provider].enabled]


def example_config() -> dict:
    return {
        "server": ServerSettings().model_dump(by_alias=True, exclude={"api_key"}),
        "defaults": Defaults().model_dump(by_alias=True),
        "providers": {
            "omnirush": {"adapter": "omnirush", "enabled": True, "command": "omnirush"},
            "opencode": {"adapter": "opencode", "enabled": False, "command": "opencode"},
            "cline": {"adapter": "cline", "enabled": False, "command": "cline"},
            "antigravity": {"adapter": "antigravity", "enabled": False, "command": "agy", "acknowledgeAgentRisk": False},
            "freebuff": {"adapter": "freebuff", "enabled": False, "command": "freebuff"},
        },
        "models": [{"id": name, "provider": name} for name in ("omnirush", "opencode", "cline", "antigravity", "freebuff")],
    }


def _no_nul(value: str, label: str, *, nonempty=True):
    if "\0" in value or (nonempty and not value.strip()):
        raise ValueError(f"{label} must be a {'nonempty ' if nonempty else ''}string without NUL bytes")


def validate_config(raw: dict, base_dir: Path | str = ".", env: dict | None = None) -> Configuration:
    env = os.environ if env is None else env
    base = Path(base_dir).resolve()
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a JSON object")
    # Merge defaults without changing the caller's object or persisted secrets.
    prepared = dict(raw)
    defaults = Defaults.model_validate(raw.get("defaults", {}))
    prepared["defaults"] = defaults
    providers = raw.get("providers")
    if not isinstance(providers, dict):
        raise ValueError("providers must be an object")
    prepared["providers"] = {}
    for name, settings in providers.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not isinstance(settings, dict):
            raise ValueError("Invalid provider name or settings")
        prepared["providers"][name] = {**defaults.model_dump(by_alias=True), **settings}
    prepared.setdefault("server", {})
    config = Configuration.model_validate(prepared)
    # Back-compat: existing wrapper.config.json files use GLOBAL_API_KEY;
    # QUANTA_API_KEY wins when both are set.
    secret = env.get(config.server.api_key_env, "") or env.get("GLOBAL_API_KEY", "") or config.server.api_key
    config.server.api_key = secret
    if len(config.server.api_key) < 24 or re.search(r"\s", config.server.api_key):
        raise ValueError(f"Set {config.server.api_key_env} or server.apiKey to a secret of at least 24 characters without whitespace")
    _no_nul(config.server.host, "server.host")
    for name, p in config.providers.items():
        _no_nul(p.command, f"{name}.command")
        _no_nul(p.workspace, f"{name}.workspace")
        for arg in p.args:
            _no_nul(arg, f"{name}.args", nonempty=False)
        for key, value in p.env.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"Invalid environment variable name for {name}")
            _no_nul(value, f"{name}.env", nonempty=False)
        p.workspace = str((base / p.workspace).resolve())
        if "/" in p.command or "\\" in p.command:
            p.command = str((base / p.command).resolve())
        if p.enabled and p.adapter in BRIDGE_NOTES:
            raise ValueError(f"{name}: {BRIDGE_NOTES[p.adapter]}")
        if p.enabled and p.adapter == "antigravity" and not p.acknowledge_agent_risk:
            raise ValueError("Configure agy deny permissions before enabling Antigravity, then set acknowledgeAgentRisk: true. See README.")
        if p.adapter == "custom":
            if p.prompt_mode is None or p.output_mode is None:
                raise ValueError(f"{name}: custom adapters require promptMode and outputMode")
            expected = 1 if p.prompt_mode == "arg" else 0
            if p.args.count("{prompt}") != expected or any("{prompt}" in a and a != "{prompt}" for a in p.args):
                raise ValueError(f"{name}: arg mode requires exactly one whole {{prompt}} argument; stdin mode requires none")
        elif p.prompt_mode or p.output_mode or any("{prompt}" in arg for arg in p.args):
            raise ValueError(f"{name}: promptMode, outputMode and {{prompt}} are custom-adapter options")
    ids = set()
    if not config.models:
        raise ValueError("Configure at least one model alias")
    for model in config.models:
        if model.id in ids or model.provider not in config.providers:
            raise ValueError("Duplicate model alias or unknown model provider")
        ids.add(model.id)
        p = config.providers[model.provider]
        if model.upstream_model is not None:
            _no_nul(model.upstream_model, "upstreamModel")
        if model.upstream_provider is not None:
            _no_nul(model.upstream_provider, "upstreamProvider")
            if p.adapter not in ("omnirush", "cline"):
                raise ValueError("upstreamProvider applies only to OmniRush and Cline")
        if p.adapter == "custom" and model.upstream_model and not any("{model}" in arg for arg in p.args):
            raise ValueError("Custom args must include {model} when upstreamModel is configured")
    return config


def load_config(filename: str | Path, overrides: dict | None = None) -> Configuration:
    filename = Path(filename).resolve()
    raw = json.loads(filename.read_text(encoding="utf-8-sig"))
    if overrides:
        raw["server"] = {**raw.get("server", {}), **overrides}
    try:
        return validate_config(raw, filename.parent)
    except ValidationError as error:
        # Never print Pydantic's raw input: it may include credentials.
        details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors(include_input=False))
        raise ValueError(f"Invalid configuration: {details}") from None


def initialize_config(filename: str | Path):
    filename = Path(filename).resolve()
    filename.parent.mkdir(parents=True, exist_ok=True)
    config = example_config()
    config["server"]["apiKey"] = "quanta_" + secrets.token_hex(32)
    descriptor = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
