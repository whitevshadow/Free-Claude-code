"""Centralized configuration using Pydantic Settings."""

import json
import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .nim import NimSettings


def _env_files() -> tuple[Path, ...]:
    """Return env file paths in priority order (later overrides earlier)."""
    files: list[Path] = [
        Path.home() / ".config" / "free-claude-code" / ".env",
        Path(".env"),
    ]
    if explicit := os.environ.get("FCC_ENV_FILE"):
        files.append(Path(explicit))
    return tuple(files)


def _configured_env_files(model_config: Mapping[str, Any]) -> tuple[Path, ...]:
    """Return the currently configured env files for Settings."""
    configured = model_config.get("env_file")
    if configured is None:
        return ()
    if isinstance(configured, (str, Path)):
        return (Path(configured),)
    return tuple(Path(item) for item in configured)


def _env_file_contains_key(path: Path, key: str) -> bool:
    """Check whether a dotenv-style file defines the given key."""
    if not path.is_file():
        return False

    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[7:].lstrip()
            name, sep, _value = stripped.partition("=")
            if sep and name.strip() == key:
                return True
    except OSError:
        return False

    return False


def _removed_env_var_message(model_config: Mapping[str, Any]) -> str | None:
    """Return a migration error for removed env vars, if present."""
    removed_key = "NIM_ENABLE_THINKING"
    replacement = "ENABLE_THINKING"

    if removed_key in os.environ:
        return (
            f"{removed_key} has been removed in this release. "
            f"Rename it to {replacement}."
        )

    for env_file in _configured_env_files(model_config):
        if _env_file_contains_key(env_file, removed_key):
            return (
                f"{removed_key} has been removed in this release. "
                f"Rename it to {replacement}. Found in {env_file}."
            )

    return None


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ==================== OpenRouter Config ====================
    open_router_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")

    # ==================== DeepSeek Config ====================
    deepseek_api_key: str = Field(default="", validation_alias="DEEPSEEK_API_KEY")

    # ==================== Messaging Platform Selection ====================
    # Valid: "telegram" | "discord"
    messaging_platform: str = Field(
        default="discord", validation_alias="MESSAGING_PLATFORM"
    )

    # ==================== NVIDIA NIM Config ====================
    nvidia_nim_api_key: str = ""

    # ==================== LM Studio Config ====================
    lm_studio_base_url: str = Field(
        default="http://localhost:1234/v1",
        validation_alias="LM_STUDIO_BASE_URL",
    )

    # ==================== Llama.cpp Config ====================
    llamacpp_base_url: str = Field(
        default="http://localhost:8080/v1",
        validation_alias="LLAMACPP_BASE_URL",
    )

    # ==================== Model ====================
    # All Claude model requests are mapped to this single model (fallback)
    # Format: provider_type/model/name
    model: str = "nvidia_nim/stepfun-ai/step-3.5-flash"

    # Per-model overrides (optional, falls back to MODEL)
    # Each can use a different provider
    model_opus: str | None = Field(default=None, validation_alias="MODEL_OPUS")
    model_sonnet: str | None = Field(default=None, validation_alias="MODEL_SONNET")
    model_haiku: str | None = Field(default=None, validation_alias="MODEL_HAIKU")

    # ==================== Per-Provider Proxy ====================
    nvidia_nim_proxy: str = Field(default="", validation_alias="NVIDIA_NIM_PROXY")
    open_router_proxy: str = Field(default="", validation_alias="OPENROUTER_PROXY")
    lmstudio_proxy: str = Field(default="", validation_alias="LMSTUDIO_PROXY")
    llamacpp_proxy: str = Field(default="", validation_alias="LLAMACPP_PROXY")

    # ==================== Provider Rate Limiting ====================
    provider_rate_limit: int = Field(default=40, validation_alias="PROVIDER_RATE_LIMIT")
    provider_rate_window: int = Field(
        default=60, validation_alias="PROVIDER_RATE_WINDOW"
    )
    provider_max_concurrency: int = Field(
        default=5, validation_alias="PROVIDER_MAX_CONCURRENCY"
    )
    enable_thinking: bool = Field(default=True, validation_alias="ENABLE_THINKING")

    # ==================== HTTP Client Timeouts ====================
    # Fallback/Default timeout if per-tier is unset
    http_read_timeout: float = Field(
        default=120.0, validation_alias="HTTP_READ_TIMEOUT"
    )
    # Tiered Timeouts
    http_read_timeout_opus: float = Field(
        default=300.0, validation_alias="HTTP_READ_TIMEOUT_OPUS"
    )
    http_read_timeout_sonnet: float = Field(
        default=150.0, validation_alias="HTTP_READ_TIMEOUT_SONNET"
    )
    http_read_timeout_haiku: float = Field(
        default=60.0, validation_alias="HTTP_READ_TIMEOUT_HAIKU"
    )

    http_write_timeout: float = Field(
        default=10.0, validation_alias="HTTP_WRITE_TIMEOUT"
    )
    http_connect_timeout: float = Field(
        default=2.0, validation_alias="HTTP_CONNECT_TIMEOUT"
    )

    # ==================== Fast Prefix Detection ====================
    fast_prefix_detection: bool = True

    # ==================== Fallback Routing ====================
    fallback_routing: bool = Field(default=True, validation_alias="FALLBACK_ROUTING")

    # ==================== Optimizations ====================
    enable_network_probe_mock: bool = True
    enable_title_generation_skip: bool = True
    enable_suggestion_mode_skip: bool = True
    enable_filepath_extraction_mock: bool = True

    # ==================== System Prompt Injections ====================
    enable_caveman: bool = Field(default=False, validation_alias="ENABLE_CAVEMAN")

    # ==================== NIM Settings ====================
    nim: NimSettings = Field(default_factory=NimSettings)

    # ==================== Voice Note Transcription ====================
    voice_note_enabled: bool = Field(
        default=True, validation_alias="VOICE_NOTE_ENABLED"
    )
    # Device: "cpu" | "cuda" | "nvidia_nim"
    # - "cpu"/"cuda": local Whisper (requires voice_local extra: uv sync --extra voice_local)
    # - "nvidia_nim": NVIDIA NIM Whisper API (requires voice extra: uv sync --extra voice)
    whisper_device: str = Field(default="cpu", validation_alias="WHISPER_DEVICE")
    # Whisper model ID or short name (for local Whisper) or NVIDIA NIM model (for nvidia_nim)
    # Local Whisper: "tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"
    # NVIDIA NIM: "nvidia/parakeet-ctc-1.1b-asr", "openai/whisper-large-v3", etc.
    whisper_model: str = Field(default="base", validation_alias="WHISPER_MODEL")
    # Hugging Face token for faster model downloads (optional, for local Whisper)
    hf_token: str = Field(default="", validation_alias="HF_TOKEN")

    # ==================== Bot Wrapper Config ====================
    telegram_bot_token: str | None = None
    allowed_telegram_user_id: str | None = None
    discord_bot_token: str | None = Field(
        default=None, validation_alias="DISCORD_BOT_TOKEN"
    )
    allowed_discord_channels: str | None = Field(
        default=None, validation_alias="ALLOWED_DISCORD_CHANNELS"
    )
    claude_workspace: str = "./agent_workspace"
    allowed_dir: str = ""

    # ==================== Server ====================
    host: str = "0.0.0.0"
    port: int = 8082
    log_file: str = "server.log"
    # Optional server API key to protect endpoints (Anthropic-style)
    # Set via env `ANTHROPIC_AUTH_TOKEN`. When empty, no auth is required.
    anthropic_auth_token: str = Field(
        default="", validation_alias="ANTHROPIC_AUTH_TOKEN"
    )
    # CORS allowed origins (comma-separated list, or "*" for all)
    # Empty = allow all origins
    cors_origins: list[str] | None = Field(
        default=None, validation_alias="CORS_ORIGINS"
    )

    @model_validator(mode="before")
    @classmethod
    def reject_removed_env_vars(cls, data: Any) -> Any:
        """Fail fast when removed environment variables are still configured."""
        if message := _removed_env_var_message(cls.model_config):
            raise ValueError(message)
        return data

    # Handle empty strings for optional string fields
    @field_validator(
        "telegram_bot_token",
        "allowed_telegram_user_id",
        "discord_bot_token",
        "allowed_discord_channels",
        mode="before",
    )
    @classmethod
    def parse_optional_str(cls, v: Any) -> Any:
        if v == "":
            return None
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> Any:
        """Parse CORS_ORIGINS from comma-separated string or list."""
        if v is None or v == "":
            return None
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("whisper_device")
    @classmethod
    def validate_whisper_device(cls, v: str) -> str:
        if v not in ("cpu", "cuda", "nvidia_nim"):
            raise ValueError(
                f"whisper_device must be 'cpu', 'cuda', or 'nvidia_nim', got {v!r}"
            )
        return v

    @field_validator("model", "model_opus", "model_sonnet", "model_haiku")
    @classmethod
    def validate_model_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        valid_providers = (
            "nvidia_nim",
            "open_router",
            "deepseek",
            "lmstudio",
            "llamacpp",
        )
        if "/" not in v:
            raise ValueError(
                f"Model must be prefixed with provider type. "
                f"Valid providers: {', '.join(valid_providers)}. "
                f"Format: provider_type/model/name"
            )
        provider = v.split("/", 1)[0]
        if provider not in valid_providers:
            raise ValueError(
                f"Invalid provider: '{provider}'. "
                f"Supported: 'nvidia_nim', 'open_router', 'deepseek', 'lmstudio', 'llamacpp'"
            )
        return v

    @model_validator(mode="after")
    def check_nvidia_nim_api_key(self) -> Settings:
        if (
            self.voice_note_enabled
            and self.whisper_device == "nvidia_nim"
            and not self.nvidia_nim_api_key.strip()
        ):
            raise ValueError(
                "NVIDIA_NIM_API_KEY is required when WHISPER_DEVICE is 'nvidia_nim'. "
                "Set it in your .env file."
            )
        return self

    @property
    def provider_type(self) -> str:
        """Extract provider type from the default model string."""
        return self.model.split("/", 1)[0]

    @property
    def model_name(self) -> str:
        """Extract the actual model name from the default model string."""
        return self.model.split("/", 1)[1]

    def _load_model_config(self) -> dict[str, Any]:
        """Load hierarchical model configuration from JSON file."""
        config_path = Path(__file__).parent.parent / "models_config.json"
        if not config_path.exists():
            logger.warning(
                f"models_config.json not found at {config_path}, using env vars only"
            )
            return {}

        try:
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load models_config.json: {e}")
            return {}

    def resolve_model(self, claude_model_name: str, attempt: int = 0) -> str:
        """Resolve a Claude model name to the configured provider/model string with hierarchical fallback.

        Classifies the incoming Claude model (opus/sonnet/haiku) and returns models in priority order.
        Uses models_config.json for hierarchical fallback, falls back to env vars if JSON not found.

        Args:
            claude_model_name: The Claude model name (e.g., "claude-opus-4-7")
            attempt: The attempt number for hierarchical fallback (0-based)

        Returns:
            provider/model string (e.g., "nvidia_nim/deepseek-ai/deepseek-v4-pro")
        """
        name_lower = claude_model_name.lower()

        # Determine tier
        if "opus" in name_lower:
            tier = "opus"
            env_fallback = self.model_opus
        elif "haiku" in name_lower:
            tier = "haiku"
            env_fallback = self.model_haiku
        elif "sonnet" in name_lower:
            tier = "sonnet"
            env_fallback = self.model_sonnet
        else:
            return self.model

        # Try hierarchical config from JSON
        config = self._load_model_config()
        if config and "model_tiers" in config:
            tier_models = config["model_tiers"].get(tier, [])
            if attempt < len(tier_models):
                model_info = tier_models[attempt]
                provider = model_info["provider"]
                model = model_info["model"]
                logger.debug(
                    f"MODEL RESOLUTION: tier={tier}, attempt={attempt}, "
                    f"using {model_info['name']} ({provider}/{model})"
                )
                return f"{provider}/{model}"

        # Fallback to env vars
        if attempt == 0 and env_fallback is not None:
            logger.debug(f"MODEL RESOLUTION: using env var for {tier}: {env_fallback}")
            return env_fallback

        # Final fallback
        logger.debug(f"MODEL RESOLUTION: using default model: {self.model}")
        return self.model

    def get_fallback_model(self, current_model: str, attempt: int = 0) -> str | None:
        """Return the next model in the hierarchical fallback chain.

        Args:
            current_model: The current model that failed
            attempt: The current attempt number

        Returns:
            Next model to try, or None if exhausted all options
        """
        # Parse the current model to determine tier
        lower = current_model.lower()

        if "opus" in lower or "deepseek-v4" in lower or "qwen3.5-397b" in lower or "mistral-large-3" in lower:
            tier = "opus"
        elif "sonnet" in lower or "devstral" in lower or "kimi-k2" in lower or "qwen3-coder" in lower:
            tier = "sonnet"
        elif "haiku" in lower or "glm" in lower or "mistral-medium" in lower or "nemotron" in lower:
            tier = "haiku"
        else:
            tier = None

        if tier is None:
            return None

        # Try next model in the hierarchical list
        config = self._load_model_config()
        if config and "model_tiers" in config:
            tier_models = config["model_tiers"].get(tier, [])
            next_attempt = attempt + 1

            if next_attempt < len(tier_models):
                model_info = tier_models[next_attempt]
                provider = model_info["provider"]
                model = model_info["model"]
                logger.info(
                    f"FALLBACK: {tier} tier attempt {next_attempt}: "
                    f"switching to {model_info['name']} ({provider}/{model})"
                )
                return f"{provider}/{model}"

        # If exhausted current tier, try next tier (Opus -> Sonnet -> Haiku)
        if tier == "opus":
            return self.model_sonnet or self.model_haiku or self.model
        if tier == "sonnet":
            return self.model_haiku or self.model

        return None

    @staticmethod
    def parse_provider_type(model_string: str) -> str:
        """Extract provider type from any 'provider/model' string."""
        return model_string.split("/", 1)[0]

    @staticmethod
    def parse_model_name(model_string: str) -> str:
        """Extract model name from any 'provider/model' string."""
        return model_string.split("/", 1)[1]

    model_config = SettingsConfigDict(
        env_file=_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
