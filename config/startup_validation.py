import json
from pathlib import Path

from loguru import logger


class ConfigError(Exception):
    pass


def validate_startup():
    """Run on application startup - fail fast if misconfigured"""
    errors = []

    # 1. Validate NVIDIA_NIM_API_KEY
    from .settings import get_settings

    settings = get_settings()
    if not settings.nvidia_nim_api_key or not settings.nvidia_nim_api_key.strip():
        errors.append("NVIDIA_NIM_API_KEY not set in .env")

    # 2. Validate models_config.json exists and is valid JSON
    config_path = Path(__file__).parent.parent / "models_config.json"
    config = None
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        errors.append("models_config.json not found - using .env fallback")
    except json.JSONDecodeError as e:
        errors.append(f"models_config.json invalid JSON: {e}")

    # 3. Validate each tier has at least one model
    if config and "model_tiers" in config:
        for tier in ["opus", "sonnet", "haiku"]:
            models = config["model_tiers"].get(tier, [])
            if not models:
                errors.append(
                    f"No models configured for tier '{tier}' in models_config.json"
                )

    # 4. Log warnings or raise on critical errors
    if errors:
        for err in errors:
            logger.error(f"STARTUP VALIDATION: {err}")
        # Raise if API key is missing
        if any("NVIDIA_NIM_API_KEY" in e for e in errors):
            raise ConfigError("Critical: NVIDIA_NIM_API_KEY missing")
    else:
        logger.info("✓ Startup validation passed")
