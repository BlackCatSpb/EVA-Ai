"""
Configuration loading, secret masking, and system info for CoreBrain.
"""
import os
import sys
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("eva.core_brain")
query_logger = logging.getLogger("eva.core_brain.query_processing")

_SENSITIVE_PATTERNS = {'secret', 'password', 'api_key', 'token', 'credentials', 'auth', 'key', 'private'}


def load_brain_config(config_path: str = None) -> Dict[str, Any]:
    """Loads configuration from brain_config.json."""
    if config_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(project_root, "brain_config.json")

    if not os.path.exists(config_path):
        logger.error(f"brain_config.json not found at {config_path}")
        raise FileNotFoundError(
            f"brain_config.json not found. Expected path: {config_path}"
        )

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        query_logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading {config_path}: {e}")
        raise


def mask_secrets(config: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a copy of config with sensitive values masked."""
    return {
        k: '***' if k.lower() in _SENSITIVE_PATTERNS else v
        for k, v in config.items()
    }


def get_system_info(brain) -> Dict[str, Any]:
    """Returns system information for logging."""
    import psutil

    system_info = {
        "python_version": sys.version.split()[0],
        "os": os.name,
        "platform": sys.platform,
        "cpu_count": os.cpu_count(),
        "memory": f"{psutil.virtual_memory().percent}%"
    }

    if hasattr(brain, 'resource_manager') and brain.resource_manager:
        try:
            system_info.update(brain.resource_manager.get_system_info())
        except Exception:
            pass

    if hasattr(brain, 'state_manager') and brain.state_manager and hasattr(brain.state_manager, 'get_state'):
        state = brain.state_manager.get_state()
        if hasattr(state, 'value'):
            system_info["system_state"] = state.value

    return system_info


class ConfigMixin:
    """Mixin providing config helpers to CoreBrain."""

    def _load_brain_config(self) -> Dict[str, Any]:
        """Loads configuration from brain_config.json."""
        return load_brain_config()

    def _mask_secrets(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns config with sensitive values masked."""
        return mask_secrets(config)

    def _get_system_info(self) -> Dict[str, Any]:
        """Returns system information for logging."""
        return get_system_info(self)
