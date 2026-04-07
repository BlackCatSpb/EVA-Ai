"""
Core subpackage for ЕВА.
Contains the central brain and coordination components.
"""
from .core_brain import CoreBrain
from .query_processor import QueryProcessor
from .response_generator import ResponseGenerator
from .event_system import EventSystem
from .background_coordinator import BackgroundCoordinator
from .token_processor import TokenProcessor
from .resource_manager import ResourceManager
from .config_manager import ConfigManager

__all__ = [
    "CoreBrain",
    "QueryProcessor",
    "ResponseGenerator",
    "EventSystem",
    "BackgroundCoordinator",
    "TokenProcessor",
    "ResourceManager",
    "ConfigManager",
]