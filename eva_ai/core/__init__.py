"""
Core subpackage for ЕВА.
Contains the central brain and coordination components.
"""
from .core_brain import CoreBrain, get_core_instance
from .query_processor import QueryProcessor
from .response_generator import ResponseGenerator
from .event_system import EventSystem
from .token_processor import TokenProcessor
from .resource_manager import ResourceManager
from .config_manager import ConfigManager
try:
    from .unified_generator import UnifiedGenerator, create_unified_generator, ModelType
except ImportError:
    UnifiedGenerator = None
    create_unified_generator = None
    ModelType = None
from .pipeline_adapter import PipelineAdapter, create_pipeline_adapter
from .hybrid_dialog_manager import (
    HybridKnowledgeDialogManager,
    create_hybrid_dialog_manager,
    KnowledgeContext,
    GenerationResult,
    DialogMessage,
)

__all__ = [
    "CoreBrain",
    "get_core_instance",
    "QueryProcessor",
    "ResponseGenerator",
    "EventSystem",
    "TokenProcessor",
    "ResourceManager",
    "ConfigManager",
    "UnifiedGenerator",
    "create_unified_generator",
    "ModelType",
    "PipelineAdapter",
    "create_pipeline_adapter",
    "HybridKnowledgeDialogManager",
    "create_hybrid_dialog_manager",
    "KnowledgeContext",
    "GenerationResult",
    "DialogMessage",
]
