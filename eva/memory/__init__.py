"""
Memory subpackage for ЕВА.
Contains memory management and caching components.
"""

def __getattr__(name):
    if name == "MemoryManager":
        from .manager_core import MemoryManager
        return MemoryManager
    if name == "HybridTokenCache":
        from .hybrid_token_cache import HybridTokenCache
        return HybridTokenCache
    if name == "WorkingMemory":
        from .working_memory import WorkingMemory
        return WorkingMemory
    if name == "LongTermMemory":
        from .long_term_memory import LongTermMemory
        return LongTermMemory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return ["MemoryManager", "HybridTokenCache", "WorkingMemory", "LongTermMemory"]


__all__ = [
    "MemoryManager",
    "HybridTokenCache",
    "WorkingMemory",
    "LongTermMemory",
]
