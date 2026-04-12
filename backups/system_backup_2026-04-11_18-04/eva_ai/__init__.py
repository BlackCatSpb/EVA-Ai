"""EVA core package initializer.
Ensures subpackages can be imported via `python -m eva_ai.<module>`.
"""
__version__ = "0.1.0"

import eva_ai.memory
import eva_ai.core
import eva_ai.mlearning

__all__ = [
    "core",
    "mlearning",
    "memory",
    "knowledge",
    "tools",
    "gui",
    "adaptation",
    "contradiction",
    "learning",
    "ethics",
    "distributed",
    "websearch",
]
