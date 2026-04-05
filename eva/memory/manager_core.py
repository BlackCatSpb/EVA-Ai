"""Core module for MemoryManager - main class, initialization, lifecycle."""
import os
import logging
import json
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Iterable
from pathlib import Path

try:
    import psutil
except Exception:
    psutil = None

try:
    from eva.knowledge.context_entity import EntityExtractor
except ImportError:
    EntityExtractor = None

try:
    from eva.core.base_component import ComponentState
except ImportError:
    class ComponentState:
        UNINITIALIZED = "uninitialized"
        INITIALIZING = "initializing"
        READY = "ready"
        STARTING = "starting"
        RUNNING = "running"
        STOPPING = "stopping"
        STOPPED = "stopped"
        ERROR = "error"

logger = logging.getLogger("eva.memory.manager")


class MemoryManager:
    """Менеджер памяти для ЕВА, управляющий различными типами памяти и кэшированием."""

    def __init__(self, cache_dir: str, brain=None, knowledge_graph=None):
        self.brain = brain
        self.knowledge_graph = knowledge_graph
        self.cache_dir = cache_dir
        self.initialized = False
        self.running = False
        self.stop_event = threading.Event()

        os.makedirs(self.cache_dir, exist_ok=True)

        self.working_memory_file = os.path.join(self.cache_dir, "working_memory.json")
        self.semantic_memory_file = os.path.join(self.cache_dir, "semantic_memory.json")
        self.episodic_memory_file = os.path.join(self.cache_dir, "episodic_memory.json")
        self.user_profiles_file = os.path.join(self.cache_dir, "user_profiles.json")

        self.working_memory = {}
        self.semantic_memory = {}
        self.episodic_memory = []
        self.user_profiles = {}
        self.hybrid_cache = None

        self.max_working_memory = 1000
        self.max_semantic_memory = 5000
        self.max_episodic_memory = 2000
        self.max_user_profiles = 100

        self.entity_extractor = EntityExtractor() if EntityExtractor else None

        self.memory_locks = {
            "working": threading.Lock(),
            "semantic": threading.Lock(),
            "episodic": threading.Lock(),
            "user_profiles": threading.Lock()
        }

        self.error = None
        self._initialize()

    def get_hybrid_cache(self):
        if not hasattr(self, 'hybrid_cache') or not self.hybrid_cache:
            if self.brain and getattr(self.brain, 'hybrid_cache', None):
                self.hybrid_cache = self.brain.hybrid_cache
                logger.debug("Используем единый HybridTokenCache из brain")
                return self.hybrid_cache

            try:
                from .hybrid_token_cache import get_shared_cache
                self.hybrid_cache = get_shared_cache(self.brain, "memory_manager")
                logger.info("Гибридный кэш успешно инициализирован через get_shared_cache")
            except ImportError as e:
                logger.error(f"Не удалось импортировать get_shared_cache: {e}")
                raise RuntimeError("Не удалось загрузить модуль гибридного кэша")
            except Exception as e:
                logger.error(f"Ошибка инициализации гибридного кэша: {e}")
                raise RuntimeError(f"Ошибка инициализации гибридного кэша: {e}")

        return self.hybrid_cache

    def get_state(self) -> ComponentState:
        if not self.initialized:
            if self.error:
                return ComponentState.ERROR
            return ComponentState.INITIALIZING

        if not self.running:
            return ComponentState.STOPPED

        try:
            if not os.path.exists(self.working_memory_file):
                return ComponentState.ERROR

            if self.knowledge_graph is not None and hasattr(self.knowledge_graph, 'is_initialized') and not self.knowledge_graph.is_initialized():
                return ComponentState.ERROR

            return ComponentState.READY
        except Exception:
            return ComponentState.ERROR

    def _init_hybrid_cache(self):
        try:
            try:
                from .hybrid_token_cache import HybridTokenCache, get_shared_cache
                logger.debug("HybridTokenCache импортирован успешно")
            except ImportError:
                logger.warning("HybridTokenCache недоступен, кэширование будет ограничено")
                return

            if self.brain and hasattr(self.brain, 'hybrid_cache') and self.brain.hybrid_cache:
                self.hybrid_cache = self.brain.hybrid_cache
                logger.debug("Используем единый HybridTokenCache из brain")
            else:
                brain_obj = self.brain
                if brain_obj is None or not hasattr(brain_obj, "cache_dir") or not brain_obj.cache_dir:
                    class _BrainShim:
                        def __init__(self, cache_dir: str):
                            self.cache_dir = cache_dir
                            self.config = {}
                    safe_cache_dir = self.cache_dir if os.path.isabs(self.cache_dir) else os.path.join(os.getcwd(), self.cache_dir)
                    os.makedirs(safe_cache_dir, exist_ok=True)
                    brain_obj = _BrainShim(safe_cache_dir)

                self.hybrid_cache = get_shared_cache(brain_obj, "memory_manager")
                logger.info(
                    f"Гибридный кэш инициализирован через get_shared_cache: memory_manager"
                )

        except Exception as e:
            logger.error(f"Ошибка инициализации гибридного кэша: {e}", exc_info=True)

    def _initialize(self):
        try:
            from .manager_operations import (
                _load_working_memory, _load_semantic_memory,
                _load_episodic_memory, _load_user_profiles
            )
            _load_working_memory(self)
            _load_semantic_memory(self)
            _load_episodic_memory(self)
            _load_user_profiles(self)

            self._init_hybrid_cache()
            self.initialized = True
        except Exception as e:
            logger.error(f"Ошибка инициализации менеджера памяти: {e}", exc_info=True)

    def initialize(self) -> bool:
        if self.initialized:
            return True
        try:
            self._initialize()
            return self.initialized
        except Exception as e:
            logger.error(f"Ошибка инициализации MemoryManager: {e}", exc_info=True)
            return False

    def start(self):
        if not self.initialized:
            logger.error("Невозможно запустить неинициализированный менеджер памяти")
            return False
        self.running = True
        self.stop_event.clear()
        logger.info("Менеджер памяти запущен")
        return True

    def stop(self):
        self.running = False
        self.stop_event.set()
        logger.info("Менеджер памяти остановлен")

    def get_memory_status(self) -> Dict[str, Any]:
        return {
            "working_memory_size": len(self.working_memory),
            "semantic_memory_size": len(self.semantic_memory),
        }
