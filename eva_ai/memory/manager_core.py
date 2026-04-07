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
    from eva_ai.knowledge.context_entity import EntityExtractor
except ImportError:
    EntityExtractor = None

try:
    from eva_ai.core.base_component import ComponentState
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

logger = logging.getLogger("eva_ai.memory.manager")


class MemoryManager:
    """Менеджер памяти для ЕВА, управляющий различными типами памяти и кэшированием."""

    def __init__(self, cache_dir: str, brain=None, knowledge_graph=None, event_bus=None, deferred_system=None, fractal_graph_v2=None):
        self.brain = brain
        self.knowledge_graph = knowledge_graph
        self.cache_dir = cache_dir
        self.event_bus = event_bus
        self.deferred_system = deferred_system
        self.fractal_graph_v2 = fractal_graph_v2
        self.config = {}
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
        self._setup_event_connections()
        self._initialize()

    def _setup_event_connections(self):
        """Подключение к EventBus и DeferredCommandSystem."""
        if self.event_bus:
            try:
                self.event_bus.subscribe("memory.optimized", self._on_memory_optimized)
                self.event_bus.subscribe("memory.warning", self._on_memory_warning)
                self.event_bus.subscribe("system.state_changed", self._on_system_state_changed)
                logger.info("MemoryManager подключён к EventBus")
            except Exception as e:
                logger.warning(f"Не удалось подключиться к EventBus: {e}")
        
        if self.deferred_system:
            try:
                from eva_ai.core.deferred_command_system import CommandPriority
                self.deferred_system.add_command(self._deferred_optimize, priority=CommandPriority.LOW)
                self.deferred_system.add_command(self._deferred_cleanup, priority=CommandPriority.NORMAL)
                logger.info("MemoryManager зарегистрировал команды в DeferredCommandSystem")
            except Exception as e:
                logger.warning(f"Не удалось зарегистрировать команды в DeferredCommandSystem: {e}")

    def _on_memory_optimized(self, event):
        logger.info("Событие memory.optimized получено")

    def _on_memory_warning(self, event):
        logger.warning(f"Событие memory.warning: {event}")

    def _on_system_state_changed(self, event):
        logger.info(f"Состояние системы изменилось: {event}")

    def _deferred_optimize(self):
        """Отложенная оптимизация памяти."""
        try:
            self.optimize()
            logger.info("Отложенная оптимизация памяти выполнена")
        except Exception as e:
            logger.error(f"Ошибка отложенной оптимизации: {e}")

    def _deferred_cleanup(self):
        """Отложенная очистка памяти."""
        try:
            self.clear_cache()
            logger.info("Отложенная очистка памяти выполнена")
        except Exception as e:
            logger.error(f"Ошибка отложенной очистки: {e}")

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

    def _init_fractal_graph(self):
        """Инициализировать fractal_graph_v2 если передан brain."""
        if self.fractal_graph_v2 is not None:
            logger.info("fractal_graph_v2 уже передан в MemoryManager")
            return

        if self.brain and hasattr(self.brain, 'fractal_graph_v2') and self.brain.fractal_graph_v2:
            self.fractal_graph_v2 = self.brain.fractal_graph_v2
            logger.info("fractal_graph_v2 получен из brain")
            return

        if self.config.get('fractal_graph_v2', {}).get('enabled', True):
            try:
                from eva_ai.memory.fractal_graph_v2 import FractalMemoryGraph
                self.fractal_graph_v2 = FractalMemoryGraph(
                    storage_dir=self.config.get('fractal_graph_v2', {}).get('storage_dir'),
                    embedding_device=self.config.get('fractal_graph_v2', {}).get('embedding_device', 'cuda')
                )
                logger.info("fractal_graph_v2 инициализирован в MemoryManager")
            except Exception as e:
                logger.warning(f"Не удалось инициализировать fractal_graph_v2: {e}")

    def search_knowledge(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Семантический поиск по знаниям через fractal_graph_v2."""
        if not self.fractal_graph_v2:
            return []

        try:
            return self.fractal_graph_v2.semantic_search(query, top_k=top_k, min_level=1)
        except Exception as e:
            logger.error(f"Ошибка поиска в fractal_graph_v2: {e}")
            return []

    def add_fact(self, subject: str, relation: str, object_: str, confidence: float = 0.5) -> Optional[Tuple]:
        """Добавить факт в fractal_graph_v2."""
        if not self.fractal_graph_v2:
            return None

        try:
            return self.fractal_graph_v2.add_knowledge(
                subject=subject,
                relation=relation,
                object_=object_,
                confidence=confidence
            )
        except Exception as e:
            logger.error(f"Ошибка добавления факта: {e}")
            return None

    def get_graph_stats(self) -> Dict[str, Any]:
        """Получить статистику fractal_graph_v2."""
        if not self.fractal_graph_v2:
            return {"available": False}

        try:
            stats = self.fractal_graph_v2.get_stats()
            stats["available"] = True
            return stats
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {"available": False, "error": str(e)}

    def verify_knowledge(self, knowledge: str) -> Dict[str, Any]:
        """Проверить знание на противоречия через self_dialogue."""
        if not self.fractal_graph_v2:
            return {"available": False}

        try:
            return self.fractal_graph_v2.self_dialogue(knowledge)
        except Exception as e:
            logger.error(f"Ошибка верификации знания: {e}")
            return {"confirmed": False, "error": str(e)}
